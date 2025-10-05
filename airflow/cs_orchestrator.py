# =============================================================================
# RUNBOOK — CS Module Airflow DAGs
#
# DAGS
# ----
# - cs_module_static_load  (manual/one-off)
#   * Loads static objects (missions/resources/recommendations) into CS.
#
# - cs_module_align_no_learn_no_intervene  (manual/one-off; optional)
#   * Replays historical updates WITHOUT learning and WITHOUT intervention.
#   * Uses FROZEN time. Rarely needed; skip for most rebuilds.
#
# - cs_module_backfill_learn_no_intervene  (hourly catch-up; FROZEN; catchup=True)
#   * Replays hourly windows from NOW - CATCH_UP_LEARNING_DAYS to NOW-1h.
#   * Learns from historical feedback/events (is_learning=True).
#   * Does NOT generate historical plans (is_intervention=False).
#
# - cs_module_live_learn_intervene  (hourly; REAL; catchup=False)
#   * From “now” forward: learns + generates plans on new mission selections.
#
# QUICK STARTS
# ------------
# From-scratch rebuild:
#   1) Trigger cs_module_static_load.
#   2) Set CATCH_UP_LEARNING_DAYS to cover the whole study period.
#   3) Toggle ON cs_module_backfill_learn_no_intervene; let it finish.
#   4) Toggle OFF cs_module_backfill_learn_no_intervene.
#   5) Toggle ON cs_module_live_learn_intervene.
#
# Routine recovery (missed N hours):
#   1) Set CATCH_UP_LEARNING_DAYS ≥ N (rounded up).
#   2) Toggle ON cs_module_backfill_learn_no_intervene; let it finish.
#   3) Toggle OFF backfill; keep cs_module_live_learn_intervene ON.
#
# Crash matrix (summary):
#   - CS/EUT/Airflow outage: restore service → backfill (learn/no_intervene) over gap
#     → resume live (learn/intervene).
#
# TUNABLES
# --------
# - CATCH_UP_LEARNING_DAYS: lookback window for backfill learning.
# - CS_STUDY_START_UTC: optional ISO; alignment floor if you use the align DAG.
# - BINDER_PROC_CAP: ProcessBinder cache cap (default 200000).
# =============================================================================

from __future__ import annotations
from functools import partial, lru_cache
from datetime import datetime, timedelta, timezone
import json, logging, requests, time

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.utils import timezone as aftz

# ────────────────────────────────────────────────────────────────────────
# Config & helpers (unchanged)
# ────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logging.Formatter.converter = time.gmtime  # enforce UTC

CATCH_UP_LEARNING_DAYS = int(Variable.get("CS_CATCH_UP_LEARNING_DAYS", default_var="1"))
STUDY_START = Variable.get("CS_STUDY_START_UTC", default_var="2025-06-01T00:00:00Z")


@lru_cache(maxsize=1)
def get_cfg():
    cs_base = Variable.get("CS_BASE_URL", default_var="http://cs_module:8000")
    eut_base = Variable.get("EUT_BASE_URL", default_var="")
    token = Variable.get("EUT_TOKEN", default_var="")
    return {
        "CS_BASE": cs_base,
        "EUT_BASE": eut_base,
        "HEADERS_EUT": {"Authorization": f"Basic {token}"},
        "HEADERS_CS": {"Content-Type": "application/json"},
    }


def _eut_get(path: str, params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['EUT_BASE']}{path}"
    logging.info("GET %s  params=%s", url, params or {})
    r = requests.get(url, headers=cfg["HEADERS_EUT"], params=params or {})
    if r.status_code != 200:
        raise RuntimeError(f"EUT GET {path} failed: {r.status_code} {r.text}")
    return r.json()


def _cs_post(endpoint: str, payload, extra_params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['CS_BASE']}{endpoint}"
    if extra_params:
        payload = {"data": payload, **extra_params}
    logging.info("POST %s  bytes=%d", url, len(json.dumps(payload)))
    r = requests.post(url, json=payload, headers=cfg["HEADERS_CS"])
    if r.status_code not in (200, 201):
        raise RuntimeError(f"CS POST {endpoint} failed: {r.status_code} {r.text}")


def _cs_get(path: str, params: dict | None = None):
    cfg = get_cfg()
    url = f"{cfg['CS_BASE']}{path}"
    logging.info("GET %s  params=%s", url, params or {})
    r = requests.get(url, headers=cfg["HEADERS_CS"], params=params or {})
    if r.status_code != 200:
        raise RuntimeError(f"CS GET {path} failed: {r.status_code} {r.text}")
    return r.json()


def utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        raise ValueError("Naïve datetime received – timezone info required")
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _prepare_cs_clock(window_end_utc: datetime, *, time_mode: str):
    time_mode = (time_mode or "REAL").upper()
    _cs_post("/set_time_mode", {"mode": time_mode})
    if time_mode == "FROZEN":
        _cs_post("/set_current_time", utc_iso(window_end_utc))


# ────────────────────────────────────────────────────────────────────────
# Tasks
# ────────────────────────────────────────────────────────────────────────


def fetch_static_objects():
    mappings = {
        "/recommendations": "/internal/recommendations?resources=false",
        "/resources": "/internal/recommendations?resources=true",
        "/missions": "/missions",
    }
    for endpoint, path in mappings.items():
        data = _eut_get(path)
        if data:
            _cs_post(endpoint, data)
        else:
            logging.warning("No data for %s", endpoint)


def _fetch_updates_window(start_dt: datetime, end_dt: datetime, *, is_learning: bool, is_intervention: bool):
    params = {"start_date": utc_iso(start_dt), "end_date": utc_iso(end_dt)}
    data = _eut_get("/updates", params=params)
    if data:
        _cs_post("/updates", data, extra_params={"is_learning": is_learning, "is_intervention": is_intervention})
    else:
        logging.info("No updates for window %s → %s", params["start_date"], params["end_date"])


def align_past_updates(*, time_mode: str):
    cursor = datetime.fromisoformat(STUDY_START.replace("Z", "+00:00"))
    stop_at = (datetime.now(timezone.utc) - timedelta(days=CATCH_UP_LEARNING_DAYS)).replace(
        minute=0, second=0, microsecond=0
    )
    while cursor < stop_at:
        window_end = min(cursor + timedelta(days=1), stop_at)
        _prepare_cs_clock(window_end, time_mode=time_mode)
        _fetch_updates_window(cursor, window_end, is_learning=False, is_intervention=False)
        cursor = window_end


def hourly_update(*, time_mode: str, is_learning: bool, is_intervention: bool, **context):
    end = context["logical_date"].astimezone(timezone.utc)
    start = (end - timedelta(hours=1)).astimezone(timezone.utc)
    _prepare_cs_clock(end, time_mode=time_mode)
    logging.info("INTERVAL %s → %s", start.isoformat(), end.isoformat())
    _fetch_updates_window(start, end, is_learning=is_learning, is_intervention=is_intervention)


def fetch_selected_contents(*, time_mode: str, **context):
    end = context["logical_date"].astimezone(timezone.utc)
    start = (end - timedelta(hours=1)).astimezone(timezone.utc)
    _prepare_cs_clock(end, time_mode=time_mode)
    params = {"start_time": utc_iso(start), "end_time": utc_iso(end)}
    data = _cs_get("/selected_contents", params=params)
    logging.info("Retrieved selected_contents payload:\n%s", json.dumps(data, indent=2))


# ────────────────────────────────────────────────────────────────────────
# DAG 0 — Static load only (manual one-off)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_static_load",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "static", "no_learn", "no_intervene"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_static:
    PythonOperator(task_id="fetch_static_objects", python_callable=fetch_static_objects)

# ────────────────────────────────────────────────────────────────────────
# DAG 1 — Alignment (optional; manual one-off; no learn / no intervene)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_align_no_learn_no_intervene",
    schedule=None,
    catchup=False,
    is_paused_upon_creation=True,
    tags=["cs_module", "align", "frozen", "no_learn", "no_intervene"],
    default_args=dict(owner="POLIMI", start_date=datetime(2025, 1, 1, tzinfo=timezone.utc), retries=0),
) as dag_align:
    t_align = PythonOperator(
        task_id="align_past_updates", python_callable=partial(align_past_updates, time_mode="FROZEN")
    )

# ────────────────────────────────────────────────────────────────────────
# DAG 2 — Backfill (learn / no intervene; hourly; FROZEN; catchup=True)
# ────────────────────────────────────────────────────────────────────────
NOW = aftz.utcnow().replace(minute=0, second=0, microsecond=0)
BACKFILL_START = NOW - timedelta(days=CATCH_UP_LEARNING_DAYS)
BACKFILL_END = NOW - timedelta(hours=1)

with DAG(
    dag_id="cs_module_backfill_learn_no_intervene",
    start_date=BACKFILL_START,
    end_date=BACKFILL_END,
    schedule="0 * * * *",
    catchup=True,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["cs_module", "backfill", "frozen", "learn", "no_intervene"],
    default_args=dict(owner="POLIMI", retries=0, retry_delay=timedelta(minutes=5)),
) as dag_backfill:
    t_updates = PythonOperator(
        task_id="hourly_update",
        python_callable=partial(hourly_update, time_mode="FROZEN", is_learning=True, is_intervention=False),
        depends_on_past=True,
        max_active_tis_per_dag=1,
    )
    t_selected = PythonOperator(
        task_id="fetch_selected_contents",
        python_callable=partial(fetch_selected_contents, time_mode="FROZEN"),
        depends_on_past=True,
    )
    t_updates >> t_selected

# ────────────────────────────────────────────────────────────────────────
# DAG 3 — Live (learn / intervene; hourly; REAL; catchup=False)
# ────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="cs_module_live_learn_intervene",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    schedule="0 * * * *",
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["cs_module", "live", "real", "learn", "intervene"],
    default_args=dict(owner="POLIMI", retries=3, retry_delay=timedelta(minutes=5)),
) as dag_live:
    t_updates = PythonOperator(
        task_id="hourly_update",
        python_callable=partial(hourly_update, time_mode="REAL", is_learning=True, is_intervention=True),
        depends_on_past=False,
        max_active_tis_per_dag=1,
    )
    t_selected = PythonOperator(
        task_id="fetch_selected_contents",
        python_callable=partial(fetch_selected_contents, time_mode="REAL"),
        depends_on_past=False,
    )
    t_updates >> t_selected
