import time
import requests
import logging
import os
from datetime import datetime, timedelta, timezone
import random
from orchestrator.config_generator import generate_preferences, generate_config, EXPERIMENTS_TO_RUN
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

MAB_TYPES = [
    "Random",
    "Learning",
    "Optimal",
]

HOURS_PER_INTERVENTION = int(24 * 7 * 12)  # 12 simulated weeks/user

HOST_CONFIG = os.path.join(os.path.dirname(__file__), "..", "generated_configs", "config.py")
if not os.path.exists(HOST_CONFIG):
    os.makedirs(os.path.dirname(HOST_CONFIG), exist_ok=True)
    # create an empty stub so Docker can mount it
    with open(HOST_CONFIG, "w") as f:
        f.write("# stub, will be overwritten by generate_config()\n")


# ---- Configuration ----
TIME_SERVICE_URL = "http://time_service:5000"
VU_API_URL = "http://vu_api:5000"
CS_MODULE_URL = "http://cs_module:8000"
OMI_MODULE_URL = "http://omi_module:5000"

CONTAINER_NAMES = [
    "ibechange_recommendation_selection_module-time_service-1",
    "ibechange_recommendation_selection_module-vu_api-1",
    "ibechange_recommendation_selection_module-cs_module-1",
    "ibechange_recommendation_selection_module-omi_module-1",
]

# --- add near the top, with other constants ---
MODULES_WITH_TIME = [
    ("VU", f"{VU_API_URL}/set_time_mode"),
    ("CS", f"{CS_MODULE_URL}/set_time_mode"),
    ("OMI", f"{OMI_MODULE_URL}/set_time_mode"),
]


def set_modules_time_mode(mode: str = "FROZEN"):
    """Tell each module to use the requested time mode."""
    for short, endpoint in MODULES_WITH_TIME:
        post_and_wait(endpoint, {"mode": mode}, label=f"{short} time mode")


def restart_containers(container_names):
    for name in container_names:
        # print(f"🔄 Restarting container: {name}")
        os.system(f"docker restart {name}")
    # print("✅ All containers restarted.\n")


def seed_vu_for_experiment(exp_idx: int):
    post_and_wait(f"{VU_API_URL}/seed", {"seed": int(exp_idx)}, label="VU seed")


def post_and_wait(endpoint, data, label="data", timeout=300):
    # print(f"\n📤 posting to {label} ...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.post(endpoint, json=data, timeout=timeout)
            response.raise_for_status()
            try:
                result = response.json()
            except ValueError:
                result = None
            if result:
                # print(f"✅ Post to {label} succeeded")
                return result
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    # print(f"❌ Timeout while posting {label} to {endpoint}")
    return None


def fetch_and_wait(endpoint, label="data", params=None, timeout=300):
    """
    GET `endpoint` until it succeeds or times-out.

    Parameters
    ----------
    endpoint : str
    label    : str   – pretty-name for log lines
    params   : dict  – query-string parameters (default None)
    timeout  : int   – overall max wait in seconds
    """
    # print(f"\n🔍 Fetching {label} from {endpoint} …")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # ← pass params straight to requests
            response = requests.get(endpoint, params=params, timeout=timeout)
            response.raise_for_status()

            # print(f"✅ Fetched {label} successfully")
            data = response.json()
            # print(f"📦 {label} content: {data!s}")
            return data  # (no need to call .json() twice)

        except requests.exceptions.RequestException:
            pass

        time.sleep(0.5)

    # print(f"❌ Timeout while fetching {label} from {endpoint}")
    return None


def run_hourly_update(last_printed_day):
    current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="current time")
    if not current_time_data:
        return last_printed_day

    now = datetime.fromisoformat(current_time_data["now"])  # aware → already includes +00:00Z
    prev = now - timedelta(hours=1)

    current_day = datetime.fromisoformat(current_time_data["now"]).date()
    if current_day != last_printed_day:
        # print(f"\n# =======================\n# Current_time: {current_time_data['now']}\n# =======================")
        last_printed_day = current_day

    updates = fetch_and_wait(f"{VU_API_URL}/updates", label="VU updates")
    if not updates:
        return last_printed_day

    post_and_wait(f"{CS_MODULE_URL}/updates", updates, label="CS updates")
    post_and_wait(f"{OMI_MODULE_URL}/updates", updates, label="OMI updates")

    params = {
        "start_time": prev.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "end_time": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    selected_contents = fetch_and_wait(f"{CS_MODULE_URL}/selected_contents", label="Selected contents", params=params)

    if selected_contents:
        post_and_wait(f"{OMI_MODULE_URL}/selected_contents", selected_contents, label="OMI selected contents")

    recommendation_plans = fetch_and_wait(f"{OMI_MODULE_URL}/recommendation_plans", label="Recommendation plans")
    if recommendation_plans:
        post_and_wait(f"{CS_MODULE_URL}/recommendation_plans", recommendation_plans, label="CS recommendation plans")
        post_and_wait(f"{VU_API_URL}/recommendation_plans", recommendation_plans, label="VU recommendation plans")

    advance_result = post_and_wait(f"{TIME_SERVICE_URL}/advance", {"hours": 1}, label="advance time")
    if not advance_result:
        return last_printed_day

    current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="current time after advancing")
    if not current_time_data:
        return last_printed_day

    current_time = current_time_data["now"]
    post_and_wait(f"{VU_API_URL}/set_current_time", current_time, label="VU set_current_time")
    post_and_wait(f"{CS_MODULE_URL}/set_current_time", current_time, label="CS set_current_time")
    post_and_wait(f"{OMI_MODULE_URL}/set_current_time", current_time, label="OMI set_current_time")

    return last_printed_day


def initialize_modules(current_time):
    post_and_wait(f"{VU_API_URL}/set_start_time", current_time, label="VU start time")
    post_and_wait(f"{CS_MODULE_URL}/set_start_time", current_time, label="CS start time")
    post_and_wait(f"{OMI_MODULE_URL}/set_start_time", current_time, label="OMI start time")

    recommendations = fetch_and_wait(f"{VU_API_URL}/recommendations", label="recommendations")
    resources = fetch_and_wait(f"{VU_API_URL}/resources", label="resources")
    missions = fetch_and_wait(f"{VU_API_URL}/missions", label="missions")

    post_and_wait(f"{CS_MODULE_URL}/recommendations", recommendations, label="CS recommendations")
    post_and_wait(f"{CS_MODULE_URL}/resources", resources, label="CS resources")
    post_and_wait(f"{CS_MODULE_URL}/missions", missions, label="CS missions")

    post_and_wait(f"{OMI_MODULE_URL}/recommendations", recommendations, label="OMI recommendations")
    post_and_wait(f"{OMI_MODULE_URL}/resources", resources, label="OMI resources")
    post_and_wait(f"{OMI_MODULE_URL}/missions", missions, label="OMI missions")


if __name__ == "__main__":
    # print("🟢 Orchestrator started...\n")
    total_hours = EXPERIMENTS_TO_RUN * len(MAB_TYPES) * HOURS_PER_INTERVENTION
    progress = tqdm(total=total_hours, desc="Simulated hours", unit="h", dynamic_ncols=True)

    for exp in range(EXPERIMENTS_TO_RUN):
        random.seed(exp)
        prefs, theta = generate_preferences()

        for mab_type in MAB_TYPES:
            print(f"🔄 Experiment {exp + 1}/{EXPERIMENTS_TO_RUN} — running with {mab_type}")

            generate_config(prefs, theta, mab_type)
            time.sleep(1)
            restart_containers(CONTAINER_NAMES)
            time.sleep(2)
            set_modules_time_mode("FROZEN")

            seed_vu_for_experiment(exp)

            # Initial time setup
            current_time_data = fetch_and_wait(f"{TIME_SERVICE_URL}/get_time", label="initial time")
            current_time = current_time_data["now"]
            time.sleep(1)
            initialize_modules(current_time)

            # Run simulation loop
            last_printed_day = None
            for _ in range(HOURS_PER_INTERVENTION):
                last_printed_day = run_hourly_update(last_printed_day)
                # time.sleep(0.2)
                progress.update(1)

            # print(f"✅  Completed run with {mab_type}\n")

        # print(f"\n🏁 Experiment {exp + 1} complete.")

    progress.close()
    print("🎉 All experiments complete.")
