import os
import numpy as np
from datetime import datetime
import psycopg2
import time
import json
import logging
from psycopg2.extras import Json


def sanitize_for_json(obj):
    """Recursively convert data into JSON-serializable format."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


class DataStorage:
    """
    DataStorage handles storing various module outputs into a structured PostgreSQL schema.
    Connection parameters are read from environment variables:
      DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
    """

    def __init__(self):
        # DB connection settings
        self.db_params = {
            "dbname": os.getenv("DB_NAME"),
            "user": os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host": os.getenv("DB_HOST"),
            "port": "5432",
        }
        # Ensure tables exist
        self._ensure_tables()

        # Create a single run identifier
        self.run_id = self._create_run()

    def _get_conn(self, retries=5, delay=3):
        for i in range(retries):
            try:
                conn = psycopg2.connect(**self.db_params)
                with conn.cursor() as cur:
                    cur.execute("SET TIME ZONE 'UTC';")
                return conn
            except psycopg2.OperationalError as e:
                print(f"[DB INIT] Attempt {i + 1} failed: {e}")
                time.sleep(delay)
        raise Exception("Failed to connect to the database after several retries")

    def _ensure_tables(self):
        """
        Create all required tables with structured schemas.
        """

        table_creations = [
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id        SERIAL PRIMARY KEY,
                created_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                description   TEXT
            );
            """,
            # Intervention runs
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,        -- e.g. "GaussianInverseGammaTS"
                initial_params      JSONB                       -- store INTERVENTION_MAB_initial_params
            );
            """,
            # Recommendation runs
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # Resource runs
            """
            CREATE TABLE IF NOT EXISTS resource_mab_runs (
                run_id      INTEGER PRIMARY KEY REFERENCES runs(run_id),
                created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                bandit_type TEXT    NOT NULL,
                initial_params      JSONB
            );
            """,
            # MAB updates tables
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_updates (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                reward INTEGER,
                params JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            # MAB samples tables
            """
            CREATE TABLE IF NOT EXISTS intervention_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                content_count INTEGER NOT NULL,
                feature_vector DOUBLE PRECISION[] NOT NULL,
                selected_rec_ids TEXT[] NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS recommendation_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                content_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS resource_mab_samples (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                content_count INTEGER NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                sample JSONB,
                PRIMARY KEY (run_id, id)
            );
            """,
            # users
            """
            CREATE TABLE IF NOT EXISTS users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                gender TEXT,
                userAge INTEGER,
                height INTEGER,
                weight INTEGER,
                recruitmentCenter TEXT,
                enrolmentDate TIMESTAMP WITH TIME ZONE,
                wearable TEXT,
                voiceRecording TEXT,
                occupation TEXT,
                education TEXT,
                digitalLiteracy TEXT,
                level INTEGER,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, user_id)
            );
            """,
            # disabled_users
            # users might be re-enabled...?
            """
            CREATE TABLE IF NOT EXISTS disabled_users (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL, 
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                date_disabled TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, user_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # escalation_levels
            """
            CREATE TABLE IF NOT EXISTS escalation_levels (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT clock_timestamp(),
                escalation_level INTEGER NOT NULL,
                pillar TEXT,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # health_habit_assessments
            """
            CREATE TABLE IF NOT EXISTS health_habit_assessments (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                assessment_timestamp TIMESTAMP WITH TIME ZONE,
                alcohol FLOAT,
                nutrition FLOAT,
                physical_activity FLOAT,
                smoking FLOAT,
                emotional_wellbeing FLOAT,
                nutrition_components JSONB,
                emotional_wellbeing_components JSONB,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # new_missions_and_contents flattened to new_missions
            """
            CREATE TABLE IF NOT EXISTS new_missions_and_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                update_timestamp TIMESTAMP WITH TIME ZONE,
                mission_id TEXT,
                recommendations TEXT[],
                resources TEXT[],
                prescribed BOOLEAN,
                selection_timestamp TIMESTAMP WITH TIME ZONE,
                finish_timestamp TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            # recommendation_plans + plan_contents
            """
            CREATE TABLE IF NOT EXISTS recommendation_plans (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                plan_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (run_id, id),
                UNIQUE (run_id, plan_id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS plan_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                plan_id TEXT NOT NULL,
                content_id TEXT,
                scheduled_for TIMESTAMP WITH TIME ZONE,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, plan_id) REFERENCES recommendation_plans(run_id, plan_id)
            );
            """,
            # selected_contents
            """
            CREATE TABLE IF NOT EXISTS selected_contents (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                plan_id TEXT NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE,
                mission_start_time TIMESTAMP WITH TIME ZONE,
                mission_end_time TIMESTAMP WITH TIME ZONE,
                content_ids TEXT[],
                mission_id TEXT,
                PRIMARY KEY (run_id, id)
            );
            """,
            # user_feedback
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                run_id INTEGER NOT NULL REFERENCES runs(run_id),
                id SERIAL NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                user_id TEXT NOT NULL,
                process_id INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE,
                event_name TEXT,
                properties JSONB,
                PRIMARY KEY (run_id, id),
                FOREIGN KEY (run_id, user_id) REFERENCES users(run_id, user_id)
            );
            """,
        ]

        conn = self._get_conn()
        cur = conn.cursor()

        # 1) Persist UTC as the default display/session TZ for this DB & role
        try:
            cur.execute("ALTER DATABASE cs_data SET timezone TO 'UTC';")
            cur.execute("ALTER ROLE cs_user SET timezone TO 'UTC';")
        except Exception as e:
            # Non-fatal if role/db already configured or lacking perms
            logging.warning("Could not persist UTC timezone defaults: %s", e)

        # 2) Ensure this session is UTC right now
        try:
            cur.execute("SET TIME ZONE 'UTC';")
        except Exception as e:
            logging.warning("Could not SET TIME ZONE to UTC for this session: %s", e)

        # 3) Create tables as you already do
        for stmt in table_creations:
            cur.execute(stmt)
        conn.commit()
        cur.close()
        conn.close()

    def _create_run(self):
        """Insert a new runs entry and return its run_id."""
        conn = self._get_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO runs(description) VALUES (%s) RETURNING run_id;", (os.getenv("PIPELINE_DESCRIPTION"),))
        run_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        return run_id

    def initialize_bandit(self, table, bandit_type, initial_params):
        conn = self._get_conn()
        cur = conn.cursor()
        clean_params = sanitize_for_json(initial_params)

        cur.execute(
            f"INSERT INTO {table} (run_id, bandit_type, initial_params) VALUES (%s, %s, %s);",
            (self.run_id, bandit_type, json.dumps(clean_params)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_intervention_mab_update(self, update: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = update.get("user_id")  # string
        process_id = update.get("process_id")
        ts = update.get("timestamp")
        feature_vector = update.get("feature_vector")
        reward = update.get("reward")
        params = update.get("params")
        clean = sanitize_for_json(params)
        cur.execute(
            "INSERT INTO intervention_mab_updates(run_id, user_id, process_id, timestamp, feature_vector, reward, params) VALUES (%s, %s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, process_id, ts, feature_vector, reward, json.dumps(clean)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_update(self, table: str, update: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = update.get("user_id")  # string
        process_id = update.get("process_id")
        ts = update.get("timestamp")
        reward = update.get("reward")
        params = update.get("params")
        clean = sanitize_for_json(params)
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, process_id, timestamp, reward, params) "
            f"VALUES (%s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, process_id, ts, reward, json.dumps(clean)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_intervention_mab_sample(self, record: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = record.get("user_id")  # string
        pc = record.get("plan_id")  # string
        cc = record.get("content_count")
        fv = record.get("feature_vector")
        sr = record.get("selected_rec_ids")
        ts = record.get("timestamp")
        sample = sanitize_for_json(record.get("sample"))
        cur.execute(
            "INSERT INTO intervention_mab_samples(run_id, user_id, plan_id, content_count, feature_vector, selected_rec_ids, timestamp, sample) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, pc, cc, fv, sr, ts, json.dumps(sample)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_mab_sample(self, table: str, record: dict):
        conn = self._get_conn()
        cur = conn.cursor()
        user_id = record.get("user_id")  # string
        pc = record.get("plan_id")  # string
        cc = record.get("content_count")
        ts = record.get("timestamp")
        sample = sanitize_for_json(record.get("sample"))
        cur.execute(
            f"INSERT INTO {table}(run_id, user_id, plan_id, content_count, timestamp, sample) "
            f"VALUES (%s, %s, %s, %s, %s, %s);",
            (self.run_id, user_id, pc, cc, ts, json.dumps(sample)),
        )
        conn.commit()
        cur.close()
        conn.close()

    def add_disabled_users(self, disabled_users):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, items in disabled_users.items():
            cur.execute(
                "INSERT INTO disabled_users(run_id, user_id, date_disabled) "
                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING;",
                (
                    self.run_id,
                    user_id,  # string
                    items["date_disabled"],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

    def add_escalation_levels(self, escalation_levels):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, levels_list in escalation_levels.items():
            for esc_level in levels_list:
                cur.execute(
                    "INSERT INTO escalation_levels(run_id, user_id, timestamp, escalation_level, pillar) "
                    "VALUES (%s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,  # string
                        esc_level["update_timestamp"],
                        esc_level["level"],
                        esc_level.get("pillar", None),
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_users(self, users):
        conn = self._get_conn()
        cur = conn.cursor()

        for user_id, u in users.items():
            cur.execute(
                """
                INSERT INTO users (
                    run_id, user_id, gender, userAge, height, weight,
                    recruitmentCenter, enrolmentDate, wearable, voiceRecording,
                    occupation, education, digitalLiteracy, level
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (run_id, user_id) DO UPDATE SET
                    gender = EXCLUDED.gender,
                    userAge = EXCLUDED.userAge,
                    height = EXCLUDED.height,
                    weight = EXCLUDED.weight,
                    recruitmentCenter = EXCLUDED.recruitmentCenter,
                    enrolmentDate = EXCLUDED.enrolmentDate,
                    wearable = EXCLUDED.wearable,
                    voiceRecording = EXCLUDED.voiceRecording,
                    occupation = EXCLUDED.occupation,
                    education = EXCLUDED.education,
                    digitalLiteracy = EXCLUDED.digitalLiteracy,
                    level = EXCLUDED.level
                """,
                (
                    self.run_id,
                    user_id,  # string
                    u.get("gender"),
                    u.get("userAge"),
                    u.get("height"),
                    u.get("weight"),
                    u.get("recruitmentCenter"),
                    u.get("enrolmentDate"),
                    u.get("wearable"),
                    u.get("voiceRecording"),
                    u.get("occupation"),
                    u.get("education"),
                    u.get("digitalLiteracy"),
                    u.get("level"),
                ),
            )

        conn.commit()
        cur.close()
        conn.close()

    def add_health_habit_assessments(self, assessments):
        conn = self._get_conn()
        cur = conn.cursor()

        def _sanitize_components_dict(d, pillar_name):
            """Return a new dict with float values; drop and warn on non-numeric."""
            if not isinstance(d, dict):
                logging.warning("HHS components for %s not a dict: %r", pillar_name, d)
                return None
            clean = {}
            for k, v in d.items():
                try:
                    clean[k] = float(v)
                except Exception:
                    logging.warning("Component %s.%s has non-numeric value %r; skipping.", pillar_name, k, v)
            return clean if clean else None

        for user_id, hhs_list in assessments.items():
            for entry in hhs_list:
                h = entry.get("hhs", {}) or {}

                # main pillars (as sent in this payload only)
                alcohol = h.get("alcohol")
                physical_activity = h.get("physical_activity")
                smoking = h.get("smoking")

                # nutrition + components (only attach components if pillar key present)
                nutrition = h.get("nutrition") if "nutrition" in h else None
                nutrition_components = None
                if "nutrition" in h and "components" in h:
                    nc = _sanitize_components_dict(h.get("components"), "nutrition")
                    nutrition_components = Json(nc) if nc is not None else None

                # emotional wellbeing + components (incl. emotional_distress-only case)
                ew = h.get("emotional_wellbeing") if "emotional_wellbeing" in h else None
                ew_components = None
                if "emotional_wellbeing" in h and "components" in h:
                    ec = _sanitize_components_dict(h.get("components"), "emotional_wellbeing")
                    ew_components = Json(ec) if ec is not None else None
                elif "emotional_distress" in h:
                    # Bi-weekly single-component update
                    try:
                        ed_val = float(h.get("emotional_distress"))
                        ew_components = Json({"emotional_distress": ed_val})
                    except Exception:
                        logging.warning(
                            "emotional_distress has non-numeric value %r; skipping.", h.get("emotional_distress")
                        )

                # If components appear without a recognized pillar, warn (don’t persist ambiguous components)
                if "components" in h and ("nutrition" not in h and "emotional_wellbeing" not in h):
                    logging.warning(
                        "Received 'components' without nutrition/emotional_wellbeing for user %s at %r; skipping components.",
                        user_id,
                        entry.get("assessment_timestamp"),
                    )

                cur.execute(
                    "INSERT INTO health_habit_assessments"
                    " (run_id, user_id, assessment_timestamp, alcohol, nutrition,"
                    "  physical_activity, smoking, emotional_wellbeing, nutrition_components, emotional_wellbeing_components)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,  # string
                        entry.get("assessment_timestamp"),
                        alcohol,
                        nutrition,
                        physical_activity,
                        smoking,
                        ew,
                        nutrition_components,
                        ew_components,
                    ),
                )

        conn.commit()
        cur.close()
        conn.close()

    def add_new_missions_and_contents(self, entries):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, entry in entries.items():
            ts = entry["update_timestamp"]
            for m in entry.get("new_missions", []):
                cur.execute(
                    "INSERT INTO new_missions_and_contents"
                    " (run_id, update_timestamp, user_id, mission_id, recommendations, resources, prescribed, selection_timestamp, finish_timestamp)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        ts,
                        user_id,  # string
                        m["mission"],
                        m.get("recommendations", []),
                        m.get("resources", []),
                        m["prescribed"],
                        m["selection_timestamp"],
                        m["finish_timestamp"],
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()

    def add_recommendation_plans(self, plans):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_plan in plans["recommendation_plans"]:
            cur.execute(
                "INSERT INTO recommendation_plans (run_id, user_id, plan_id) VALUES (%s, %s, %s);",
                (self.run_id, user_plan["user_id"], user_plan["plan_id"]),
            )
            for c in user_plan.get("plans", []):
                cur.execute(
                    "INSERT INTO plan_contents (run_id, plan_id, content_id, scheduled_for) VALUES (%s, %s, %s, %s);",
                    (self.run_id, user_plan["plan_id"], c["content_id"], c["scheduled_for"]),
                )
        conn.commit()
        cur.close()
        conn.close()

    # for simplicity, we assume 1 mission per user
    def add_selected_contents(self, selections):
        conn = self._get_conn()
        cur = conn.cursor()
        ts = selections["timestamp"]
        user_to_mission_id = selections["mission_id"]
        for user_id, items in selections["selected_contents"].items():
            cur.execute(
                "INSERT INTO selected_contents"
                " (run_id, user_id, plan_id, timestamp, mission_start_time, mission_end_time, content_ids, mission_id)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
                (
                    self.run_id,
                    user_id,  # string
                    items["plan_id"],  # string
                    ts,
                    items["mission_start_time"],
                    items["mission_end_time"],
                    [item["id"] for item in items["contents"]],  # TEXT[] in schema
                    user_to_mission_id[user_id],
                ),
            )
        conn.commit()
        cur.close()
        conn.close()

    def add_user_feedback(self, feedback):
        conn = self._get_conn()
        cur = conn.cursor()
        for user_id, user_feedback in feedback.items():
            for event in user_feedback["events"]:
                cur.execute(
                    "INSERT INTO user_feedback"
                    " (run_id, user_id, process_id, timestamp, event_name, properties)"
                    " VALUES (%s, %s, %s, %s, %s, %s);",
                    (
                        self.run_id,
                        user_id,  # string
                        event["process_id"],
                        event["timestamp"],
                        event["event_name"],
                        Json(event["properties"]),
                    ),
                )
        conn.commit()
        cur.close()
        conn.close()
