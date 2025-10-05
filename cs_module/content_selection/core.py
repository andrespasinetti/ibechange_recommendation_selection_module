import logging
import os
import json
import numpy as np
from datetime import timedelta
from cs_module.content_selection.engine import RecommendationEngine
from cs_module.content_selection.mab_initialiser import MABInitialiser
from cs_module.content_selection.user_manager import UserManager
from cs_module.content_selection.mab_updater import MABUpdater
from cs_module.utils.data_storage import DataStorage
from cs_module.utils.process_binder import ProcessBinder
from cs_module.utils.logging_utils import pretty


class ContentSelection:
    def __init__(
        self,
        time_handler,
    ):
        logging.info("Initialising ContentSelection...")
        self.time_handler = time_handler
        self.missions = {}
        self.recommendations = {}
        self.resources = {}
        self.user_manager = UserManager(self.time_handler)
        self.binder = ProcessBinder()
        self.data_storage = DataStorage()
        mab_initializer = MABInitialiser(self.data_storage)
        self.resource_mab = mab_initializer.resource_mab
        self.intervention_mab = mab_initializer.intervention_mab
        self.recommendation_mab = mab_initializer.recommendation_mab

        self.mab_updater = None
        self.recommendation_engine = None

    def initialise_missions(self, missions):
        self.missions = {rec["mission_id"]: rec for rec in missions}

    def initialise_recommendations(self, recommendations):
        self.recommendations = {rec["rec_id"]: rec for rec in recommendations}

    def initialise_resources(self, resources):
        self.resources = {rec["rec_id"]: rec for rec in resources}

    def convert_ndarrays_to_lists(self, obj):
        """Recursively convert all ndarrays in a structure to lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_ndarrays_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarrays_to_lists(item) for item in obj]
        return obj

    def save_output(self, output, filename):
        path = f"outputs/{filename}.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Convert any ndarray objects to lists
        output = self.convert_ndarrays_to_lists(output)

        try:
            with open(path, "w") as f:
                json.dump(output, f, indent=4)
            print(f"Output saved to {path}")
            logging.info(f"Output saved to {path}")
        except (IOError, OSError) as e:
            logging.error(f"Error saving output to {path}: {e}")

    # Hourly
    def update(self, updates, is_learning, is_intervention):
        logging.info("Delivering intervention: %s", is_intervention)
        logging.info("MAB learning: %s", is_learning)
        if self.mab_updater is None:
            self.mab_updater = MABUpdater(
                binder=self.binder,
                time_handler=self.time_handler,
                resource_mab=self.resource_mab,
                intervention_mab=self.intervention_mab,
                recommendation_mab=self.recommendation_mab,
                user_manager=self.user_manager,
                recommendations=self.recommendations,
                missions=self.missions,
                resources=self.resources,
                data_storage=self.data_storage,
            )

        if "new_users" in updates:
            logging.info("New users:\n%s", pretty(updates["new_users"]))
            new_users = updates["new_users"]
            self._normalize_new_users(new_users)
            self.data_storage.add_users(new_users)
            self.user_manager.add_users(new_users)

        known_users = set(self.user_manager.get_all_users())

        if "health_habit_assessments" in updates:
            logging.info("Health Habit Assessments:\n%s", pretty(updates["health_habit_assessments"]))
            health_habit_assessments = updates["health_habit_assessments"]

            # 1) Filter to known users (like user_feedback)
            filtered_hha = {}
            for user_id, entries in health_habit_assessments.items():
                if user_id in known_users:
                    filtered_hha[user_id] = entries
                else:
                    logging.warning("User %s not found. Skipping health_habit_assessments.", user_id)

            if filtered_hha:
                self.data_storage.add_health_habit_assessments(filtered_hha)
                self.user_manager.update_health_habits(filtered_hha)

        # --- Initialize filtered dicts up-front ---------------------------------
        filtered_feedback = {}
        filtered_nmac = {}

        # User feedback (persist + filtered) -------------------------------------
        raw_feedback = updates.get("user_feedback", {})
        if raw_feedback:
            logging.info("User feedback:\n%s", pretty(raw_feedback, max_chars=20000))
            filtered_feedback = {uid: fb for uid, fb in raw_feedback.items() if uid in known_users}
            missing = set(raw_feedback) - set(filtered_feedback)
            for uid in missing:
                logging.warning(f"User {uid} not found. Skipping feedback.")

            for uid, fb in filtered_feedback.items():
                if "events" in fb and fb["events"]:
                    fb["events"] = self._sort_feedback_events(fb["events"])
            self.data_storage.add_user_feedback(filtered_feedback)

        # New missions and contents (persist + filtered) -------------------------
        new_missions_and_contents = updates.get("new_missions_and_contents", {})
        if new_missions_and_contents:
            logging.info("New missions and contents:\n%s", pretty(new_missions_and_contents))
            filtered_nmac = {uid: nm for uid, nm in new_missions_and_contents.items() if uid in known_users}
            missing = set(new_missions_and_contents) - set(filtered_nmac)
            for uid in missing:
                logging.warning(f"User {uid} not found. Skipping new_missions_and_contents.")
            self.data_storage.add_new_missions_and_contents(filtered_nmac)

        # 3) Merge & order mission-affecting events by timestamp
        merged_by_user = {}

        # 3a) From feedback: mission accomplished
        for user_id, fb in filtered_feedback.items():
            events = fb.get("events", [])
            for ev in events:
                if ev.get("event_name") != "mission_accomplished":
                    continue
                ts_raw = ev.get("timestamp")
                try:
                    ts = self.time_handler.parse_client_ts(ts_raw, mute_naive_warning=True)
                except Exception as e:
                    logging.warning("Bad feedback timestamp %r for user %s (%s) -> skip", ts_raw, user_id, e)
                    continue
                merged_by_user.setdefault(user_id, []).append(
                    {
                        "ts": ts,
                        "type": "accomplish",
                        "mission_id": ev["properties"]["mission_id"],
                        "score": ev["properties"].get("score"),
                    }
                )

        # 3b) From new missions: mission selected
        for user_id, entry in filtered_nmac.items():
            container_ts_raw = entry.get("update_timestamp")
            # We prefer the mission's own selection_timestamp if present
            for m in entry.get("new_missions", []):
                ts_raw = m.get("selection_timestamp") or container_ts_raw
                try:
                    ts = self.time_handler.parse_client_ts(ts_raw, mute_naive_warning=True)
                except Exception as e:
                    logging.warning("Bad selection timestamp %r for user %s (%s) -> use now()", ts_raw, user_id, e)
                    ts = self.time_handler.now
                merged_by_user.setdefault(user_id, []).append(
                    {
                        "ts": ts,
                        "type": "select",
                        "mission": m,  # full mission dict (mission, recommendations, resources, prescribed, ... timestamps)
                    }
                )

        # 3c) Apply per-user in chronological order
        for user_id, items in merged_by_user.items():
            # select < accomplish on ties
            order = {"select": 0, "accomplish": 1}
            items.sort(key=lambda x: (x["ts"], order[x["type"]]))
            for ev in items:
                if ev["type"] == "select":
                    self.user_manager.apply_mission_selected(user_id, ev["mission"])
                elif ev["type"] == "accomplish":
                    self.user_manager.apply_mission_accomplished(user_id, ev["mission_id"], ev.get("score"))

        # 4) MAB learning still uses the full filtered_feedback (unchanged)
        if raw_feedback and is_learning:
            # Filter to known users (again) for MAB update
            filtered_feedback = {uid: fb for uid, fb in raw_feedback.items() if uid in known_users}
            self.mab_updater.update_all(filtered_feedback)

        if "escalation_level" in updates:
            logging.info("Escalation levels:\n%s", pretty(updates["escalation_level"]))
            escalation_levels = updates["escalation_level"]
            self.data_storage.add_escalation_levels(escalation_levels)
            self.user_manager.update_escalation_levels(escalation_levels)

        if "disabled_users" in updates:
            logging.info("Disabled users:\n%s", pretty(updates["disabled_users"]))
            disabled_users = updates["disabled_users"]
            self.data_storage.add_disabled_users(disabled_users)
            self.user_manager.disable_users(disabled_users)

        if is_intervention:
            self._select_contents()

    def get_selected_contents(self, start_time, end_time):
        selected_contents = {}
        for user_id, user in self.user_manager.get_all_users().items():
            if not user.selected_contents:
                continue

            mission_start = self.time_handler.parse_client_ts(user.selected_contents["mission_start_time"])
            if start_time and mission_start < start_time:
                continue
            if end_time and mission_start >= end_time:
                continue
            selected_contents[user_id] = user.selected_contents
        return selected_contents

    def save_recommendation_plans(self, recommendation_plans):
        self.data_storage.add_recommendation_plans(recommendation_plans)
        self.user_manager.save_recommendation_plans(recommendation_plans)
        return True

    def _normalize_new_users(self, new_users: dict) -> None:
        """Ensure enrolmentDate is a valid ISO string; on invalid, log and use time_handler.now."""
        for uid, data in new_users.items():
            raw = data.get("enrolmentDate")
            try:
                # Treat missing/empty/sentinel as invalid
                if raw in (None, "", "Invalid Date"):
                    raise ValueError("missing or invalid enrolmentDate")
                dt = self.time_handler.parse_client_ts(raw, mute_naive_warning=True)  # -> aware datetime,
            except Exception as e:
                dt = self.time_handler.now  # aware UTC from TimeHandler
                logging.warning(
                    "User %s enrolmentDate invalid (%r, %s); using now=%s", uid, raw, e, self.time_handler.utc_iso(dt)
                )
            # Keep a normalized ISO UTC string for downstream (DB & logs)
            data["enrolmentDate"] = self.time_handler.utc_iso(dt)

    def _sort_feedback_events(self, events):
        order = {
            "recommendation_sent": 0,
            "recommendation_opened": 1,
            "recommendation_rated": 2,
            "resource_rated": 2,  # after opens, before anything else you might add
        }

        def key(ev):
            # parse ts safely (assume invalid -> now so it sinks to the end deterministically)
            ts_raw = ev.get("timestamp")
            try:
                ts = self.time_handler.parse_client_ts(ts_raw, mute_naive_warning=True)
            except Exception:
                ts = self.time_handler.now
            # event priority (unknowns go last), then process_id to get stable order on ties
            typ = ev.get("event_name") or ""
            pri = order.get(typ, 99)
            pid = ev.get("process_id") or 0
            return (ts, pri, pid)

        return sorted(events, key=key)

    def _select_contents(self):
        if self.recommendation_engine is None:
            self.recommendation_engine = RecommendationEngine(
                self.user_manager,
                self.binder,
                self.missions,
                self.resources,
                self.recommendations,
                self.resource_mab,
                self.intervention_mab,
                self.recommendation_mab,
                self.data_storage,
                self.time_handler,
            )

        selected_contents = {}
        user_to_mission_id = {}
        for user_id, user in self.user_manager.get_all_users().items():
            if user.new_plan_required:
                new_missions = user.get_new_missions()
                # PILOT STUDY WILL HAVE ONE MISSION AT A TIME
                if new_missions:
                    # pick the most recent selection, can happen during normal run if user changes inside hour but also if something breaks
                    # IMAGINE MY MODULE CRASHES, IN THE GAP TO FIX IT USERS CAN CHANGE MISSIONS,
                    # WHEN WE GO BACK TO LIVE WE PLAN ONLY FOR THE MOST RECENT ONE
                    mission = max(
                        new_missions, key=lambda m: user.time_handler.parse_client_ts(m["selection_timestamp"])
                    )
                    user_to_mission_id[user_id] = mission["mission"]

                    # CLEAR only if other missions have been selected before this one:
                    older_ids = [m["mission"] for m in new_missions if m["mission"] != mission["mission"]]
                    if older_ids:
                        user.set_missions_plan_to_false(older_ids)
                else:
                    logging.warning(f"No new missions found for user {user_id}. Skipping content selection.")
                    continue
                selected_contents[user_id] = {}
                selected_resources = self.recommendation_engine.get_resources_to_send(user_id)
                selected_recommendations = self.recommendation_engine.get_recommendations_to_send(user_id)
                selected_contents[user_id]["contents"] = selected_resources + selected_recommendations

                start_time = self.time_handler.parse_client_ts(mission["selection_timestamp"])
                end_time = start_time + timedelta(days=7)

                selected_contents[user_id]["mission_start_time"] = self.time_handler.utc_iso(start_time)
                selected_contents[user_id]["mission_end_time"] = self.time_handler.utc_iso(end_time)

                plan_id = self.recommendation_engine.get_current_plan_id(user_id)
                selected_contents[user_id]["plan_id"] = str(plan_id)
                logging.info("Selected contents for user %s:\n%s", user_id, pretty(selected_contents[user_id]))
                user.new_plan_required = False
                user.set_missions_plan_to_false([user_to_mission_id[user_id]])
                user.selected_contents = selected_contents[user_id]
                self.recommendation_engine.rotate_plan_id(user_id)

        if selected_contents:
            timestamp = self.time_handler.utc_iso(self.time_handler.now)
            selected_contents_and_timestamps = {
                "timestamp": timestamp,
                "selected_contents": selected_contents,
                "mission_id": user_to_mission_id,
            }
            self.data_storage.add_selected_contents(selected_contents_and_timestamps)
