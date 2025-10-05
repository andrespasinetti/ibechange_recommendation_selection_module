# content_selection/user_manager.py
import logging
from cs_module.content_selection.user import User

logger = logging.getLogger(__name__)


class UserManager:
    def __init__(self, time_handler):
        self.time_handler = time_handler
        self.users = {}
        self.active_user_ids = []

    def get_all_users(self):
        return self.users

    def get_user(self, user_id):
        return self.users.get(user_id)

    def add_users(self, new_users):
        if not new_users:
            return

        for user_id, data in new_users.items():
            enrol_raw = data.get("enrolmentDate")  # normalized ISO string from core
            try:
                enrol_dt = self.time_handler.parse_client_ts(enrol_raw)
            except Exception as e:
                # Should be rare now, but keep a safe fallback and a log
                enrol_dt = self.time_handler.now
                logging.warning(
                    "User %s enrolmentDate could not be parsed (%r, %s); falling back to now=%s",
                    user_id,
                    enrol_raw,
                    e,
                    enrol_dt.isoformat(),
                )

            # Create/refresh in-memory user
            self.users[user_id] = User(
                user_id=user_id,
                time_handler=self.time_handler,
                personal_data=data,  # keep ISO string in dict
                intervention_start_date=enrol_dt,  # datetime for computations
            )
            if user_id not in self.active_user_ids:
                self.active_user_ids.append(user_id)

    def get_active_user_ids(self):
        return self.active_user_ids

    def disable_users(self, disabled_users):
        for user_id in disabled_users:
            if user_id in self.active_user_ids:
                self.active_user_ids.remove(user_id)
                self.users[user_id].disable()

    def update_escalation_levels(self, escalation_levels):
        for user_id, level_list in escalation_levels.items():
            user = self.get_user(user_id)
            if not user:
                logging.warning(f"User {user_id} not found for escalation level update.")
                continue

            for level in level_list:
                user.update_escalation_level(level["level"])

    def update_health_habits(self, assessments):
        for user_id, hhs_list in assessments.items():
            user = self.get_user(user_id)
            if not user:
                logging.warning(f"User {user_id} not found for health habit update.")
                continue

            # HHS are in ascending order by timestamp, so we start updating from the oldest
            for hhs in hhs_list:
                user.update_health_habit_assessment(hhs["hhs"])

    def update_mission_accomplished(self, user_feedback):
        for user_id, feedback in user_feedback.items():
            user = self.get_user(user_id)
            if not user:
                logging.warning(f"User {user_id} not found. Skipping feedback.")
                continue

            mission_scores = []

            for event in feedback.get("events", []):
                if event.get("event_name") != "mission_accomplished":
                    continue

                mission_id = event["properties"]["mission_id"]
                mission_score = event["properties"]["score"]
                mission_scores.append(mission_score)

                # purely for sanity/logging — do not mutate “active” state
                try:
                    evt_ts = self.time_handler.parse_client_ts(event["timestamp"])
                except Exception:
                    evt_ts = self.time_handler.now

                snap = user.mission_snapshot_at(mission_id, evt_ts)
                if snap is None:
                    logging.warning(
                        f"User {user_id} completed mission {mission_id} at {evt_ts.isoformat()}, "
                        f"but no selection snapshot exists at or before that time."
                    )

            if mission_scores:
                avg_mission_score = sum(mission_scores) / len(mission_scores)
                user.set_previous_mission_score(avg_mission_score)

    def update_missions_and_contents(self, new_missions_and_contents):
        for user_id, missions_and_contents in new_missions_and_contents.items():
            user = self.get_user(user_id)
            if user:
                user.update_missions_and_contents(missions_and_contents)
            else:
                logging.warning(f"User {user_id} not found for mission update.")

    def save_recommendation_plans(self, recommendation_plans):
        for user_plan in recommendation_plans["recommendation_plans"]:
            user = self.get_user(user_plan["user_id"])
            if user:
                user.save_recommendation_plan(user_plan)
            else:
                logging.warning(f"User {user_plan['user_id']} not found for saving weekly plan.")

    # NEW: single-event primitive — apply one "mission selected"
    def apply_mission_selected(self, user_id: str, mission: dict) -> None:
        user = self.get_user(user_id)
        if not user:
            logging.warning(f"User {user_id} not found for mission selection.")
            return

        # Reuse existing user method: it already sets plan_required=True,
        # appends to selected_missions_and_contents, sets intervention_start_date
        # on first mission, and flips new_plan_required=True.
        user.update_missions_and_contents({"new_missions": [mission]})

    # NEW: single-event primitive — apply one "mission accomplished"
    def apply_mission_accomplished(self, user_id: str, mission_id: str, score: float | int) -> None:
        user = self.get_user(user_id)
        if not user:
            logging.warning(f"User {user_id} not found for mission accomplished.")
            return

        user.set_previous_mission_score(score)
