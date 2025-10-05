from virtual_user.services.user_factory import UserFactory
from virtual_user.services.feedback_manager import FeedbackManager
from virtual_user.services.content_manager import ContentManager
from virtual_user.utils.contents import load_json_files
from virtual_user.config import ENTRANCE_TIMES, NUM_WEEKLY_USERS


class VirtualUser:
    """Simulated backend service for handling tasks."""

    def __init__(self, time_handler, num_weeks_per_user=12, num_new_weekly_users=NUM_WEEKLY_USERS):
        self.time_handler = time_handler
        self.num_weeks_per_user = num_weeks_per_user
        self.num_new_weekly_users = num_new_weekly_users
        self.updates = {}
        self.users = {}
        self.content_manager = ContentManager()
        self.user_factory = UserFactory(self.time_handler, self.content_manager, self.num_weeks_per_user)
        self.feedback_manager = FeedbackManager(self.time_handler, self.users, self.num_weeks_per_user)
        self.raw_missions, self.raw_recommendations, self.raw_resources = load_json_files()
        self.missions, self.recommendations, self.resources = load_json_files("as_dict")

    def get_updates(self):
        return self.updates

    def save_weekly_recommendation_plans(self, recommendation_plans):
        # logging.info(f"recommendation_plans: {recommendation_plans}")
        for user_plan in recommendation_plans["recommendation_plans"]:
            self.users[user_plan["user_id"]].update_recommendation_plan(user_plan)
            self.users[user_plan["user_id"]].add_stored_resource(
                [content["content_id"] for content in user_plan["plans"] if content["type"] == "resource"]
            )

        return True

    def simulate_user_feedback(self):
        # Simulate feedback for all users
        user_feedback = {}
        for user_id, user in self.users.items():
            if user.active:
                events = user.simulate_user_feedback()
                if events:
                    user_feedback[user_id] = {"events": events}

        return user_feedback

    def generate_new_users(self):
        if self.time_handler.now.replace(second=0, microsecond=0) in ENTRANCE_TIMES:
            new_users = self.user_factory.generate_users(self.num_new_weekly_users)
            self.users.update(new_users)
            return {
                user_id: {
                    key: user.profile[key]
                    for key in [
                        "gender",
                        "userAge",
                        "height",
                        "weight",
                        "recruitmentCenter",
                        "enrolmentDate",
                        "wearable",
                        "voiceRecording",
                        "occupation",
                        "education",
                        "digitalLiteracy",
                        "level",
                    ]
                }
                for user_id, user in new_users.items()
            }
        return {}

    def get_disabled_users(self):
        date_disabled = self.time_handler.utc_iso(self.time_handler.now)
        disabled_users = {
            user_id: {"date_disabled": date_disabled}
            for user_id, user in self.users.items()
            if (user.active and user.intervention_week >= self.num_weeks_per_user)
        }
        for user_id in disabled_users:
            self.users[user_id].disable()

        return disabled_users

    def simulate_health_habit_assessment(self):
        health_assessments = {}
        for user_id, user in self.users.items():
            if user.active:
                health_assessment = user.simulate_health_habit_assessment()
                if health_assessment:
                    health_assessments[user_id] = health_assessment
        return health_assessments

    def generate_new_missions_for_users(self):
        """Generate new missions and contents for all users."""
        new_missions_and_contents = {}
        for user_id, user in self.users.items():
            if user.active:
                user_new_missions_and_contents = user.select_new_missions()
                if user_new_missions_and_contents:
                    new_missions_and_contents[user_id] = user_new_missions_and_contents

        return new_missions_and_contents

    def simulate_hour(self):
        self.updates = {
            "user_feedback": self.simulate_user_feedback(),
            "new_users": self.generate_new_users(),
            "disabled_users": self.get_disabled_users(),
            "health_habit_assessments": self.simulate_health_habit_assessment(),
            "new_missions_and_contents": self.generate_new_missions_for_users(),
        }
