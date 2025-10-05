import random
from virtual_user.services.feedback_manager import FeedbackManager
from virtual_user.services.user_mission_manager import UserMissionManager
from copy import deepcopy
from datetime import timedelta
import logging
from virtual_user.utils.recommendation_history_tracker import RecommendationHistoryTracker
from virtual_user.config import MISSION_SELECTION_MODE, PILLARS
from virtual_user.utils.encoding import get_personal_data_encoding


class User:
    def __init__(self, time_handler, user_id, profile, num_weeks_per_user, content_manager):
        self.time_handler = time_handler
        self.intervention_start_time = deepcopy(time_handler.now)
        self.user_id = user_id
        self.profile = profile  # Store the profile as a dictionary
        self.demography = self.generate_demography()
        self.demography_encoding = self.get_demography_encoding()
        self.num_weeks_per_user = num_weeks_per_user
        self.active = True
        self.health_habit_assessment = {}
        self.hhs_encoding = []
        self.hhs_questionnaire_times = [self.intervention_start_time + timedelta(weeks=i * 4) for i in range(0, 3)]
        self.weekly_pillars = []
        self.current_missions = []
        self.new_missions_and_contents = {}
        self.weekly_available_recommendations = []
        self.weekly_available_resources = []
        self.feedback = []
        self.weekly_recommendation_plan = {}
        self.weekly_required_delayed_feedback = []
        self.stored_resources = []
        self.new_mission_expiration = None
        self.content_manager = content_manager
        self.feedback_manager = FeedbackManager(self, self.time_handler, self.num_weeks_per_user)
        self.user_mission_manager = UserMissionManager(self, self.content_manager.missions)
        self.opened_rec_tracker = RecommendationHistoryTracker()
        self.current_rated_contents = []

    def get_demography_encoding(self):
        demography_encoding = get_personal_data_encoding(self.demography)
        return demography_encoding

    def generate_demography(self):
        return {
            "userAge": self.profile["userAge"],
            "gender": self.profile["gender"],
            "recruitmentCenter": self.profile["recruitmentCenter"],
            "education": self.profile["education"],
        }

    def get_demography(self):
        return self.demography

    def get_hhs(self):
        hhs = {}
        for hhs_elem in self.health_habit_assessment:
            key = list(hhs_elem["hhs"].keys())[0]
            value = list(hhs_elem["hhs"].values())[0]
            hhs[key] = value

        return hhs

    def disable(self):
        self.active = False

    @property
    def intervention_day(self):
        return (self.time_handler.now - self.intervention_start_time).days

    @property
    def intervention_week(self):
        return self.intervention_day // 7

    def update_profile(self, profile_data):
        """Update the user's profile with new data."""
        self.profile.update(profile_data)

    def simulate_health_habit_assessment(self):
        """Simulate health habit assessment for this user."""
        # Each 4 weeks
        if self.time_handler.now.replace(second=0, microsecond=0) in self.hhs_questionnaire_times:
            if not self.health_habit_assessment:
                self.health_habit_assessment = [
                    {
                        "hhs": {
                            pillar: random.randint(0, 100),
                        },
                        "assessment_timestamp": self.time_handler.now,
                    }
                    for pillar in PILLARS
                ]
            else:
                for entry in self.health_habit_assessment:
                    for pillar in entry["hhs"]:
                        delta = random.randint(-10, 10)
                        entry["hhs"][pillar] = max(0, min(100, entry["hhs"][pillar] + delta))
                # Optionally update timestamp to reflect new assessment
                for entry in self.health_habit_assessment:
                    entry["assessment_timestamp"] = self.time_handler.now
            return self.health_habit_assessment
        return {}

    def add_mission(self, mission):
        """Add a mission to the user's weekly missions."""
        self.current_missions.append(mission)

    def update_recommendation_plan(self, plan):
        """Update the user's weekly recommendation plan."""
        self.weekly_recommendation_plan = plan

    def add_stored_resource(self, resource_id):
        """Store a resource in the user's resources."""
        self.stored_resources.extend(resource_id)

    def simulate_user_feedback(self):
        """User provides feedback using the FeedbackManager."""
        feedback = self.feedback_manager.simulate_feedback()
        self.feedback.append(feedback)
        return feedback

    def select_new_missions(self):
        """Delegates mission selection to the UserMissionManager."""
        if (
            ((self.new_mission_expiration is not None) and (not self.time_handler.now >= self.new_mission_expiration))
            or (self.intervention_day % 7 != 0)
            or (self.intervention_week >= self.num_weeks_per_user)
        ):
            return None

        logging.info(f"Selecting new missions. self.new_mission_expiration: {self.new_mission_expiration}")
        user_new_missions = self.user_mission_manager.select_new_missions(
            mode=MISSION_SELECTION_MODE
        )  # Always select a random mission from a random pillar
        available_recommendations, available_resources = (
            self.content_manager.get_available_recommendations_and_resources(user_new_missions, self.stored_resources)
        )
        update_timestamp = self.time_handler.now
        self.new_mission_expiration = update_timestamp + timedelta(days=7)
        self.current_missions = user_new_missions
        self.weekly_available_recommendations = available_recommendations
        self.weekly_available_resources = available_resources
        self.new_missions_and_contents = {
            "update_timestamp": update_timestamp.isoformat(),
            "new_missions": [
                {
                    "mission": mission_id,
                    "recommendations": available_recommendations.get(mission_id, []),
                    "resources": available_resources.get(mission_id, []),
                    "prescribed": False,
                    "selection_timestamp": update_timestamp.isoformat(),  # ASSUME ONE MISSION AT A TIME IN PILOT
                    "finish_timestamp": None,  # in iBeChange pilot it indicates WHEN FINISHED
                }
                for mission_id in user_new_missions
            ],
        }
        self.current_rated_contents = []
        return self.new_missions_and_contents

    def track_opened_recommendations(self, open_timestamp, process_id, rec_id, intervention_type):
        self.opened_rec_tracker.add_recommendation(open_timestamp, process_id, rec_id, intervention_type)

    def get_total_frequency(self, time_window=None):
        """Get the global frequency of the user."""
        total_frequency = self.opened_rec_tracker.get_count(time_window=time_window, rec_id=None, single_intv=None)
        return total_frequency

    def get_recommendation_frequency(self, rec_id, time_window=None):
        """Get the frequency of the recommendation."""
        frequency = self.opened_rec_tracker.get_count(time_window=time_window, rec_id=rec_id, single_intv=None)
        return frequency

    def get_intervention_frequency(self, intervention_type, time_window=None):
        """Get the frequency of the intervention.

        - If intervention_type is a list with at least 2 elements, return the average frequency across types.
        - If it's a single type (string or single-element list), return the total count for that type.
        - If it's empty or None, return 0.
        """
        if not intervention_type:
            return 0
        total = sum(
            self.opened_rec_tracker.get_count(time_window=time_window, rec_id=None, single_intv=itype)
            for itype in intervention_type
        )
        return total / len(intervention_type)

    def get_contents_to_rate(self):
        last_contents = []
        for i, c1 in enumerate(self.weekly_recommendation_plan["plans"]):
            is_last_content = True
            for j in range(i + 1, len(self.weekly_recommendation_plan["plans"])):
                c2 = self.weekly_recommendation_plan["plans"][j]
                if c2["content_id"] == c1["content_id"]:
                    is_last_content = False

            if is_last_content:
                last_contents.append(c1)

        contents_to_rate = []
        for content in last_contents:
            if content not in self.current_rated_contents:
                contents_to_rate.append(content)

        return contents_to_rate
