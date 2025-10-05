from datetime import timedelta
import random
import numpy as np
from virtual_user.utils.encoding import get_intervention_feature_vector
from virtual_user.utils.contents import load_json_files
from virtual_user.utils.get_pillar import get_pillar
from virtual_user.config import REWARD_TYPE, OPEN_PROBABILITY, RATE_PROBABILITY
import logging


class FeedbackManager:
    process_count = 0

    def __init__(self, user, time_handler, num_weeks_per_user):
        self.time_handler = time_handler
        self.user = user
        self.num_weeks_per_user = num_weeks_per_user
        self.missions, self.recommendations, self.resources = load_json_files("as_dict")

    def get_rec_bias(self, mission_id, rec_id):
        res_preferences, int_preferences, rec_preferences = self.user.profile["preferences"]
        demography = self.user.get_demography()
        hhs = self.user.get_hhs()
        num_intervention_days = self.user.intervention_day

        # ASSUME ONE MISSION AT A TIME
        selection_timestamp_str = self.user.new_missions_and_contents["update_timestamp"]
        selection_timestamp = self.time_handler.parse_client_ts(selection_timestamp_str)
        time_window_past_week = (selection_timestamp - timedelta(weeks=1), selection_timestamp)
        time_window_scheduled = (selection_timestamp, self.time_handler.now)

        intervention_type = self.recommendations[rec_id]["intervention_type"]

        # Create feature vector for this recommendation
        int_feature_vector = get_intervention_feature_vector(
            demography,
            hhs,
            num_intervention_days,
            pillar=get_pillar(rec_id),
            mission_frequency=self.missions[mission_id]["weekly_frequency"],
            total_frequency_past_week=self.user.get_total_frequency(time_window_past_week),
            total_frequency_scheduled=self.user.get_total_frequency(time_window_scheduled),
            intervention=self.recommendations[rec_id]["intervention_type"],
            intervention_frequency_past_week=self.user.get_intervention_frequency(
                intervention_type, time_window_past_week
            ),
            intervention_frequency_scheduled=self.user.get_intervention_frequency(
                intervention_type, time_window_scheduled
            ),
            recommendation_frequency_past_week=self.user.get_recommendation_frequency(rec_id, time_window_past_week),
            recommendation_frequency_scheduled=self.user.get_recommendation_frequency(rec_id, time_window_scheduled),
        )
        if len(int_preferences) != len(int_feature_vector):
            logging.warning(
                f"Mismatch: int_preferences({len(int_preferences)}), int_feature_vector({len(int_feature_vector)})"
            )

        int_score = np.array(int_preferences) @ np.array(int_feature_vector)
        rec_bias = rec_preferences[rec_id]

        # Assume additive effects of recommendations
        preference_score = int_score + rec_bias

        if REWARD_TYPE == "thumbs":
            prob = 1 / (1 + np.exp(-preference_score))
            reward_rand = np.random.rand() < prob
            rating = "liked" if reward_rand else "disliked"  # FIX AVG. PARAMS (SHOULD THEY SUM TO 0?)

        elif REWARD_TYPE == "float":
            rating = preference_score

        return rating

    def get_resource_rating(self, rec_id):
        res_preferences, int_preferences, rec_preferences = self.user.profile["preferences"]

        if REWARD_TYPE == "thumbs":
            prob = 1 / (1 + np.exp(-res_preferences[rec_id]))
            reward_rand = np.random.rand() < prob
            rating = "liked" if reward_rand else "disliked"  # FIX AVG. PARAMS (SHOULD THEY SUM TO 0?)

        elif REWARD_TYPE == "float":
            rating = res_preferences[rec_id]

        return rating

    def simulate_feedback(self):
        events = []

        if self.user.weekly_recommendation_plan.get("plans") and self.user.intervention_week < self.num_weeks_per_user:
            # Step 1: Get contents sent in the past hour
            hour_contents = []
            for content in self.user.weekly_recommendation_plan["plans"]:
                ts = self.time_handler.parse_client_ts(content["scheduled_for"])
                if ts.day == self.time_handler.now.day and ts.hour == (self.time_handler.now - timedelta(hours=1)).hour:
                    hour_contents.append(content)

            # Step 2:
            for content in hour_contents:
                events.append(
                    {
                        "process_id": FeedbackManager.process_count,
                        "timestamp": content["scheduled_for"],
                        "event_name": "notification_sent",
                        "properties": {
                            "content_id": content["content_id"],
                            "content_type": content["type"],
                            "mission_id": content["mission_id"],
                            "is_end_mission": False,
                        },
                    }
                )
                if random.random() < OPEN_PROBABILITY:
                    open_timestamp = content["scheduled_for"]
                    events.append(
                        {
                            "process_id": FeedbackManager.process_count,
                            "timestamp": open_timestamp,
                            "event_name": "notification_opened",
                            "properties": {
                                "content_id": content["content_id"],
                                "content_type": content["type"],
                                "mission_id": content["mission_id"],
                                "is_end_mission": False,
                            },
                        }
                    )
                    if content["type"] == "recommendation":
                        self.user.track_opened_recommendations(
                            self.time_handler.parse_client_ts(open_timestamp),
                            FeedbackManager.process_count,
                            content["content_id"],
                            self.recommendations[content["content_id"]]["intervention_type"],
                        )

                    if random.random() < RATE_PROBABILITY:
                        events.append(
                            {
                                "process_id": FeedbackManager.process_count,
                                "timestamp": content["scheduled_for"],
                                "event_name": "notification_rated",
                                "properties": {
                                    "content_id": content["content_id"],
                                    "content_type": content["type"],
                                    "mission_id": content["mission_id"],
                                    "is_end_mission": False,
                                    "rating": self.get_rec_bias(content["mission_id"], content["content_id"])
                                    if content["type"] == "recommendation"
                                    else self.get_resource_rating(content["content_id"]),
                                },
                            }
                        )
                        self.user.current_rated_contents.append(content)
                FeedbackManager.process_count += 1

        # FIX IF MORE MISSIONS ACCOMPLISHED AT DIFFERENT TIMES
        if (
            (self.time_handler.now >= self.user.new_mission_expiration)
            and (self.user.intervention_day % 7 == 0)
            and (self.user.intervention_week < 12)
        ):
            for mission_id in self.user.current_missions:
                # DELAYED RECOMMENDATIONS AND RESOURCES FOR LAST ONES NOT RATED
                for content in self.user.get_contents_to_rate():
                    events.append(
                        {
                            "process_id": FeedbackManager.process_count,
                            "timestamp": content["scheduled_for"],
                            "event_name": "notification_rated",
                            "properties": {
                                "content_id": content["content_id"],
                                "content_type": content["type"],
                                "mission_id": content["mission_id"],
                                "is_end_mission": True,
                                "rating": self.get_rec_bias(content["mission_id"], content["content_id"])
                                if content["type"] == "recommendation"
                                else self.get_resource_rating(content["content_id"]),
                            },
                        }
                    )
                    FeedbackManager.process_count += 1

                events.append(
                    {
                        "process_id": FeedbackManager.process_count,
                        "timestamp": self.time_handler.utc_iso(self.time_handler.now),
                        "event_name": "mission_accomplished",
                        "properties": {
                            "mission_id": mission_id,
                            "score": random.random(),
                        },
                    }
                )
                FeedbackManager.process_count += 1

        return events
