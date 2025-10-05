import random
import numpy as np
from virtual_user.user import User
from virtual_user.config import NUMERIC_FEATURES_MIN_MAX, PERSONAL_DATA_CATEGORICAL_FEATURES
import os
import json
import uuid


class UserFactory:
    def __init__(self, time_handler, content_manager, num_weeks_per_user):
        self.time_handler = time_handler
        self.num_weeks_per_user = num_weeks_per_user
        self.preferences = self.load_user_preferences()
        self.config = self.default_config()
        # Habit scores distributions (defined here)
        self.smoking_dist = lambda: random.randint(0, 100)
        self.alcohol_dist = lambda: random.randint(0, 100)
        self.nutrition_dist = lambda: random.randint(0, 100)
        self.physical_activity_dist = lambda: random.randint(0, 100)
        self.emotional_score_dist = lambda: random.randint(0, 100)

        self.content_manager = content_manager

    def load_user_preferences(self):
        folder = os.path.join(os.path.dirname(__file__), "..", "user_preferences")
        filenames = ["res_preferences.json", "int_preferences.json", "rec_preferences.json"]
        loaded_files = []

        for name in filenames:
            path = os.path.join(folder, name)
            with open(path, "r") as file:
                loaded_files.append(json.load(file))

        return loaded_files

    def default_config(self):
        return {
            "userAge": lambda: random.randint(
                NUMERIC_FEATURES_MIN_MAX["userAge"][0], NUMERIC_FEATURES_MIN_MAX["userAge"][1]
            ),
            "gender": lambda: random.choice(["female", "male"]),
            "recruitmentCenter": lambda: random.choice(PERSONAL_DATA_CATEGORICAL_FEATURES["recruitmentCenter"]),
            "education": lambda: random.choice(PERSONAL_DATA_CATEGORICAL_FEATURES["education"]),
            # "open_probability": lambda: min(1.0, max(0.0, np.random.normal(0.8, 0.05))),
            # "rate_probability": lambda: min(1.0, max(0.0, np.random.normal(0.6, 0.1))),
            "pillar_retain_probability": lambda: np.random.uniform(0.0, 1.0),
            "mission_retain_probability": lambda: np.random.uniform(0.0, 1.0),
            "mission_achieve_probability": lambda: np.random.uniform(0.0, 1.0),
            "preferences": self.preferences,
            "height": lambda: int(np.clip(random.gauss(170, 10), 140, 200)),  # cm
            "weight": lambda: int(np.clip(random.gauss(70, 15), 45, 150)),  # kg
            "enrolmentDate": self.time_handler.utc_iso(self.time_handler.now),
            "wearable": lambda: random.choice(["yes", "no"]),
            "voiceRecording": lambda: random.choice(["yes", "no"]),
            "occupation": lambda: random.choice(["employed", "unemployed", "student", "retired", "other"]),
            "digitalLiteracy": lambda: random.choice(["low", "moderate", "high"]),
            "level": 0,
        }

    def sample_user_profile(self):
        """Sample a user profile based on the defined config."""
        profile = {}
        for key, sampler in self.config.items():
            profile[key] = sampler() if callable(sampler) else sampler
        return profile

    def sample_hhs(self):
        """Sample the health habit score."""
        return {
            "smoking": self.smoking_dist(),
            "alcohol": self.alcohol_dist(),
            "nutrition": self.nutrition_dist(),
            "physical_activity": self.physical_activity_dist(),
            "emotional_wellbeing": self.emotional_score_dist(),
        }

    def generate_users(self, count):
        """Generate new users with unique IDs and random profiles."""
        users = {}
        for i in range(count):
            profile = self.sample_user_profile()  # Assuming this method exists to generate profiles
            user_id = str(uuid.uuid4())
            user = User(
                self.time_handler,
                user_id,
                profile,
                num_weeks_per_user=self.num_weeks_per_user,
                content_manager=self.content_manager,
            )

            users[user_id] = user

        return users
