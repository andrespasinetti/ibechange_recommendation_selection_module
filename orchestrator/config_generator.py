import random
import os
import json
from datetime import datetime

EXPERIMENTS_TO_RUN = 10

"""
PREFERENCE_KEYS = [
    "bias",
    "gender",
    "userAge",
    "education",
    "recruitmentCenter_ICO",
    "recruitmentCenter_UMFCD",
    "recruitmentCenter_UNIPA",
    "HHS_smoking",
    "HHS_alcohol",
    "HHS_nutrition",
    "HHS_physical_activity",
    "HHS_emotional_wellbeing",
    "num_intervention_days",
    "PILLAR_alcohol",
    "PILLAR_nutrition",
    "PILLAR_physical_activity",
    "PILLAR_emotional_wellbeing",
    "mission_frequency",
    "TOTAL_FREQ_past_week",
    "TOTAL_FREQ_scheduled",
    "INTERVENTION_Education",
    "INTERVENTION_Enablement",
    "INTERVENTION_Environmental_Restructuring",
    "INTERVENTION_Incentivisation",
    "INTERVENTION_Modelling",
    "INTERVENTION_Training",
    "INTERVENTION_Persuasion",
    "INTERVENTION_Restrictions",
    "INTERVENTION_FREQ_past_week",
    "INTERVENTION_FREQ_scheduled",
    "RECOMMENDATION_FREQ_past_week",
    "RECOMMENDATION_FREQ_scheduled",
    "
]
"""

INT_PREFERENCE_RANGES = {
    # Fixed bias so base like-rate ≈ 0.5
    "bias": (0.0, 0.0),
    # Demographics
    "gender": (-0.5, 0.5),
    "userAge": (-0.5, 0.5),
    "education": (-0.5, 0.5),
    "recruitmentCenter_ICO": (-0.5, 0.5),
    "recruitmentCenter_UMFCD": (-0.5, 0.5),
    "recruitmentCenter_UNIPA": (-0.5, 0.5),
    # Health habits (bigger potential effect)
    "HHS_smoking": (-0.5, 0.5),
    "HHS_alcohol": (-0.5, 0.5),
    "HHS_nutrition": (-0.5, 0.5),
    "HHS_physical_activity": (-0.5, 0.5),
    "HHS_emotional_wellbeing": (-0.5, 0.5),
    # Intervention history
    "num_intervention_days": (-0.5, 0.5),
    # Pillar one-hots
    "PILLAR_alcohol": (-0.5, 0.5),
    "PILLAR_nutrition": (-0.5, 0.5),
    "PILLAR_physical_activity": (-0.5, 0.5),
    "PILLAR_emotional_wellbeing": (-0.5, 0.5),
    # Mission frequency
    "mission_frequency": (0.0, 0.5),
    # Total frequency features
    "TOTAL_FREQ_past_week": (-0.5, 0.5),
    "TOTAL_FREQ_scheduled": (-0.5, 0.0),  # fatigue → negative only
    # Intervention type one-hots
    "INTERVENTION_Education": (-0.5, 0.5),
    "INTERVENTION_Enablement": (-0.5, 0.5),
    "INTERVENTION_Environmental_Restructuring": (-0.5, 0.5),
    "INTERVENTION_Incentivisation": (-0.5, 0.5),
    "INTERVENTION_Modelling": (-0.5, 0.5),
    "INTERVENTION_Training": (-0.5, 0.5),
    "INTERVENTION_Persuasion": (-0.5, 0.5),
    "INTERVENTION_Restrictions": (-0.5, 0.5),
    # Intervention frequency features
    "INTERVENTION_FREQ_past_week": (-0.5, 0.5),
    "INTERVENTION_FREQ_scheduled": (-0.5, 0.0),  # fatigue → negative only
    # Recommendation frequency features
    "RECOMMENDATION_FREQ_past_week": (-0.5, 0.5),
    "RECOMMENDATION_FREQ_scheduled": (-0.5, 0.0),  # fatigue → negative only
    # End of week
    "prompted": (-0.5, 0.5),
}

REC_PREFERENCE_RANGE = (0.0, 0.0)  # (0.0, 0.0), (-0.5, 0.5)
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "generated_configs")


def load_json_files():
    folder = os.path.join(os.path.dirname(__file__), "contents")
    filenames = ["missions.json", "recommendations.json", "resources.json"]
    loaded_files = []

    for name in filenames:
        path = os.path.join(folder, name)
        with open(path, "r") as file:
            loaded_files.append(json.load(file))

    return tuple(loaded_files)


# 2) Sampling util (uniform on [lo, hi])
def _sample_from_range(lo: float, hi: float) -> float:
    return lo if lo == hi else random.uniform(lo, hi)


# 3) Main generator — returns (prefs_dict, theta_vector_in_order)
def generate_int_preferences(seed: int | None = None):
    """
    Sample a fresh preference vector using PREFERENCE_RANGES.
    - prefs: dict {feature_name: weight}
    - theta: list of weights in the SAME order as PREFERENCE_RANGES keys
    """
    if seed is not None:
        random.seed(seed)

    prefs = {}
    theta = []
    for k, (lo, hi) in INT_PREFERENCE_RANGES.items():
        w = _sample_from_range(lo, hi)
        prefs[k] = w
        theta.append(w)
    return prefs, theta


def generate_preferences():
    missions, recommendations, resources = load_json_files()
    rec_preferences = {}
    for rec_id in list(recommendations.keys()):
        rec_preferences[rec_id] = random.uniform(REC_PREFERENCE_RANGE[0], REC_PREFERENCE_RANGE[1])

    # This makes the base rate controlled by bias, not by chance
    average = sum(rec_preferences.values()) / len(rec_preferences)
    rec_preferences = {key: value - average for key, value in rec_preferences.items()}

    res_preferences = {}
    for res_id in list(resources.keys()):
        res_preferences[res_id] = random.uniform(REC_PREFERENCE_RANGE[0], REC_PREFERENCE_RANGE[1])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the rec_preferences to a JSON file
    with open(f"{OUT_PATH}/user_preferences/rec_preferences.json", "w") as f:
        json.dump(rec_preferences, f, indent=4)
    with open(f"{OUT_PATH}/user_preferences_storage/{timestamp}_rec_preferences.json", "w") as f:
        json.dump(rec_preferences, f, indent=4)

    # Save the res_preferences to a JSON file
    with open(f"{OUT_PATH}/user_preferences/res_preferences.json", "w") as f:
        json.dump(res_preferences, f, indent=4)
    with open(f"{OUT_PATH}/user_preferences_storage/{timestamp}_res_preferences.json", "w") as f:
        json.dump(res_preferences, f, indent=4)

    int_preferences, theta = generate_int_preferences()
    with open(f"{OUT_PATH}/user_preferences/int_preferences.json", "w") as f:
        json.dump(theta, f, indent=4)
    with open(f"{OUT_PATH}/user_preferences_storage/{timestamp}_int_preferences.json", "w") as f:
        json.dump(int_preferences, f, indent=4)
    return int_preferences, theta


def generate_config(prefs, theta, mab_type):
    """
    Generates a fresh config.py with randomized PREFERENCES and matching theta,
    writing it to /generated_configs/config.py (relative to where this is run).
    """

    CONFIG_PATH = os.path.join(OUT_PATH, "config.py")
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)

    if mab_type == "Random":
        intervention_mab_type = "None"
        recommendation_mab_type = "RandomBandit"
        resource_mab_type = "RandomBandit"
    elif mab_type == "Learning":
        intervention_mab_type = "LogisticLaplaceTS"  # "LogisticLaplaceTS"
        recommendation_mab_type = "BernoulliBetaTS"  # "BernoulliBetaTS"
        resource_mab_type = "BernoulliBetaTS"  # "BernoulliBetaTS"
    elif mab_type == "Optimal":
        intervention_mab_type = "None"
        recommendation_mab_type = "RecommendationOptimalBandit"
        resource_mab_type = "ResourceOptimalBandit"

    # compose the new config.py content
    content = f"""from datetime import datetime, timedelta, timezone

    
# --- Time ---
USE_REAL_TIME = False
ENTRANCE_TIMES = [datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)] # + timedelta(weeks=i) for i in range(3)]
NUM_WEEKLY_USERS = 90

# --- Features ---
PERSONAL_DATA_FEATURES = ["gender", "userAge", "education", "recruitmentCenter"]
PERSONAL_DATA_CATEGORICAL_FEATURES = {{
    "gender": ["female", "male", "decline", "other"],
    "recruitmentCenter": ["IEO", "ICO", "UMFCD", "UNIPA"],
    "education": ["no-education", "primary", "secondary", "vocational", "university", "postgraduate", "other"],
}}
CATEGORICAL_TO_NUMERIC = ["education", "gender"]
CATEGORICAL_TO_NUMERIC_EXPLICIT = {{
    "gender": {{"female": 0, "decline": 0.5, "other": 0.5, "male": 1}},
}}
NUMERIC_FEATURES_MIN_MAX = {{"userAge": [45, 80]}}

PILLARS = ["smoking", "alcohol", "nutrition", "physical_activity", "emotional_wellbeing"]
INTERVENTION_TYPES = [
    "Education",
    "Enablement",
    "Environmental Restructuring",
    "Incentivisation",
    "Modelling",
    "Training",
    "Persuasion",
    "Restrictions",
]

# no INTERVENTION since there are None
# education we want it to be an integer feature --> "other" is middle value
LEAVE_OUT_VARS = {{
    "recruitmentCenter": ["IEO"],
    "education": ["other"],
    "pillar": ["smoking"],  # BECAUSE IN PILOT ONE PILLAR AT A TIME
}}

# --- User Preferences ---
REWARD_TYPE = "thumbs"  # "float" or "thumbs"
LOGISTIC_STEEPNESS = 1

OPEN_PROBABILITY = 1
RATE_PROBABILITY = 1

REC_PREFERENCE_RANGE = {REC_PREFERENCE_RANGE}
PREFERENCES = {prefs!r}
MISSION_SELECTION_MODE = "user_specific"  # "random", "fixed", "user_keep_pillar", "user_specific"


# --- MAB ---
RESOURCE_MAB_CONFIG = {{ 
    "type": "{resource_mab_type}", 
}}
INTERVENTION_MAB_CONFIG = {{
    "type": "{intervention_mab_type}",
}}
RECOMMENDATION_MAB_CONFIG = {{ 
    "type": "{recommendation_mab_type}", 
}}


# --- MAB / Frequency Settings ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 10
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 4



# --- Features / Encoding ---
FREQUENCY_FEATURE_DEGREES = {{ "MF": 1, "TF": 1, "IF": 1, "RF": 1 }}
INTERVENTION_MAB_FEATURES = {{
    "D": True,
    "H": True,
    "ND": True,
    "P": True,
    "MF": True,
    "TF": True,
    "I": True,
    "IF": True,
    "RF": True,
    "PR": True,
    "D_H": False,
    "D_P": False,
    "D_I": False,
    "D_IF": False,
    "D_MF": False,
    "D_TF": False,
    "P_I": False,
    "P_MF": False,
    "P_TF": False,
    "P_IF": False,
    "I_IF": False,
}}
RECOMMENDATION_MAB_FEATURES = {{ "RF": False }}
"""

    # write it out
    with open(CONFIG_PATH, "w") as f:
        f.write(content)
