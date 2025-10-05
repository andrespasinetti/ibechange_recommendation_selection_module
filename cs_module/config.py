import os

# --- Time ---
USE_REAL_TIME = os.getenv("CS_USE_REAL_TIME", "true").lower() == "true"

# --- Features ---
PERSONAL_DATA_FEATURES = ["gender", "userAge", "education", "recruitmentCenter"]
PERSONAL_DATA_CATEGORICAL_FEATURES = {
    "gender": ["female", "male", "decline", "other"],
    "recruitmentCenter": ["IEO", "ICO", "UMFCD", "UNIPA"],
    "education": ["no-education", "primary", "secondary", "vocational", "university", "postgraduate", "other"],
}
CATEGORICAL_TO_NUMERIC = ["education", "gender"]
CATEGORICAL_TO_NUMERIC_EXPLICIT = {
    "gender": {"female": 0, "decline": 0.5, "other": 0.5, "male": 1},
}
NUMERIC_FEATURES_MIN_MAX = {"userAge": [45, 80]}

PILLARS = ["smoking", "alcohol", "nutrition", "physical_activity", "emotional_wellbeing"]
PILLARS_WITH_COMPONENTS = {"nutrition", "emotional_wellbeing"}

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

LEAVE_OUT_VARS = {
    "recruitmentCenter": ["IEO"],
    "education": ["other"],
    "pillar": ["smoking"],  # pilot: one pillar at a time
}

# --- Rewards / bandits ---
REWARD_TYPE = "thumbs"  # "thumbs" or "float"

RESOURCE_MAB_CONFIG = {"type": "BernoulliBetaTS"}
INTERVENTION_MAB_CONFIG = {"type": "LogisticLaplaceTS"}
RECOMMENDATION_MAB_CONFIG = {"type": "BernoulliBetaTS"}

# --- Weekly caps ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 10
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 4

# --- Frequency featurization (keep all degree-1) ---
FREQUENCY_FEATURE_DEGREES = {"MF": 1, "TF": 1, "IF": 1, "RF": 1}

# ================================
# Feature flags (single logic)
# ================================
# Base blocks (main effects)
INTERVENTION_MAB_FEATURES = {
    "D": True,  # Demographics
    "H": True,  # Current pillar score (scalar in [0,1])
    "ND": True,  # Days since start (normalized)
    "P": True,  # Pillar one-hot (LOO)
    "MF": True,  # Mission frequency
    "TF": True,  # Total freq (past, sched)
    "IT": True,  # Intervention type mixture (8-dim, normalized)
    "NIT": True,  # Num intervention types (scalar)
    "IF": True,  # Mixture-weighted intervention freq (past, sched)
    "RF": True,  # Same-item freq (past, sched)
    "ER": False,  # Engagement rate (past week)
    "PR": True,  # Prompted flag
    "MS": False,  # Previous mission score 
    # Curated interactions only (scheduled where meaningful)
    "MF_x_TF_sched": False,  # mostly intercept/threshold shift; keep off unless you need it
    "MF_x_RF_sched": True,
    "MF_x_IF_sched": True,
    "AGEc_x_RF_sched": True,
    "AGEc_x_IT": True,
    "H_c_x_RF_sched": False,
    "H_c_x_IT": False,
    "NIT_x_IF_sched": False,
}

# Legacy cartesian interactions are intentionally removed to avoid the “second logic”.
# If you must keep the keys around for backward compatibility, set them all to False:
# for k in ("D_H","D_P","D_IT","D_IF","D_MF","D_TF","P_IT","P_MF","P_TF","P_IF","IT_IF","I_RF","P_RF","D_RF"):
#     INTERVENTION_MAB_FEATURES[k] = False

RECOMMENDATION_MAB_FEATURES = {"RF": False}
