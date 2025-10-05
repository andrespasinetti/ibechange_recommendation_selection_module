from datetime import datetime, timedelta, timezone

    
# --- Time ---
USE_REAL_TIME = False
ENTRANCE_TIMES = [datetime(2025, 9, 1, 9, 0, 0, tzinfo=timezone.utc)] # + timedelta(weeks=i) for i in range(3)]
NUM_WEEKLY_USERS = 90

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
LEAVE_OUT_VARS = {
    "recruitmentCenter": ["IEO"],
    "education": ["other"],
    "pillar": ["smoking"],  # BECAUSE IN PILOT ONE PILLAR AT A TIME
}

# --- User Preferences ---
REWARD_TYPE = "thumbs"  # "float" or "thumbs"
LOGISTIC_STEEPNESS = 1

OPEN_PROBABILITY = 1
RATE_PROBABILITY = 1

REC_PREFERENCE_RANGE = (0.0, 0.0)
PREFERENCES = {'bias': 0.0, 'gender': 0.2987612613790629, 'userAge': 0.14451264830327548, 'education': -0.0768814795128897, 'recruitmentCenter_ICO': -0.48087474473552416, 'recruitmentCenter_UMFCD': -0.06451688648161813, 'recruitmentCenter_UNIPA': -0.10988657728739204, 'HHS_smoking': 0.4449981242056764, 'HHS_alcohol': 0.4735752387680351, 'HHS_nutrition': 0.4568627922262275, 'HHS_physical_activity': 0.4801730623169138, 'HHS_emotional_wellbeing': -0.33109579527389066, 'num_intervention_days': -0.4703301545821664, 'PILLAR_alcohol': -0.20792319096406475, 'PILLAR_nutrition': -0.492087300727142, 'PILLAR_physical_activity': -0.38115792279846106, 'PILLAR_emotional_wellbeing': 0.07718142126362104, 'mission_frequency': 0.10046022176956665, 'TOTAL_FREQ_past_week': 0.11073725001565549, 'TOTAL_FREQ_scheduled': -0.30968136524334755, 'INTERVENTION_Education': -0.22592431537161173, 'INTERVENTION_Enablement': 0.469305343085642, 'INTERVENTION_Environmental_Restructuring': -0.31373155882848713, 'INTERVENTION_Incentivisation': 0.38236180002929043, 'INTERVENTION_Modelling': 0.026828866633933246, 'INTERVENTION_Training': 0.1629654266537821, 'INTERVENTION_Persuasion': -0.3559933762252626, 'INTERVENTION_Restrictions': 0.2449563847564682, 'INTERVENTION_FREQ_past_week': 0.389567652208593, 'INTERVENTION_FREQ_scheduled': -0.4009501624660932, 'RECOMMENDATION_FREQ_past_week': -0.027217788366925122, 'RECOMMENDATION_FREQ_scheduled': -0.17335358362776632}
MISSION_SELECTION_MODE = "user_specific"  # "random", "fixed", "user_keep_pillar", "user_specific"


# --- MAB ---
RESOURCE_MAB_CONFIG = { 
    "type": "ResourceOptimalBandit", 
}
INTERVENTION_MAB_CONFIG = {
    "type": "None",
}
RECOMMENDATION_MAB_CONFIG = { 
    "type": "RecommendationOptimalBandit", 
}


# --- MAB / Frequency Settings ---
MAX_SAME_REC_SENT_PER_MISSION = 3
MAX_NUM_REC_PER_MISSION = 10
MIN_NUM_REC_PER_MISSION = 3
MAX_DISTINCT_REC_PER_MISSION = 4



# --- Features / Encoding ---
FREQUENCY_FEATURE_DEGREES = { "MF": 1, "TF": 1, "IF": 1, "RF": 1 }
INTERVENTION_MAB_FEATURES = {
    "D": True,
    "H": True,
    "ND": True,
    "P": True,
    "MF": True,
    "TF": True,
    "I": True,
    "IF": True,
    "RF": True,
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
}
RECOMMENDATION_MAB_FEATURES = { "RF": False }
