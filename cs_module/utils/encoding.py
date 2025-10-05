# --- Unified feature schema -----------------------------------------------

from itertools import product
from cs_module.config import (
    PILLARS,
    INTERVENTION_TYPES,
    LEAVE_OUT_VARS,
    PERSONAL_DATA_FEATURES,
    PERSONAL_DATA_CATEGORICAL_FEATURES,
    CATEGORICAL_TO_NUMERIC,
    CATEGORICAL_TO_NUMERIC_EXPLICIT,
    NUMERIC_FEATURES_MIN_MAX,
    FREQUENCY_FEATURE_DEGREES,
    INTERVENTION_MAB_FEATURES,
    RECOMMENDATION_MAB_FEATURES,
    MAX_NUM_REC_PER_MISSION,
    MAX_SAME_REC_SENT_PER_MISSION,
)

from .min_max_norm import min_max_norm


# ========== Helpers (small, reused) ==========


def one_hot_encode(value, all_values):
    return [1 if value == v else 0 for v in all_values]


def encode_frequency(value, degree, min_val, max_val):
    v = max(min_val, min(max_val, value))  # clip
    return [(v**d - min_val**d) / (max_val**d - min_val**d) for d in range(1, degree + 1)]


def _center(x):  # for interaction-only centering of [0,1] vars
    return x - 0.5


def _split_pairs(block):  # [past, sched]
    return block[0], block[1]


# ========== Personal data and atomic encoders ==========


def get_personal_data_dimension():
    dim = 0
    for feature in PERSONAL_DATA_FEATURES:
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            if feature in CATEGORICAL_TO_NUMERIC:
                dim += 1
            else:
                present = PERSONAL_DATA_CATEGORICAL_FEATURES[feature]
                dim += len(present)
                if feature in LEAVE_OUT_VARS:
                    dim -= sum(1 for v in LEAVE_OUT_VARS[feature] if v in present)
        else:
            dim += 1
    return dim


def get_personal_data_labels():
    labels = []
    for feature in PERSONAL_DATA_FEATURES:
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            if feature in CATEGORICAL_TO_NUMERIC:
                labels.append(feature)
            else:
                values = [
                    v for v in PERSONAL_DATA_CATEGORICAL_FEATURES[feature] if v not in LEAVE_OUT_VARS.get(feature, [])
                ]
                labels.extend(values)
        else:
            labels.append(feature)
    return labels


def get_personal_data_encoding(personal_data):
    enc = []
    for feature in PERSONAL_DATA_FEATURES:
        if feature in PERSONAL_DATA_CATEGORICAL_FEATURES:
            values = [
                v for v in PERSONAL_DATA_CATEGORICAL_FEATURES[feature] if v not in LEAVE_OUT_VARS.get(feature, [])
            ]
            if feature in CATEGORICAL_TO_NUMERIC:
                if feature not in personal_data or personal_data[feature] is None:
                    val = 0.5
                elif feature in CATEGORICAL_TO_NUMERIC_EXPLICIT:
                    val = CATEGORICAL_TO_NUMERIC_EXPLICIT[feature].get(personal_data.get(feature), 0.5)
                else:
                    val = (
                        0.5
                        if personal_data[feature] not in values
                        else values.index(personal_data[feature]) / (len(values) - 1)
                    )
            else:
                val = personal_data.get(feature)
                val = [0] * len(values) if val not in values else one_hot_encode(val, values)
        else:
            if feature not in personal_data or personal_data[feature] is None:
                val = 0.5
            else:
                val = personal_data[feature]
                if feature in NUMERIC_FEATURES_MIN_MAX:
                    a, b = NUMERIC_FEATURES_MIN_MAX[feature]
                    val = min_max_norm(val, a, b)

        if isinstance(val, list):
            enc.extend(val)
        else:
            enc.append(val)
    return enc


# ========== Core base blocks (order = output order) ==========

BASE_LABELS = {
    "D": get_personal_data_labels(),  # demographics
    "H": [p for p in PILLARS],  # (unused if flag False)
    "Hc": ["hhs_current"],  # current pillar HHS in [0,1]
    "ND": ["num_intervention_days"],
    "P": [f"pillar_{p}" for p in PILLARS if p not in LEAVE_OUT_VARS.get("pillar", [])],
    "MF": ["mission_frequency"],
    "TF": ["TF_past", "TF_sched"],
    "IT": [f"IT_{t}" for t in INTERVENTION_TYPES],
    "NIT": ["num_int_types"],
    "IF": ["IF_past", "IF_sched"],
    "RF": ["RF_past", "RF_sched"],
    "ER": ["engagement_rate_past"],
    "PR": ["prompted"],
    "MS": ["prev_mission_score"],
}

BASE_DIMENSIONS = {k: len(v) for k, v in BASE_LABELS.items()}


# ========== Atomic encoders for each block ==========


def get_hhs_encoding(hhs):
    return [hhs.get(p, 50) / 100.0 for p in PILLARS]


def get_hhs_current_encoding(hhs, pillar):
    return [hhs.get(pillar, 50) / 100.0]


def get_num_intervention_days_encoding(days):
    if days is None:
        return [0.0]
    v = days / (12 * 7)
    return [max(0.0, min(1.0, v))]


def get_pillar_encoding(pillar):
    if pillar not in PILLARS:
        raise ValueError(f"Invalid pillar: {pillar}")
    return one_hot_encode(pillar, [p for p in PILLARS if p not in LEAVE_OUT_VARS.get("pillar", [])])


def get_mission_frequency_encoding(mf):
    return encode_frequency(mf, FREQUENCY_FEATURE_DEGREES["MF"], min_val=1, max_val=7)


def get_total_frequency_encoding(x, *, scheduled):
    cap = MAX_NUM_REC_PER_MISSION - 1 if scheduled else MAX_NUM_REC_PER_MISSION
    return encode_frequency(x, FREQUENCY_FEATURE_DEGREES["TF"], 0, cap)


def get_intervention_encoding(intervention):
    intervention = intervention or []
    if not all(i in INTERVENTION_TYPES for i in intervention):
        raise ValueError(f"Invalid intervention types: {intervention}")
    mh = [1 if i in intervention else 0 for i in INTERVENTION_TYPES]
    s = sum(mh)
    return [0.0] * len(INTERVENTION_TYPES) if s == 0 else [x / s for x in mh]


def get_num_int_types_encoding(intervention):
    k = len(intervention or [])
    return [k / float(len(INTERVENTION_TYPES))]


def get_intervention_frequency_encoding(x, *, scheduled):
    cap = MAX_NUM_REC_PER_MISSION - 1 if scheduled else MAX_NUM_REC_PER_MISSION
    return encode_frequency(x, FREQUENCY_FEATURE_DEGREES["IF"], 0, cap)


def get_recommendation_frequency_encoding(x, *, scheduled):
    cap = MAX_SAME_REC_SENT_PER_MISSION - 1 if scheduled else MAX_SAME_REC_SENT_PER_MISSION
    return encode_frequency(x, FREQUENCY_FEATURE_DEGREES["RF"], 0, cap)


def get_engagement_rate_encoding(er_value):
    v = 0.5 if er_value is None else max(0.0, min(1.0, float(er_value)))
    return [v]


def get_prompted_encoding(prompted):
    return [1.0 if prompted else 0.0]


# ========== Grab positions used for interactions ==========


def _get_age_centered(D):
    age_idx = BASE_LABELS["D"].index("userAge")
    return _center(D[age_idx])


def _get_hhs_current_centered(H, pillar):
    idx = PILLARS.index(pillar)
    return _center(H[idx])


# ========== Public: labels & dims ==========


def get_intervention_feature_vector_labels():
    labs = ["bias"]
    # base blocks in order
    for key in BASE_LABELS:
        if INTERVENTION_MAB_FEATURES.get(key, False):
            labs.extend(BASE_LABELS[key])

    # custom interactions (scheduled-only, compact)
    if INTERVENTION_MAB_FEATURES.get("MF_x_TF_sched", False):
        labs.append("MF_x_TF_sched")
    if INTERVENTION_MAB_FEATURES.get("MF_x_RF_sched", False):
        labs.append("MF_x_RF_sched")
    if INTERVENTION_MAB_FEATURES.get("MF_x_IF_sched", False):
        labs.append("MF_x_IF_sched")
    if INTERVENTION_MAB_FEATURES.get("NIT_x_IF_sched", False):
        labs.append("NIT_x_IF_sched")

    if INTERVENTION_MAB_FEATURES.get("AGEc_x_RF_sched", False):
        labs.append("AGEc_x_RF_sched")
    if INTERVENTION_MAB_FEATURES.get("AGEc_x_IT", False):
        labs.extend([f"AGEc_x_IT_{t}" for t in INTERVENTION_TYPES])

    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_RF_sched", False):
        labs.append("HHS_c_x_RF_sched")
    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_IT", False):
        labs.extend([f"HHS_c_x_IT_{t}" for t in INTERVENTION_TYPES])

    # (optional) classic cartesian families
    INTERACTION_PAIRS = {
        "D_H": ("D", "H"),
        "D_P": ("D", "P"),
        "D_IT": ("D", "IT"),
        "D_MF": ("D", "MF"),
        "D_TF": ("D", "TF"),
        "D_IF": ("D", "IF"),
        "D_RF": ("D", "RF"),
        "P_IT": ("P", "IT"),
        "P_MF": ("P", "MF"),
        "P_TF": ("P", "TF"),
        "P_IF": ("P", "IF"),
        "I_IF": ("IT", "IF"),
        "I_RF": ("IT", "RF"),
    }
    for inter_key, (a, b) in INTERACTION_PAIRS.items():
        if INTERVENTION_MAB_FEATURES.get(inter_key, False):
            labs.extend([f"{la}_{lb}" for la in BASE_LABELS[a] for lb in BASE_LABELS[b]])

    return labs


def get_dim_intervention_feature_vector(include_bias=True):
    dim = 1 if include_bias else 0
    for key in BASE_LABELS:
        if INTERVENTION_MAB_FEATURES.get(key, False):
            dim += BASE_DIMENSIONS[key]

    # custom interactions
    if INTERVENTION_MAB_FEATURES.get("MF_x_TF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("MF_x_RF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("MF_x_IF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("NIT_x_IF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("AGEc_x_RF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("AGEc_x_IT", False):
        dim += len(INTERVENTION_TYPES)
    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_RF_sched", False):
        dim += 1
    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_IT", False):
        dim += len(INTERVENTION_TYPES)

    # optional cartesian families
    INTERACTION_PAIRS = {
        "D_H": ("D", "H"),
        "D_P": ("D", "P"),
        "D_IT": ("D", "IT"),
        "D_MF": ("D", "MF"),
        "D_TF": ("D", "TF"),
        "D_IF": ("D", "IF"),
        "D_RF": ("D", "RF"),
        "P_IT": ("P", "IT"),
        "P_MF": ("P", "MF"),
        "P_TF": ("P", "TF"),
        "P_IF": ("P", "IF"),
        "I_IF": ("IT", "IF"),
        "I_RF": ("IT", "RF"),
    }
    for inter_key, (a, b) in INTERACTION_PAIRS.items():
        if INTERVENTION_MAB_FEATURES.get(inter_key, False):
            dim += BASE_DIMENSIONS[a] * BASE_DIMENSIONS[b]

    return dim


# ========== Public: feature vector builder ==========


def get_encodings(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_scheduled,
    intervention,
    intervention_frequency_past_week,
    intervention_frequency_scheduled,
    recommendation_frequency_past_week,
    recommendation_frequency_scheduled,
    er_past_value=None,
    prompted=False,
    prev_mission_score=0.0,
):
    D = get_personal_data_encoding(personal_data)
    H = get_hhs_encoding(hhs)
    Hc = get_hhs_current_encoding(hhs, pillar)
    ND = get_num_intervention_days_encoding(num_intervention_days)
    P = get_pillar_encoding(pillar)
    MF = get_mission_frequency_encoding(mission_frequency)
    TF = get_total_frequency_encoding(total_frequency_past_week, scheduled=False) + get_total_frequency_encoding(
        total_frequency_scheduled, scheduled=True
    )
    IT = get_intervention_encoding(intervention)
    NIT = get_num_int_types_encoding(intervention)
    IF = get_intervention_frequency_encoding(
        intervention_frequency_past_week, scheduled=False
    ) + get_intervention_frequency_encoding(intervention_frequency_scheduled, scheduled=True)
    RF = get_recommendation_frequency_encoding(
        recommendation_frequency_past_week, scheduled=False
    ) + get_recommendation_frequency_encoding(recommendation_frequency_scheduled, scheduled=True)
    ER = get_engagement_rate_encoding(er_past_value if er_past_value is not None else 0.0)
    PR = get_prompted_encoding(prompted)
    MS = [max(0.0, min(1.0, float(prev_mission_score)))]
    return D, H, Hc, ND, P, MF, TF, IT, NIT, IF, RF, ER, PR, MS


def get_intervention_feature_vector(
    personal_data,
    hhs,
    num_intervention_days,
    pillar,
    mission_frequency,
    total_frequency_past_week,
    total_frequency_scheduled,
    intervention,
    intervention_frequency_past_week,
    intervention_frequency_scheduled,
    recommendation_frequency_past_week,
    recommendation_frequency_scheduled,
    er_past_value=None,
    prompted=False,
    prev_mission_score=0.0,
):
    D, H, Hc, ND, P, MF, TF, IT, NIT, IF, RF, ER, PR, MS = get_encodings(
        personal_data,
        hhs,
        num_intervention_days,
        pillar,
        mission_frequency,
        total_frequency_past_week,
        total_frequency_scheduled,
        intervention,
        intervention_frequency_past_week,
        intervention_frequency_scheduled,
        recommendation_frequency_past_week,
        recommendation_frequency_scheduled,
        er_past_value,
        prompted,
        prev_mission_score,
    )

    fv = [1]  # bias

    # 1) Base blocks in order
    base_parts = {
        "D": D,
        "H": H,
        "Hc": Hc,
        "ND": ND,
        "P": P,
        "MF": MF,
        "TF": TF,
        "IT": IT,
        "NIT": NIT,
        "IF": IF,
        "RF": RF,
        "ER": ER,
        "PR": PR,
        "MS": MS,
    }
    for key in BASE_LABELS:
        if INTERVENTION_MAB_FEATURES.get(key, False):
            fv.extend(base_parts[key])

    # 2) Custom scheduled-only interactions
    TF_past, TF_sched = _split_pairs(TF)
    RF_past, RF_sched = _split_pairs(RF)
    IF_past, IF_sched = _split_pairs(IF)

    if INTERVENTION_MAB_FEATURES.get("MF_x_TF_sched", False):
        fv.append(MF[0] * TF_sched)
    if INTERVENTION_MAB_FEATURES.get("MF_x_RF_sched", False):
        fv.append(MF[0] * RF_sched)
    if INTERVENTION_MAB_FEATURES.get("MF_x_IF_sched", False):
        fv.append(MF[0] * IF_sched)
    if INTERVENTION_MAB_FEATURES.get("NIT_x_IF_sched", False):
        fv.append(NIT[0] * IF_sched)

    age_c = _get_age_centered(D)
    if INTERVENTION_MAB_FEATURES.get("AGEc_x_RF_sched", False):
        fv.append(age_c * RF_sched)
    if INTERVENTION_MAB_FEATURES.get("AGEc_x_IT", False):
        fv.extend([age_c * w for w in IT])

    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_RF_sched", False):
        fv.append(_get_hhs_current_centered(H, pillar) * RF_sched)
    if INTERVENTION_MAB_FEATURES.get("HHS_c_x_IT", False):
        hhs_c = _get_hhs_current_centered(H, pillar)
        fv.extend([hhs_c * w for w in IT])

    # 3) Optional cartesian families (kept behind flags)
    INTERACTION_PAIRS = {
        "D_H": ("D", "H"),
        "D_P": ("D", "P"),
        "D_IT": ("D", "IT"),
        "D_MF": ("D", "MF"),
        "D_TF": ("D", "TF"),
        "D_IF": ("D", "IF"),
        "D_RF": ("D", "RF"),
        "P_IT": ("P", "IT"),
        "P_MF": ("P", "MF"),
        "P_TF": ("P", "TF"),
        "P_IF": ("P", "IF"),
        "I_IF": ("IT", "IF"),
        "I_RF": ("IT", "RF"),
    }
    blockvals = {"D": D, "H": H, "P": P, "MF": MF, "TF": TF, "IT": IT, "IF": IF, "RF": RF}
    for inter_key, (a, b) in INTERACTION_PAIRS.items():
        if INTERVENTION_MAB_FEATURES.get(inter_key, False):
            A, B = blockvals[a], blockvals[b]
            fv.extend([x * y for x, y in product(A, B)])

    return tuple(fv)


def get_recommendation_feature_vector(recommendation_frequency=None):
    fv = [1]
    if RECOMMENDATION_MAB_FEATURES.get("RF", False):
        if recommendation_frequency is None:
            raise ValueError("recommendation_frequency must be provided when RF feature is enabled.")
        fv.extend(get_recommendation_frequency_encoding(recommendation_frequency, scheduled=False))
    return tuple(fv)
