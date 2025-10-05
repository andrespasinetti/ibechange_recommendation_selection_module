from datetime import timedelta
from cs_module.utils.encoding import get_intervention_feature_vector, get_intervention_encoding
from cs_module.utils.get_pillar import get_pillar
from cs_module.config import INTERVENTION_TYPES


def get_intervention_frequency_scheduled(intervention_type, intv_to_freq_offset):
    """
    Mixture-weighted scheduled intervention frequency for THIS item.
    - intervention_type: list[str] (the item's raw tags)
    - intv_to_freq_offset: dict[type -> float] mixture-weighted counts this week
    Returns unnormalised freq --> normalised in encoding
    """
    mix = get_intervention_encoding(intervention_type)  # len 8, sum=1 if non-empty
    # dot(mix, counters)
    sIF = sum(w * float(intv_to_freq_offset.get(t, 0.0)) for t, w in zip(INTERVENTION_TYPES, mix))
    return sIF


def get_mission_to_feature_vec_to_rec_ids(
    user,
    missions,
    recommendations,
    mission_id_to_avail_rec_ids,
    total_freq_offset,
    intv_to_freq_offset,
    rec_to_freq_offset,
    prompted=False,
):
    th = user.time_handler
    personal_data = user.get_personal_data()
    num_intervention_days = user.get_num_intervention_days()
    hhs = user.get_hhs()
    user_missions = user.get_new_missions()
    if not user_missions:
        return {}
    mission_to_feature_vec_to_rec = {m["mission"]: {} for m in user_missions}
    # ASSUMING ONE MISSION AT A TIME FOR PILOT STUDY
    sel_ts = th.parse_client_ts(user_missions[0]["selection_timestamp"])
    time_window = (sel_ts - timedelta(weeks=1), sel_ts)
    er_past = user.get_engagement_rate(time_window=time_window)
    prev_mission_score = user.get_previous_mission_score()

    for mission_id, avail_rec_ids in mission_id_to_avail_rec_ids.items():
        for rec_id in avail_rec_ids:
            rec = recommendations[rec_id]
            mission = missions[mission_id]
            intervention_type = rec["intervention_type"]
            intervention_frequency_scheduled = get_intervention_frequency_scheduled(
                intervention_type, intv_to_freq_offset
            )

            fv = get_intervention_feature_vector(
                personal_data,
                hhs,
                num_intervention_days,
                get_pillar(rec_id),
                mission_frequency=mission["weekly_frequency"],
                total_frequency_past_week=user.get_total_frequency(time_window),
                total_frequency_scheduled=total_freq_offset,
                intervention=intervention_type,
                intervention_frequency_past_week=user.get_intervention_frequency(intervention_type, time_window),
                intervention_frequency_scheduled=intervention_frequency_scheduled,
                recommendation_frequency_past_week=user.get_recommendation_frequency(rec_id, time_window),
                recommendation_frequency_scheduled=rec_to_freq_offset.get(rec_id, 0),
                er_past_value=er_past,
                prompted=prompted,
                prev_mission_score=prev_mission_score,
            )
            mission_to_feature_vec_to_rec[mission_id].setdefault(fv, []).append(rec_id)

    return mission_to_feature_vec_to_rec
