from cs_module.config import MAX_SAME_REC_SENT_PER_MISSION, MAX_DISTINCT_REC_PER_MISSION, INTERVENTION_TYPES
from cs_module.utils.encoding import get_intervention_encoding  # normalized mixture


def update_frequency_offsets(
    sel_rec_id,
    mission_id,
    mission_to_selected_rec_to_count,
    total_freq_offset,
    intv_to_freq_offset,
    rec_to_freq_offset,
    mission_to_available_recs,
    recommendations,
):
    # count selection
    mission_to_selected_rec_to_count.setdefault(mission_id, {})
    mission_to_selected_rec_to_count[mission_id][sel_rec_id] = (
        mission_to_selected_rec_to_count[mission_id].get(sel_rec_id, 0) + 1
    )
    # update offsets
    total_freq_offset += 1

    mix = get_intervention_encoding(recommendations[sel_rec_id]["intervention_type"])  # len=8 weights
    for itype, w in zip(INTERVENTION_TYPES, mix):
        if w:
            intv_to_freq_offset[itype] = intv_to_freq_offset.get(itype, 0.0) + w

    rec_to_freq_offset[sel_rec_id] = rec_to_freq_offset.get(sel_rec_id, 0) + 1

    # enforce repetition limits
    if mission_to_selected_rec_to_count[mission_id][sel_rec_id] >= MAX_SAME_REC_SENT_PER_MISSION:
        mission_to_available_recs[mission_id].remove(sel_rec_id)

    if len(mission_to_selected_rec_to_count[mission_id]) >= MAX_DISTINCT_REC_PER_MISSION:
        curr = set(mission_to_available_recs[mission_id])
        used = set(mission_to_selected_rec_to_count[mission_id].keys())
        mission_to_available_recs[mission_id] = list(curr & used)

    return (
        mission_to_selected_rec_to_count,
        total_freq_offset,
        intv_to_freq_offset,
        rec_to_freq_offset,
        mission_to_available_recs,
    )
