from cs_module.config import (
    RECOMMENDATION_MAB_CONFIG,
    INTERVENTION_MAB_CONFIG,
)


def select_recommendation(
    feature_vec_to_rec_ids,
    intervention_mab,
    recommendation_mab,
    data_storage,
    time_handler,
    user,
    select_anyway,
    selection_id,
):
    if not feature_vec_to_rec_ids:  # Some missions have no rec (e.g., EM88), or no more available
        return None, None

    cfg = INTERVENTION_MAB_CONFIG["type"]

    """    
    SRc52: This recommendation must appear only once during the intervention but is mandatory.
    --> first one to be selected if not already
    """
    fv_with_s52 = next((fv for fv, ids in feature_vec_to_rec_ids.items() if "SRc52" in ids), None)
    if fv_with_s52 is not None:
        if "SRc52" not in user.only_one_rec_already:
            feature_vec_to_rec_ids = {fv_with_s52: ["SRc52"]}
        else:
            # remove SRc52 cleanly without mutating during iteration
            new_map = {}
            for fv, ids in feature_vec_to_rec_ids.items():
                new_ids = [rid for rid in ids if rid != "SRc52"]
                if new_ids:
                    new_map[fv] = new_ids
            feature_vec_to_rec_ids = new_map

    if cfg == "None":
        return handle_recommendation_mab_only(
            user, feature_vec_to_rec_ids, recommendation_mab, data_storage, time_handler, select_anyway, selection_id
        )  # returns (sel, None)

    return handle_intervention_mab(
        cfg,
        feature_vec_to_rec_ids,
        intervention_mab,
        recommendation_mab,
        user,
        data_storage,
        time_handler,
        select_anyway,
        selection_id,
    )


def handle_recommendation_mab_only(
    user, feature_vec_to_rec_ids, recommendation_mab, data_storage, time_handler, select_anyway, selection_id
):
    rec_cfg = RECOMMENDATION_MAB_CONFIG["type"]
    rec_ids = [rid for fv in feature_vec_to_rec_ids for rid in feature_vec_to_rec_ids[fv]]

    if rec_cfg == "RecommendationOptimalBandit":
        sel, sampled = recommendation_mab.select_action(
            [feature_vec_to_rec_ids[fv] for fv in feature_vec_to_rec_ids], list(feature_vec_to_rec_ids)
        )
    elif rec_cfg == "RandomBandit":
        sel, sampled = recommendation_mab.select_action(rec_ids)
    else:
        raise ValueError(f"Unknown recommendation_mab_mode: {rec_cfg}")

    if not select_anyway and sampled["estimated_reward"] <= 0.5:
        return None, None

    data_storage.add_mab_sample(
        table="recommendation_mab_samples",
        record={
            "user_id": user.user_id,
            "plan_id": selection_id["plan_id"],
            "content_count": selection_id["content_count"],
            "timestamp": time_handler.utc_iso(time_handler.now),
            "sample": sampled,
        },
    )
    return sel, None


def handle_intervention_mab(
    cfg,
    feature_vec_to_rec_ids,
    intervention_mab,
    recommendation_mab,
    user,
    data_storage,
    time_handler,
    select_anyway,
    selection_id,
):
    fvs = list(feature_vec_to_rec_ids.keys())

    if cfg == "LogisticLaplaceTS":
        selected_rec_ids, selected_feature_vector, sampled = intervention_mab.select_action(
            [feature_vec_to_rec_ids[fv] for fv in fvs], fvs
        )
        if not select_anyway and sampled["estimated_reward"] <= 0.5:
            return None, None

        sel, _ = handle_recommendation_mab(
            user,
            selected_rec_ids,
            recommendation_mab,
            data_storage,
            time_handler,
            selection_id,
        )
    else:
        raise ValueError(f"Unknown intervention_mab_mode: {cfg}")

    data_storage.add_intervention_mab_sample(
        record={
            "user_id": user.user_id,
            "plan_id": selection_id["plan_id"],
            "content_count": selection_id["content_count"],
            "feature_vector": selected_feature_vector,
            "selected_rec_ids": selected_rec_ids,
            "timestamp": time_handler.utc_iso(time_handler.now),
            "sample": sampled,
        },
    )
    return sel, selected_feature_vector


def handle_recommendation_mab(user, rec_ids, recommendation_mab, data_storage, time_handler, selection_id):
    rec_cfg = RECOMMENDATION_MAB_CONFIG["type"]

    if rec_cfg == "BernoulliBetaTS":
        sel, sampled = recommendation_mab.select_action(rec_ids)
    else:
        raise ValueError(f"Unknown recommendation_mab_mode: {rec_cfg}")

    data_storage.add_mab_sample(
        table="recommendation_mab_samples",
        record={
            "user_id": user.user_id,
            "plan_id": selection_id["plan_id"],
            "content_count": selection_id["content_count"],
            "timestamp": time_handler.utc_iso(time_handler.now),
            "sample": sampled,
        },
    )
    return sel, None


def select_resource(user, resource_mab, data_storage, time_handler, selection_id):
    selected_resources = []
    content_count = 0
    for mission in user.get_new_missions():
        rids = [r for r in mission.get("resources", []) if r not in user.get_received_resources()]

        if not rids:
            continue

        sel, sampled = resource_mab.select_action(rids)
        data_storage.add_mab_sample(
            table="resource_mab_samples",
            record={
                "user_id": user.user_id,
                "plan_id": selection_id["plan_id"],
                "content_count": content_count,
                "timestamp": time_handler.utc_iso(time_handler.now),
                "sample": sampled,
            },
        )
        selected_resources.append({"id": sel, "type": "resource", "mission_id": mission["mission"]})
        user.add_received_resource(sel)  # Assuming it will receive this resource
        content_count += 1

    return selected_resources
