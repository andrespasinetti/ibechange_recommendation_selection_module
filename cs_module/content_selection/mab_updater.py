import logging
from cs_module.config import REWARD_TYPE, RECOMMENDATION_MAB_CONFIG, INTERVENTION_MAB_CONFIG, RESOURCE_MAB_CONFIG
from cs_module.utils.feedback_handler import (
    get_sent_recommendations,
    get_rated_recommendations,
    get_rated_resources,
)
from cs_module.utils.process_binder import ProcessBinder
from cs_module.utils.logging_utils import pretty
from cs_module.content_selection.feature_builders import get_mission_to_feature_vec_to_rec_ids
from cs_module.content_selection.frequency_updaters import update_frequency_offsets
from datetime import timedelta


class MABUpdater:
    def __init__(
        self,
        binder: ProcessBinder,
        time_handler,
        resource_mab,
        intervention_mab,
        recommendation_mab,
        user_manager,
        recommendations,
        missions,
        resources,
        data_storage,
    ):
        self.binder = binder
        self.time_handler = time_handler
        self.resource_mab = resource_mab
        self.intervention_mab = intervention_mab
        self.recommendation_mab = recommendation_mab
        self.user_manager = user_manager
        self.recommendations = recommendations
        self.missions = missions
        self.resources = resources
        self.data_storage = data_storage

    def update_all(self, feedback):
        for user_id, user_feedback in feedback.items():
            if user_id not in self.user_manager.get_all_users():
                logging.warning(f"User {user_id} not found. Skipping.")
                continue

            for event in user_feedback["events"]:
                # If mission was prescribed --> do not update MABs
                # No need to check for user level
                mission_id = event["properties"].get("mission_id")
                evt_ts = self.time_handler.parse_client_ts(event["timestamp"])

                user = self.user_manager.get_user(user_id)
                snap = user.mission_snapshot_at(mission_id, evt_ts)
                if snap is None:
                    logging.warning(
                        f"No mission snapshot at {evt_ts.isoformat()} for user={user_id}, mission={mission_id}; skipping feedback."
                    )
                    continue

                if snap["prescribed"]:
                    # Do not learn from prescribed missions
                    continue

                self._process_sent_recommendation(user_id, event)
                self._process_rated_recommendation(user_id, event)
                self._process_rated_resource(user_id, event)

    def _sent_seq_for_mission(self, user, mission_id, sel_ts):
        """Collect the actually-sent items for this mission in the 7-day plan window."""
        end_ts = sel_ts + timedelta(days=7)  # cap exactly like live planning
        seq = []
        for ts, pid, rid, _mix, mid in user.sent_rec_tracker.history:
            if mid != mission_id:
                continue
            if not (sel_ts <= ts < end_ts):
                continue
            seq.append({"sent_ts": ts, "process_id": pid, "rec_id": rid})
        seq.sort(key=lambda x: x["sent_ts"])
        for i, ev in enumerate(seq, start=1):
            ev["slot_index"] = i
        return seq

    def _fv_for_slot_at_selection_time(self, user, mission_id, rec_id, seq, slot_index, prompted: bool):
        avail = {mission_id: list(user.get_available_recommendations(mission_id))}
        total_off, intv_off, rec_off = 0, {}, {}
        for i in range(1, slot_index):
            prior = seq[i - 1]["rec_id"]
            _, total_off, intv_off, rec_off, avail = update_frequency_offsets(
                sel_rec_id=prior,
                mission_id=mission_id,
                mission_to_selected_rec_to_count={},
                total_freq_offset=total_off,
                intv_to_freq_offset=intv_off,
                rec_to_freq_offset=rec_off,
                mission_to_available_recs=avail,
                recommendations=self.recommendations,
            )
            # mirror live post-selection pruning
            avail = user.update_avail_recommendations(avail, prior)

        # 🔴 key: for prompted/EoW we also count THIS item (like live does)
        if prompted:
            _, total_off, intv_off, rec_off, avail = update_frequency_offsets(
                sel_rec_id=rec_id,
                mission_id=mission_id,
                mission_to_selected_rec_to_count={},
                total_freq_offset=total_off,
                intv_to_freq_offset=intv_off,
                rec_to_freq_offset=rec_off,
                mission_to_available_recs=avail,
                recommendations=self.recommendations,
            )
            avail = user.update_avail_recommendations(avail, rec_id)

        fv_map = get_mission_to_feature_vec_to_rec_ids(
            user=user,
            missions=self.missions,
            recommendations=self.recommendations,
            mission_id_to_avail_rec_ids=avail,
            total_freq_offset=total_off,
            intv_to_freq_offset=intv_off,
            rec_to_freq_offset=rec_off,
            prompted=prompted,
        ).get(mission_id, {})
        for key_fv, rec_ids in fv_map.items():
            if rec_id in rec_ids:
                try:
                    return [float(x) for x in key_fv]
                except Exception:
                    return None
        return None

    def _process_sent_recommendation(self, user_id, event):
        if not get_sent_recommendations([event]):
            return

        rec_id = event["properties"]["content_id"]
        if rec_id not in self.recommendations:
            logging.warning(f"Unknown recommendation ID {rec_id}")
            return
        mission_id = event["properties"].get("mission_id")
        process_id = event["process_id"]

        user = self.user_manager.get_user(user_id)
        ts = self.time_handler.parse_client_ts(event["timestamp"])
        intervention = self.recommendations[rec_id]["intervention_type"]
        user.track_sent_recommendations(ts, event["process_id"], rec_id, intervention, mission_id)

        # Try normal binder bind first
        plan_id = user.current_recommendation_plan.get("plan_id") or self._safe_plan_id_fallback(user_id)
        snap = self.binder.bind_on_sent(user_id, plan_id, rec_id, mission_id, process_id)

        if snap is not None:
            return  # all good: binder already has FV from selection-time

        # Fallback (works in LIVE and REPLAY): synthesize selection-time FV once and cache it
        snap_mission = user.mission_snapshot_at(mission_id, ts)
        if snap_mission is None or snap_mission["prescribed"]:
            # either we can’t locate selection or it was prescribed → don’t learn intervention later
            self.binder.set_snapshot(process_id, rec_id=rec_id, mission_id=mission_id, feature_vector=None)
            return

        sel_ts = snap_mission["selection_timestamp"]
        seq = self._sent_seq_for_mission(user, mission_id, sel_ts)

        # slot index of THIS send (prefer exact process_id)
        slot_index = next((ev["slot_index"] for ev in seq if ev["process_id"] == process_id), None)
        if slot_index is None:
            # fallback: first event at same timestamp & rec_id
            slot_index = next((ev["slot_index"] for ev in seq if ev["rec_id"] == rec_id and ev["sent_ts"] == ts), None)

        if slot_index is None:
            # still bind a minimal snapshot so rating lookups won’t crash
            self.binder.set_snapshot(process_id, rec_id=rec_id, mission_id=mission_id, feature_vector=None)
            return

        fv = self._fv_for_slot_at_selection_time(
            user=user,
            mission_id=mission_id,
            rec_id=rec_id,
            seq=seq,
            slot_index=slot_index,
            prompted=False,
        )
        self.binder.set_snapshot(process_id, rec_id=rec_id, mission_id=mission_id, feature_vector=fv)

        # Also precompute the EoW/prompted FV now, so if a prompted rating arrives later we have it.
        fv_prompted = self._fv_for_slot_at_selection_time(
            user=user,
            mission_id=mission_id,
            rec_id=rec_id,
            seq=seq,
            slot_index=slot_index,
            prompted=True,  # <— key: prompted=True for end-of-week context
        )
        if fv_prompted is not None:
            user.eow_rec_id_to_fv[rec_id] = fv_prompted

    def _process_rated_recommendation(self, user_id, event):
        if not get_rated_recommendations([event]):
            return
        logging.info("Updating %s MABs with event:\n%s", user_id, pretty(event))
        user = self.user_manager.get_user(user_id)
        rating = event["properties"]["rating"]
        reward = self._compute_reward(rating)
        process_id = event["process_id"]

        # From the event (fallbacks)
        event_rec_id = event["properties"].get("content_id")
        event_mission_id = event["properties"].get("mission_id")

        if event.get("properties", {}).get("is_end_misison", True):
            logging.info(f"End of mission feedback ({event_rec_id})")
            eow_rec_id_to_fv = self.user_manager.get_user(user_id).eow_rec_id_to_fv
            rec_id = event_rec_id
            mission_id = event_mission_id

            # Sanity checks
            if rec_id not in self.recommendations or mission_id not in self.missions:
                logging.warning(f"Unknown recommendation or mission ID: {rec_id}, {mission_id}")
                return
            if rec_id not in eow_rec_id_to_fv:
                logging.warning(
                    f"Recommendation {rec_id} not stored as end of mission in user object--> "
                    "unable to retrieve feature vector. "
                    "It can happen if the current model is not in intervention mode, "
                    "if it was a previous run that sent that recommendation, "
                    "or if EUT itself did."
                )
                return
            feature_vector = eow_rec_id_to_fv[rec_id]

        else:
            # --- 1) Try to recover the original decision via binder ---
            snap = None
            try:
                snap = self.binder.lookup(process_id=process_id)
                if snap is None:
                    logging.warning(
                        f"No binder snapshot found for process_id={process_id}, "
                        f"user_id={user_id}, event_rec_id={event.get('properties', {}).get('content_id')}."
                    )
                    return
            except Exception as e:
                logging.warning(f"Binder lookup failed for process_id={process_id}: {e}")
                return

            rec_id = snap.get("rec_id") if snap else event_rec_id
            feature_vector = snap.get("feature_vector") if (snap and "feature_vector" in snap) else None
            mission_id = snap.get("mission_id") if (snap and "mission_id" in snap) else event_mission_id

            # Sanity checks
            if rec_id not in self.recommendations or mission_id not in self.missions:
                logging.warning(f"Unknown recommendation or mission ID: {rec_id}, {mission_id}")
                return
            if snap and event_rec_id and rec_id != event_rec_id:
                logging.info(
                    f"Binder rec_id ({rec_id}) != event content_id ({event_rec_id}); trusting binder snapshot."
                )

        ts = self.time_handler.parse_client_ts(event["timestamp"])
        is_eow = bool(event["properties"].get("is_end_misison", False))
        user.track_rating(ts, rec_id, is_eow)

        # If intervention bandit disabled, just update recommendation bandit
        t = INTERVENTION_MAB_CONFIG["type"]
        if t == "None":
            self._update_recommendation_mab(rec_id, user, reward, process_id)
            return

        if t == "LogisticLaplaceTS":
            params = self.intervention_mab.update(feature_vector, reward)
            logging.info(f"Updated intervention MAB for recommendation {rec_id}.")
        else:
            raise ValueError(f"Unknown intervention_mab_mode: {t}")

        timestamp = self.time_handler.utc_iso(self.time_handler.now)
        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "feature_vector": feature_vector,
            "reward": reward,
            "params": params,
        }

        self.data_storage.add_intervention_mab_update(update=update)
        self.binder.release(process_id)
        self._update_recommendation_mab(rec_id, user, reward, process_id)

    def _update_recommendation_mab(self, rec_id, user, reward, process_id):
        t = RECOMMENDATION_MAB_CONFIG["type"]

        if t == "BernoulliBetaTS":
            params = self.recommendation_mab.update(rec_id, reward)
            logging.info(f"Updated recommendation MAB for {rec_id}.")

        elif t in ["RandomBandit", "RecommendationOptimalBandit"]:
            params = self.recommendation_mab.update(rec_id)

        else:
            raise ValueError(f"Unknown recommendation_mab_mode: {t}")

        timestamp = self.time_handler.utc_iso(self.time_handler.now)
        update = {
            "user_id": user.user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }
        self.data_storage.add_mab_update(table="recommendation_mab_updates", update=update)

    def _process_rated_resource(self, user_id, event):
        if not get_rated_resources([event]):
            return
        logging.info(f"Updating {user_id} MABs with event: {event}")

        res_id = event["properties"]["content_id"]

        if res_id not in self.resources:
            logging.warning(f"Unknown resource ID {res_id}")
            return

        if event["properties"].get("is_end_misison"):
            logging.info(f"End of mission feedback ({res_id})")

        reward = self._compute_reward(event["properties"]["rating"])
        t = RESOURCE_MAB_CONFIG["type"]

        if t == "BernoulliBetaTS":
            params = self.resource_mab.update(res_id, reward)

        elif t in ["RandomBandit", "ResourceOptimalBandit"]:
            params = self.resource_mab.update(reward)

        else:
            raise ValueError(f"Unknown resource_mab_mode: {t}")

        timestamp = self.time_handler.utc_iso(self.time_handler.now)
        process_id = event["process_id"]
        update = {
            "user_id": user_id,
            "timestamp": timestamp,
            "process_id": process_id,
            "reward": reward,
            "params": params,
        }

        self.data_storage.add_mab_update(table="resource_mab_updates", update=update)

        logging.info(f"Updated resource MAB for {res_id}.")

    def _compute_reward(self, rating):
        if REWARD_TYPE == "thumbs":
            return 1 if rating == "liked" else 0
        elif REWARD_TYPE == "float":
            return float(rating)
        raise ValueError(f"Unknown REWARD_TYPE: {REWARD_TYPE}")

    def _safe_plan_id_fallback(self, user_id):
        user = self.user_manager.get_user(user_id)
        # prefer current plan if present
        pid = user.current_recommendation_plan.get("plan_id") if user.current_recommendation_plan else None
        if pid:
            return pid
        # fallback to the last selected_contents plan_id (stored when CS selected)
        return (user.selected_contents or {}).get("plan_id")
