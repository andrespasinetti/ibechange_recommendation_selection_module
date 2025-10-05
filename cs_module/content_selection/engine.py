from cs_module.config import MIN_NUM_REC_PER_MISSION, MAX_NUM_REC_PER_MISSION
from .user_manager import UserManager
from .feature_builders import get_mission_to_feature_vec_to_rec_ids
from .selector import select_recommendation, select_resource
from .frequency_updaters import update_frequency_offsets
from cs_module.utils.process_binder import ProcessBinder

import uuid
import logging

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(
        self,
        user_manager: UserManager,
        binder: ProcessBinder,
        missions,
        resources,
        recommendations,
        resource_mab,
        intervention_mab,
        recommendation_mab,
        data_storage,
        time_handler,
    ):
        self.user_manager = user_manager
        self.binder = binder
        self.missions = missions
        self.resources = resources
        self.recommendations = recommendations
        self.resource_mab = resource_mab
        self.intervention_mab = intervention_mab
        self.recommendation_mab = recommendation_mab
        self.data_storage = data_storage
        self.time_handler = time_handler
        self.selection_id = {}

    # helper to roll a new weekly plan **after** everything is saved
    def rotate_plan_id(self, user_id):
        self.selection_id[user_id] = {"plan_id": str(uuid.uuid4()), "content_count": 1}

    def get_current_plan_id(self, user_id):
        if user_id not in self.selection_id:
            self.selection_id[user_id] = {"plan_id": str(uuid.uuid4()), "content_count": 1}
        return self.selection_id[user_id]["plan_id"]

    def _to_float_array_or_none(self, fv):
        if fv is None:
            return None
        try:
            return [float(x) for x in fv]
        except Exception:
            return None

    def get_recommendations_to_send(self, user_id):
        user = self.user_manager.get_user(user_id)
        # ONE MISSION AT A TIME FOR PILOT STUDY
        user_missions = user.get_new_missions()

        mission_to_available_recs = {
            m["mission"]: user.get_available_recommendations(m["mission"]) for m in user_missions
        }

        mission_to_available_recs_unchanged = {
            m["mission"]: user.get_available_recommendations(m["mission"]) for m in user_missions
        }

        selected_recs = []
        total_freq_offset = 0
        intv_to_freq_offset = {}
        rec_to_freq_offset = {}
        mission_to_selected_rec_to_count = {}

        # Assuming equal distribution of recommendations across missions
        mission_to_num_slots = {m["mission"]: MAX_NUM_REC_PER_MISSION // len(user_missions) for m in user_missions}

        if user_id not in self.selection_id:
            self.selection_id[user_id] = {"plan_id": str(uuid.uuid4()), "content_count": 1}

        for mission_id, num_slots in mission_to_num_slots.items():
            count = 1
            for _ in range(num_slots):
                mission_to_feature_vec_to_rec_ids = get_mission_to_feature_vec_to_rec_ids(
                    user=self.user_manager.get_user(user_id),
                    missions=self.missions,
                    recommendations=self.recommendations,
                    mission_id_to_avail_rec_ids=mission_to_available_recs,
                    total_freq_offset=total_freq_offset,
                    intv_to_freq_offset=intv_to_freq_offset,
                    rec_to_freq_offset=rec_to_freq_offset,
                    prompted=False,
                )
                fv_map = mission_to_feature_vec_to_rec_ids.get(mission_id)
                if not fv_map:
                    logger.warning("No FV map for mission %s (user %s); skipping slot.", mission_id, user_id)
                    break
                sel_rec_id, sel_fv = select_recommendation(
                    feature_vec_to_rec_ids=mission_to_feature_vec_to_rec_ids[mission_id],
                    intervention_mab=self.intervention_mab,
                    recommendation_mab=self.recommendation_mab,
                    data_storage=self.data_storage,
                    time_handler=self.time_handler,
                    user=user,
                    select_anyway=count <= MIN_NUM_REC_PER_MISSION,
                    selection_id=self.selection_id[user_id],
                )

                if sel_rec_id is None:
                    continue

                # enqueue snapshot for later binding on 'sent'
                self.binder.enqueue_decision(
                    user_id,
                    self.selection_id[user_id]["plan_id"],
                    {
                        "content_count": self.selection_id[user_id]["content_count"],
                        "rec_id": sel_rec_id,
                        "mission_id": mission_id,
                        "feature_vector": sel_fv,  # may be None if no intervention bandit
                        "selection_time": self.time_handler.now,
                    },
                )

                selected_recs.append({"id": sel_rec_id, "type": "recommendation", "mission_id": mission_id})

                (
                    mission_to_selected_rec_to_count,
                    total_freq_offset,
                    intv_to_freq_offset,
                    rec_to_freq_offset,
                    mission_to_available_recs,
                ) = update_frequency_offsets(
                    sel_rec_id=sel_rec_id,
                    mission_id=mission_id,
                    mission_to_selected_rec_to_count=mission_to_selected_rec_to_count,
                    total_freq_offset=total_freq_offset,
                    intv_to_freq_offset=intv_to_freq_offset,
                    rec_to_freq_offset=rec_to_freq_offset,
                    mission_to_available_recs=mission_to_available_recs,
                    recommendations=self.recommendations,
                )

                count += 1
                self.selection_id[user_id]["content_count"] += 1

                # ANTICIPATE END OF WEEK RECOMMENDATIONS
                # IT SHOULD WORK EVEN WITH MULTIPLE
                mission_to_feature_vec_to_rec_ids = get_mission_to_feature_vec_to_rec_ids(
                    user=self.user_manager.get_user(user_id),
                    missions=self.missions,
                    recommendations=self.recommendations,
                    mission_id_to_avail_rec_ids=mission_to_available_recs_unchanged,
                    total_freq_offset=total_freq_offset,
                    intv_to_freq_offset=intv_to_freq_offset,
                    rec_to_freq_offset=rec_to_freq_offset,
                    prompted=True,  # Prompted to user
                )
                feature_vec_to_rec_ids = mission_to_feature_vec_to_rec_ids[mission_id]

                # Find the matching feature vector key, if any
                matches = [key for key, rec_ids in feature_vec_to_rec_ids.items() if sel_rec_id in rec_ids]
                if matches:
                    eow_fv = matches[0]
                    # Option A (robust): store a normalized list so later code is simpler
                    user.eow_rec_id_to_fv[sel_rec_id] = self._to_float_array_or_none(eow_fv)
                else:
                    logger.warning(f"No EoW feature vector found for rec_id={sel_rec_id} in mission {mission_id}")

                mission_to_available_recs = user.update_avail_recommendations(mission_to_available_recs, sel_rec_id)
                if not mission_to_available_recs[mission_id]:
                    break

        return selected_recs

    def get_resources_to_send(self, user_id):
        if user_id not in self.selection_id:  # first plan for this user
            self.selection_id[user_id] = {"plan_id": str(uuid.uuid4()), "content_count": 1}

        user = self.user_manager.get_user(user_id)
        self.selection_id[user_id]["content_count"] = 1
        selected_resources = select_resource(
            user, self.resource_mab, self.data_storage, self.time_handler, selection_id=self.selection_id[user_id]
        )
        self.selection_id[user_id]["content_count"] = len(selected_resources)
        return selected_resources
