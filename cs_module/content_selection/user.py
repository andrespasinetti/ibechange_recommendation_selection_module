from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
from datetime import datetime
from cs_module.utils.recommendation_history_tracker import RecommendationHistoryTracker
from cs_module.services.time_handler import TimeHandler
from cs_module.config import MAX_NUM_REC_PER_MISSION, PILLARS
from cs_module.utils.encoding import get_intervention_encoding
import logging


def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


@dataclass
class User:
    user_id: str
    time_handler: TimeHandler
    intervention_start_date: Optional[datetime] = field(default=None)
    personal_data: Dict[str, Any] = field(default_factory=dict)
    health_habit_assessment: Dict[str, Any] = field(default_factory=dict)

    active: bool = True
    missions_started: bool = False

    new_plan_required: bool = field(default=False)

    selected_missions_and_contents: List[Dict] = field(default_factory=list)
    selected_contents: Dict[str, Any] = field(default_factory=dict)
    current_recommendation_plan: Dict[str, Any] = field(default_factory=dict)

    intervention_open_history: Dict[str, List[Any]] = field(default_factory=dict)
    recommendation_open_history: Dict[str, List[Any]] = field(default_factory=dict)
    global_open_history: Dict[str, List[Any]] = field(default_factory=dict)

    sent_rec_tracker: RecommendationHistoryTracker = field(default_factory=RecommendationHistoryTracker)

    disliked_rec_to_remaining_days: Dict[str, int] = field(default_factory=dict)
    past_week_rec_to_frequency: Dict[str, int] = field(default_factory=dict)

    received_resources: List[str] = field(default_factory=list)

    only_one_rec_already: List[str] = field(default_factory=list)

    eow_rec_id_to_fv: Dict[str, Any] = field(default_factory=dict)
    rating_history: List[Tuple[datetime, str, bool]] = field(default_factory=list)

    previous_mission_score = 0

    def add_received_resource(self, resource_id: str):
        """Add a resource to the list of received resources."""
        if resource_id not in self.received_resources:
            self.received_resources.append(resource_id)

    def get_received_resources(self) -> List[str]:
        """Get the list of received resources."""
        return self.received_resources

    def disable(self):
        self.active = False

    def update_escalation_level(self, level: int):
        self.personal_data["level"] = level

    def get_personal_data(self):
        return self.personal_data

    def update_health_habit_assessment(self, hhs: Dict[str, Any]):
        if not isinstance(hhs, dict):
            logging.warning("HHS payload not a dict: %r", hhs)
            return

        # 1) Main pillar scores
        for key, value in hhs.items():
            if key in ("components", "emotional_distress"):
                continue
            if key in PILLARS:
                fv = _to_float_or_none(value)
                if fv is None:
                    logging.warning("Pillar %s has non-numeric value %r; skipping.", key, value)
                else:
                    self.health_habit_assessment[key] = fv
            else:
                logging.warning("Unknown HHS key %r (value=%r); ignoring.", key, value)

        # 2) Components (nutrition or emotional_wellbeing)
        if "components" in hhs:
            target_pillar = None
            if "nutrition" in hhs:
                target_pillar = "nutrition"
            elif "emotional_wellbeing" in hhs:
                target_pillar = "emotional_wellbeing"
            else:
                logging.warning("Received 'components' without nutrition/emotional_wellbeing key; skipping components.")
            if target_pillar:
                comp_key = f"{target_pillar}_components"
                if comp_key not in self.health_habit_assessment or not isinstance(
                    self.health_habit_assessment.get(comp_key), dict
                ):
                    self.health_habit_assessment[comp_key] = {}
                for cname, cval in (hhs["components"] or {}).items():
                    fv = _to_float_or_none(cval)
                    if fv is None:
                        logging.warning(
                            "Component %s.%s has non-numeric value %r; skipping.", target_pillar, cname, cval
                        )
                        continue
                    # overwrite or add
                    self.health_habit_assessment[comp_key][cname] = fv

        # 3) Bi-weekly emotional_distress → update emotional_wellbeing_components
        if "emotional_distress" in hhs:
            fv = _to_float_or_none(hhs["emotional_distress"])
            if fv is None:
                logging.warning("emotional_distress has non-numeric value %r; skipping.", hhs["emotional_distress"])
            else:
                ew_key = "emotional_wellbeing_components"
                if ew_key not in self.health_habit_assessment or not isinstance(
                    self.health_habit_assessment.get(ew_key), dict
                ):
                    self.health_habit_assessment[ew_key] = {}
                self.health_habit_assessment[ew_key]["emotional_distress"] = fv

    def get_hhs(self):
        return self.health_habit_assessment

    def get_num_intervention_days(self):
        """Get the number of days since the user started using the app."""
        if self.intervention_start_date:
            return int((self.time_handler.now - self.intervention_start_date).days)
        return None

    def set_missions_plan_to_false(self, mission_ids: list):
        """Set the mission plan to false for a given mission ID."""
        for mission in self.selected_missions_and_contents:
            if mission["mission"] in mission_ids:  # same mission might have been selected multiple times... no problem
                mission["plan_required"] = False

    def update_missions_and_contents(self, missions_and_contents: Dict[str, Any]):
        for mission in missions_and_contents.get("new_missions", []):
            mission["plan_required"] = True
            self.selected_missions_and_contents.append(mission)

            if not self.missions_started:
                self.missions_started = True
                self.intervention_start_date = self.time_handler.parse_client_ts(mission["selection_timestamp"])

        self.new_plan_required = True

    def get_new_missions(self):
        new_missions = []
        for mission in self.selected_missions_and_contents:
            if mission["plan_required"]:
                new_missions.append(mission)
        return new_missions

    def is_winter(self) -> bool:
        """Return True if it's winter (Dec–Feb) in the Northern Hemisphere."""
        now = getattr(self.time_handler, "now", None) or datetime.now()
        return now.month in (12, 1, 2)

    def is_spring(self) -> bool:
        """Return True if it's spring (Mar–May) in the Northern Hemisphere."""
        now = getattr(self.time_handler, "now", None) or datetime.now()
        return now.month in (3, 4, 5)

    def get_available_recommendations(self, mission_id: str) -> List[str]:
        """Get available contents for a given mission.
        ERc65, ERc66: Winter season
        ERc110: Spring season
        """

        for mission in self.selected_missions_and_contents:
            if mission["mission"] == mission_id:
                recommendations = list(mission["recommendations"])

                if "ERc65" in recommendations and not self.is_winter():
                    recommendations.remove("ERc65")

                if "ERc66" in recommendations and not self.is_winter():
                    recommendations.remove("ERc66")

                if "ERc110" in recommendations and not self.is_spring():
                    recommendations.remove("ERc110")

                return recommendations
        return []

    def save_recommendation_plan(self, recommendation_plan: Dict[str, Any]):
        """Save the computed weekly recommendation plan to the dataclass instance."""
        self.current_recommendation_plan = recommendation_plan
        self.new_plan_required = False

    def update_rec_plan_to_position(self):
        """Update the recommendation plan to position mapping."""
        rec_plan_to_position_temp = {}
        self.rec_plan_to_position = {}

        contents = self.current_recommendation_plan.get("plans", [])  # Extract contents

        for content in contents:
            if content["type"] == "recommendation":  # Check if the content is a recommendation
                rec_plan_to_position_temp[content["id"]] = rec_plan_to_position_temp.get(content["id"], 0) + 1
                self.rec_plan_to_position[(content["send_timestamp"], content["id"])] = rec_plan_to_position_temp[
                    content["id"]
                ]

    def update_rec_plan_to_frequency(self):
        """Update the recommendation plan to frequency mapping."""
        self.rec_plan_to_frequency = {}

        contents = self.current_recommendation_plan.get("plans", [])  # Extract contents

        for content in contents:
            if content["type"] == "recommendation":  # Check if the content is a recommendation
                self.rec_plan_to_frequency[content["id"]] = self.rec_plan_to_frequency.get(content["id"], 0) + 1

    def get_sample_feedback_position(self, sample_feedback: Tuple[str, str]) -> int:
        """Get the position of the sample feedback in the recommendation plan."""
        return self.rec_plan_to_position[sample_feedback]

    def get_sample_feedback_frequency(self, sample_feedback: str) -> int:
        """Get the frequency of the sample feedback in the recommendation plan."""
        return self.rec_plan_to_frequency[sample_feedback]

    def update_recommendation_open_history(self, availability_feedback: Any):
        """Update the recommendation history with the given feedback."""
        for feedback in availability_feedback:
            if feedback["type"] == "recommendation":
                ts = self.time_handler.parse_client_ts(feedback["sent_timestamp"])
                self.recommendation_open_history.setdefault(feedback["content_id"], []).append(ts)

    def get_recommendation_sliding_frequency(self, content_id, timestamp, sliding_window=7) -> int:
        """Get the sliding frequency of the recommendation."""
        rec_history = self.recommendation_open_history.get(content_id, [])

        filtered_history = [
            rec_timestamp for rec_timestamp in rec_history if (timestamp - rec_timestamp).days <= sliding_window
        ]

        # Here we just return the length of the filtered history
        sliding_frequency = len(filtered_history)
        return sliding_frequency

    def track_sent_recommendations(self, timestamp, process_id, rec_id, intervention_type, mission_id):
        self.sent_rec_tracker.add_recommendation(timestamp, process_id, rec_id, intervention_type, mission_id)

    def get_total_frequency(self, time_window=None):
        """Get the global frequency of the user."""
        total_frequency = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=None, single_intv=None)
        return total_frequency

    def get_recommendation_frequency(self, rec_id, time_window=None):
        """Get the frequency of the recommendation."""
        frequency = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=rec_id, single_intv=None)
        return frequency

    def get_intervention_frequency(self, intervention_type, time_window=None):
        """
        Mixture-weighted past-week intervention frequency for THIS item:
        dot(item_mix, past_type_counters)/MAX_NUM_REC_PER_MISSION  ∈ [0,1]
        """
        if not intervention_type:
            return 0.0
        item_mix = get_intervention_encoding(intervention_type)  # len 8
        type_counts = self.sent_rec_tracker.get_type_counters(time_window=time_window)  # len 8 floats
        burden = sum(w * c for w, c in zip(item_mix, type_counts))
        return burden / float(MAX_NUM_REC_PER_MISSION)

    """
    SRc52: This recommendation must appear only once during the intervention but is mandatory.
    SRc100, SRc101: Only one of these two recommendations can be presented for the same mission.
    """

    def update_avail_recommendations(self, mission_to_recommendations, sel_rec_id):
        for mission, recs in mission_to_recommendations.items():
            # SRc52 appears only once in the intervention
            if sel_rec_id == "SRc52" and sel_rec_id in recs and "SRc52" not in self.only_one_rec_already:
                self.only_one_rec_already.append("SRc52")
                recs.remove("SRc52")

            # SRc100 and SRc101 are mutually exclusive
            if sel_rec_id in {"SRc100", "SRc101"} and sel_rec_id in recs:
                other = "SRc101" if sel_rec_id == "SRc100" else "SRc100"
                if other in recs:
                    recs.remove(other)

            mission_to_recommendations[mission] = recs

        return mission_to_recommendations

    # called from your updater when a rating arrives
    def track_rating(self, timestamp: datetime, rec_id: str, is_end_misison: bool):
        self.rating_history.append((timestamp, rec_id, bool(is_end_misison)))

    def get_engagement_rate(self, time_window=None) -> float:
        """
        ER^{past} = (# voluntary ratings last week) / (# items sent last week).
        - voluntary = not prompted (is_end_misision == False)
        - denominator = items SENT (from sent_rec_tracker) in the same window
        Returns 0.0 if denominator is 0.
        """
        # denominator: # sent
        sent = self.sent_rec_tracker.get_count(time_window=time_window, rec_id=None, single_intv=None)

        if sent == 0:  # FIX
            return 0.0

        # numerator: # voluntary ratings (exclude end-of-week prompted)
        vol = sum(
            1
            for ts, rid, is_prompted in self.rating_history
            if (time_window is None or (time_window[0] <= ts < time_window[1])) and not is_prompted
        )
        return max(0.0, min(1.0, vol / sent))

    def set_previous_mission_score(self, score):
        self.previous_mission_score = score

    def get_previous_mission_score(self):
        return self.previous_mission_score

    def mission_snapshot_at(self, mission_id: str, ts: datetime) -> Optional[dict]:
        """
        Return the most recent selection record for mission_id with selection_timestamp <= ts.
        Includes at least {'prescribed': bool, 'record': dict}. Returns None if not found.
        """
        best = None
        best_sel = None
        for rec in self.selected_missions_and_contents:
            if rec.get("mission") != mission_id:
                continue
            try:
                sel_ts = self.time_handler.parse_client_ts(rec.get("selection_timestamp"))
            except Exception:
                continue
            if sel_ts <= ts and (best_sel is None or sel_ts > best_sel):
                best, best_sel = rec, sel_ts

        if best is None:
            return None

        return {
            "prescribed": bool(best.get("prescribed", False)),
            "record": best,  # full mission record if needed downstream
            "selection_timestamp": best_sel,
            "finish_timestamp": best.get("finish_timestamp"),
        }
