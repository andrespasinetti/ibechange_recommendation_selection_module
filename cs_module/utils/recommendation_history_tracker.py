from bisect import insort
from cs_module.utils.encoding import get_intervention_encoding
from cs_module.config import INTERVENTION_TYPES


class RecommendationHistoryTracker:
    """Tracks frequency of recommendations, keeping entries ordered by time."""

    def __init__(self):
        # store (timestamp, process_id, rec_id, mix_vector, mission_id)
        self.history = []

    def add_recommendation(self, timestamp, notification_id, rec_id, intervention_type, mission_id):
        """Add a recommendation (auto-sorted by time)."""
        mix = get_intervention_encoding(intervention_type)
        insort(self.history, (timestamp, notification_id, rec_id, mix, mission_id))

    def get_count(self, time_window=None, rec_id=None, single_intv=None):
        """Get count of recommendations, optionally filtered by rec_id, intervention, and time window."""

        return sum(
            1
            for ts, nid, rid, mix, mid in self.history
            if (time_window is None or (time_window[0] <= ts < time_window[1]))
            and (rec_id is None or rid == rec_id)
            and (single_intv is None or (single_intv in mix))
        )

    def get_type_counters(self, time_window=None):
        """Return per-type mixture-weighted counters over an optional time window."""
        counters = [0.0] * len(INTERVENTION_TYPES)
        for ts, nid, rid, mix, mid in self.history:
            if time_window is None or (time_window[0] <= ts < time_window[1]):
                for j, w in enumerate(mix):
                    counters[j] += w
        return counters
