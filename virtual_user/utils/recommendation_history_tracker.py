from bisect import insort


class RecommendationHistoryTracker:
    """Tracks frequency of recommendations, keeping entries ordered by time."""

    def __init__(self):
        self.history = []  # List of (timestamp, rec_id), always sorted by timestamp

    def add_recommendation(self, timestamp, notification_id, rec_id, intervention_type):
        """Add a recommendation (auto-sorted by time)."""
        insort(self.history, (timestamp, notification_id, rec_id, intervention_type))

    def get_count(self, time_window=None, rec_id=None, single_intv=None):
        """Get count of recommendations, optionally filtered by rec_id, intervention, and time window."""

        return sum(
            1
            for ts, nid, rid, itype in self.history
            if (time_window is None or (time_window[0] <= ts < time_window[1]))
            and (rec_id is None or rid == rec_id)
            and (single_intv is None or (single_intv in itype))
        )
