# cs_module/utils/process_binder.py
from collections import defaultdict, deque
import os


class ProcessBinder:
    """
    Maps planned recommendations (snapshots) to concrete process_ids
    that EUT generates when it sends a notification.

    - pending[user_id][plan_id] is a deque of snapshots waiting to be sent.
    - proc_map[process_id] gives the bound snapshot once a rec has been sent.
    """

    def __init__(self, proc_cap: int | None = None):
        # Each user_id → plan_id → deque of snapshots (FIFO order).
        self.pending = defaultdict(lambda: defaultdict(deque))
        # After binding, we can recover the snapshot from the EUT process_id.
        self.proc_map = {}
        self._cap = proc_cap or int(os.getenv("BINDER_PROC_CAP", "200000"))

    def enqueue_decision(self, user_id, plan_id, snapshot):
        """
        Add a planned recommendation snapshot to the pending queue.

        snapshot must include at least:
          - rec_id: recommendation ID
          - mission_id: mission ID
          - feature_vector: FV used by intervention bandit (or None if unused)
          - content_count: position in the plan (helps disambiguate repeats)
          - selection_time: mission selection timestamp
        """
        self.pending[user_id][plan_id].append(snapshot)

    def bind_on_sent(self, user_id, plan_id, rec_id, mission_id, process_id):
        """
        When EUT emits a 'sent' event with a process_id, bind it to the
        first matching pending snapshot (matching rec_id and mission_id).

        Removes the snapshot from pending (so repeats are handled in order).
        Returns the bound snapshot, or None if not found.
        """
        dq = self.pending[user_id][plan_id]
        bound = None
        new_dq = deque()

        # scan in FIFO order
        while dq:
            snap = dq.popleft()
            if bound is None and snap["rec_id"] == rec_id and (mission_id is None or snap["mission_id"] == mission_id):
                bound = snap  # consume this one
            else:
                new_dq.append(snap)

        # update pending deque
        self.pending[user_id][plan_id] = new_dq

        # record binding
        if bound is not None:
            self.proc_map[process_id] = bound
        return bound

    def set_snapshot(self, process_id, *, rec_id, mission_id, feature_vector, selection_time=None, extra=None):
        """
        Used by replay: cache a snapshot at SEND time so ratings can look it up by process_id.
        """
        snap = {
            "rec_id": rec_id,
            "mission_id": mission_id,
            "feature_vector": feature_vector,
            "selection_time": selection_time,
        }
        if extra:
            snap.update(extra)
        self._remember(process_id, snap)

    def lookup(self, process_id):
        """
        Look up the snapshot bound to a process_id (from open/rated events).
        Returns None if not found.
        """
        return self.proc_map.get(process_id)

    def release(self, process_id):
        """
        Drop the mapping after final feedback (e.g. after 'rated'),
        to prevent unbounded memory growth.
        """
        self.proc_map.pop(process_id, None)

    def _remember(self, process_id, snapshot):
        self.proc_map[process_id] = snapshot
        if len(self.proc_map) > self._cap:
            # Evict the oldest inserted mapping (dicts preserve insertion order)
            oldest_key = next(iter(self.proc_map))
            if oldest_key != process_id:
                self.proc_map.pop(oldest_key, None)
