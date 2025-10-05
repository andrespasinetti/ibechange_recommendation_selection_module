import json
import pprint
import numpy as np

def convert_ndarrays_to_lists(self, obj):
    """Recursively convert all ndarrays in a structure to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: self.convert_ndarrays_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [self.convert_ndarrays_to_lists(item) for item in obj]
    return obj


def _safe_serialize(obj):
    # If available in this scope, reuse your ndarray converter; otherwise no-op.
    try:
        return convert_ndarrays_to_lists(obj)
    except Exception:
        return obj


def pretty(obj, *, max_chars=None, indent=2, sort_keys=True):
    """Return a multi-line, human-readable string of obj for logs."""
    try:
        s = json.dumps(_safe_serialize(obj), indent=indent, sort_keys=sort_keys, ensure_ascii=False, default=str)
    except Exception:
        s = pprint.pformat(obj, width=120, compact=False, sort_dicts=True)
    if max_chars is not None and len(s) > max_chars:
        return s[:max_chars] + f"\n... <truncated {len(s) - max_chars} chars>"
    return s
