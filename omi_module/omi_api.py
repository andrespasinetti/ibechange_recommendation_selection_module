# omi_api.py
from flask import Flask, jsonify, request
import logging
import random
from datetime import datetime, timedelta, timezone
from omi_module.services.time_handler import TimeHandler

# ──────────────────────────────────────────────────────────────
# Flask & logging
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

# ──────────────────────────────────────────────────────────────
# Globals (simulated stores)
# ──────────────────────────────────────────────────────────────
time_handler = TimeHandler()
users: dict = {}
selected_contents: dict = {}
new_missions_and_contents: dict = {}


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def generate_plan_timestamps(num: int) -> list[str]:
    """
    Return `num` uniformly spaced timestamps over the next 7 days,
    serialised as canonical UTC strings.
    """
    start_dt = time_handler.now
    end_dt = start_dt + timedelta(days=6, hours=23, minutes=59, seconds=59)
    ts_float = sorted(random.uniform(start_dt.timestamp(), end_dt.timestamp()) for _ in range(num))
    return [time_handler.utc_iso(datetime.fromtimestamp(ts, tz=timezone.utc)) for ts in ts_float]


# ──────────────────────────────────────────────────────────────
# Health & time mode
# ──────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mode": time_handler.mode, "now": time_handler.now.isoformat()}), 200


@app.route("/set_time_mode", methods=["POST"])
def set_time_mode():
    body = request.get_json(silent=True) or {}
    mode = body.get("mode")
    if not mode:
        return jsonify({"error": "Missing 'mode'. Use REAL | FROZEN"}), 400
    try:
        time_handler.set_mode(mode)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"mode": time_handler.mode}), 200


# ──────────────────────────────────────────────────────────────
# Time control
# ──────────────────────────────────────────────────────────────
@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-04-13T12:00:00Z"}), 400

    time_handler.set_start_time(dt)
    return jsonify({"start_time": time_handler.utc_iso(dt)}), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)
    except ValueError:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-04-13T12:00:00Z"}), 400

    time_handler.set(dt)  # effective only in FROZEN
    return jsonify({"current_time": time_handler.utc_iso(time_handler.now)}), 200


# ──────────────────────────────────────────────────────────────
# Data ingestion
# ──────────────────────────────────────────────────────────────
@app.route("/recommendations", methods=["POST"])
def recommendations_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Recommendations initialised"}), 201


@app.route("/resources", methods=["POST"])
def resources_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Resources initialised"}), 201


@app.route("/missions", methods=["POST"])
def missions_endpoint():
    if not request.get_json(silent=True):
        return jsonify({"error": "Invalid JSON data"}), 400
    return jsonify({"message": "Missions initialised"}), 201


@app.route("/updates", methods=["POST"])
def updates_endpoint():
    logging.info("Received request at /updates")
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "Invalid JSON data"}), 400

    global users, new_missions_and_contents
    users.update(data.get("new_users", {}))
    new_missions_and_contents = data.get("new_missions_and_contents", {})
    logging.info("Daily updates successfully processed")
    return jsonify({"message": "Daily updates received"}), 201


# ──────────────────────────────────────────────────────────────
# Planning I/O
# ──────────────────────────────────────────────────────────────
@app.route("/selected_contents", methods=["POST"])
def selected_contents_endpoint():
    global selected_contents
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Invalid JSON data"}), 400
    selected_contents = payload
    return jsonify({"message": "Selected contents received"}), 201


@app.route("/recommendation_plans", methods=["GET"])
def process_recommendation_plans():
    global selected_contents
    logging.info("Generating recommendation plans …")

    recommendation_plans = {"recommendation_plans": []}

    if not selected_contents:
        return jsonify(recommendation_plans), 200

    for user_id, items in selected_contents.items():
        contents = items.get("contents", [])
        ts_list = generate_plan_timestamps(len(contents))
        user_plan = {
            "user_id": user_id,
            "plans": [
                {
                    "content_id": content["id"],
                    "type": content["type"],  # local testing convenience
                    "mission_id": content["mission_id"],
                    "scheduled_for": ts_list[i],
                }
                for i, content in enumerate(contents)
            ],
            "plan_id": items.get("plan_id"),
        }

        recommendation_plans["recommendation_plans"].append(user_plan)

    selected_contents = {}
    logging.info("✅ Recommendation plans ready")
    return jsonify(recommendation_plans), 200


# ──────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.info("🚀 Starting OMI API server …")
    app.run(host="0.0.0.0", port=5000, debug=False)
