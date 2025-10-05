from flask import Flask, jsonify, request
from virtual_user.virtual_user import VirtualUser
from virtual_user.services.time_handler import TimeHandler
import logging
import numpy as np, random

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

time_handler = TimeHandler()
service = VirtualUser(time_handler)
app = Flask(__name__)


@app.route("/seed", methods=["POST"])
def seed_endpoint():
    body = request.get_json(silent=True) or {}
    s = int(body.get("seed", 0))
    random.seed(s)
    np.random.seed(s)
    return jsonify({"seed": s}), 200


# NEW: allow orchestrator to set REAL/FROZEN
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


@app.route("/set_start_time", methods=["POST"])
def set_start_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)  # aware UTC
    except Exception:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-09-02T08:00:00Z"}), 400

    time_handler.set_start_time(dt)
    return jsonify({"start_time": time_handler.utc_iso(dt)}), 200


@app.route("/set_current_time", methods=["POST"])
def set_current_time():
    iso_time_str = request.get_json(silent=True)
    if not isinstance(iso_time_str, str):
        return jsonify({"error": "Body must be an ISO 8601 string"}), 400
    try:
        dt = time_handler.parse_client_ts(iso_time_str)  # aware UTC
    except Exception:
        return jsonify({"error": "Invalid datetime format. Use ISO 8601 like 2025-09-02T08:00:00Z"}), 400

    time_handler.set(dt)  # no-op in REAL, effective in FROZEN
    return jsonify({"current_time": time_handler.utc_iso(time_handler.now)}), 200


@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    return jsonify(service.raw_recommendations), 200


@app.route("/resources", methods=["GET"])
def get_resources():
    return jsonify(service.raw_resources), 200

 
@app.route("/missions", methods=["GET"])
def get_missions():
    return jsonify(service.raw_missions), 200


@app.route("/updates", methods=["GET"])
def get_updates():
    service.simulate_hour()
    return jsonify(service.get_updates()), 200


@app.route("/recommendation_plans", methods=["POST"])
def save_weekly_recommendation_plans():
    data = request.get_json(silent=True) or {}
    response = service.save_weekly_recommendation_plans(data)
    return jsonify(response), 201


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
