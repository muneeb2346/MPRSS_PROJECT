"""
MPRSS Module 1 - Behavioral Analysis Service  (port 5003)

Run from the MPRSS_Module1_Behavior folder:   python behavior_service.py
Needs models\\behavior_lstm.pt and models\\behavior_meta.json (created by train_final.py).

POST /api/analyze/behavior
    body:  { "events": [ { "ts": 1700000000123, "api": "java.lang.Runtime.exec", ... }, ... ] }
    reply: { "behavioral_score": 0-100 or null, "n_events_received": ..., "n_tokens_used": ...,
             "tokens_used_preview": [...], "note": "..." }
GET  /api/health
"""

from flask import Flask, request, jsonify

try:
    from flask_cors import CORS
except ImportError:                      # optional: the orchestrator calls this server-to-server
    CORS = None


def create_app(scorer):
    app = Flask(__name__)
    if CORS:
        CORS(app)

    @app.route("/api/health", methods=["GET"])
    def health():
        meta = scorer.meta
        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "vocab_size": len(meta["vocab"]),
            "min_tokens_needed": meta["min_len"],
            "trained_at": meta.get("trained_at"),
            "evaluation": meta.get("evaluation"),
        })

    @app.route("/api/analyze/behavior", methods=["POST"])
    def analyze_behavior():
        data = request.get_json(silent=True)
        if not data or not isinstance(data.get("events"), list):
            return jsonify({"error": "Request must be JSON with an 'events' list"}), 400
        return jsonify(scorer.score(data["events"]))

    return app


if __name__ == "__main__":
    from behavior_core import BehaviorScorer

    scorer = BehaviorScorer("models")
    print(f"Module 1 model loaded: {len(scorer.vocab)} monitored APIs, needs >= {scorer.min_len} calls to score")
    create_app(scorer).run(port=5003, debug=False, threaded=True)