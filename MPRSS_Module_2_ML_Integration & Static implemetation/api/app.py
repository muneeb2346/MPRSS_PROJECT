"""
MPRSS Static Analysis API - Module 2 / Module 7 integration
Serves the trained permission-risk Random Forest model to the React portal.

Run with:  python api/app.py
Endpoint:  POST http://localhost:5000/api/analyze/static
"""

import os
import sys
import json
import tempfile
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from apk_static_parser import APKStaticParser

app = Flask(__name__)
CORS(app)  # allow requests from the React dev server (localhost:3000)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB, matches UploadArea.jsx limit

MODEL_PATH    = os.path.join('models', 'permission_risk_rf.pkl')
METADATA_PATH = os.path.join('models', 'feature_columns.json')

apk_parser = None

model            = None
feature_columns  = []
feature_categories = {}


def load_model():
    """Load the trained model + feature metadata once at startup"""
    global model, feature_columns, feature_categories, apk_parser

    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(
            f"Model files not found. Run 'python src/risk_model_trainer.py' first.\n"
            f"Expected: {MODEL_PATH} and {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    feature_columns    = metadata['feature_columns']
    feature_categories = metadata['feature_categories']

    apk_parser = APKStaticParser(METADATA_PATH)

    print(f"✅ Model loaded: {len(feature_columns)} features")
    print(f"   Training metrics: {metadata.get('metrics', {})}")


def build_feature_vector(active_features):
    """
    Build a model-ready feature vector from a list of active feature names
    (permissions/API calls/intents present in the analyzed APK).

    active_features: list[str] — names matching columns in feature_columns.json.
    Unknown names sent by the client are ignored (not silently mismatched).
    """
    active_set = set(active_features)
    vector = [1 if f in active_set else 0 for f in feature_columns]
    return np.array(vector).reshape(1, -1)


def get_top_contributing_features(active_features, top_n=5):
    """
    For explainability: intersect the submitted active features with the
    model's global feature_importances_ (via predict_proba isn't per-sample
    SHAP, but ranking active features by their trained importance gives a
    reasonable lightweight explanation until Module 4 (XAI/SHAP) is built).
    """
    importances = model.feature_importances_
    importance_map = dict(zip(feature_columns, importances))

    active_ranked = sorted(
        [(f, importance_map.get(f, 0), feature_categories.get(f, 'Unknown'))
         for f in active_features if f in importance_map],
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {'feature': f, 'importance': round(w, 4), 'category': c}
        for f, w, c in active_ranked[:top_n]
    ]


def score_active_features(active_features):
    """Shared scoring logic used by both the JSON and file-upload endpoints"""
    X = build_feature_vector(active_features)

    proba = model.predict_proba(X)[0]          # [P(benign), P(malware)]
    prediction = model.predict(X)[0]

    malware_confidence = float(proba[1])
    static_score = round(malware_confidence * 100)

    return {
        'static_score':  static_score,
        'prediction':    'malicious' if prediction == 1 else 'benign',
        'confidence':    round(float(max(proba)), 4),
        'top_features':  get_top_contributing_features(active_features, top_n=5)
    }


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'n_features': len(feature_columns)
    })


@app.route('/api/analyze/static', methods=['POST'])
def analyze_static():
    """
    Request body (JSON):
        { "active_features": ["SEND_SMS", "READ_PHONE_STATE", "transact", ...] }

    Response (JSON):
        {
            "static_score": 78,          # 0-100, risk score for the RiskBreakdown panel
            "prediction": "malicious",   # or "benign"
            "confidence": 0.94,
            "top_features": [ {feature, importance, category}, ... ]
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json(silent=True)
    if not data or 'active_features' not in data:
        return jsonify({'error': "Request must include 'active_features': [list of strings]"}), 400

    active_features = data['active_features']
    if not isinstance(active_features, list):
        return jsonify({'error': "'active_features' must be a list of strings"}), 400

    return jsonify(score_active_features(active_features))


@app.route('/api/analyze/apk', methods=['POST'])
def analyze_apk():
    """
    Real APK upload endpoint — this is what UploadArea.jsx should call.

    Request: multipart/form-data with a file field named 'apk'
    Response (JSON): same shape as /api/analyze/static, plus 'app_info'
    """
    if model is None or apk_parser is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'apk' not in request.files:
        return jsonify({'error': "Request must include a file field named 'apk'"}), 400

    file = request.files['apk']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not file.filename.lower().endswith('.apk'):
        return jsonify({'error': 'File must be a .apk'}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        parsed = apk_parser.parse(tmp_path, verbose=False)
        response = score_active_features(parsed['active_features'])
        response['app_info'] = parsed['app_info']
        # Needed by Module 7's family_predictor.predict_static_family() -
        # previously computed here internally (via score_active_features)
        # but never included in the response, so Module 5's static family
        # classification always silently fell back to "no active features".
        response['active_features'] = parsed['active_features']

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': f'Failed to analyze APK: {str(e)}'}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000, threaded=True)