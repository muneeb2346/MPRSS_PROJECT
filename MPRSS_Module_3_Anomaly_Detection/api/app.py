"""
MPRSS Module 3 - Anomaly Detection Flask API
Serves the trained One-Class SVM (benign-only) anomaly detector.

Accepts either:
  - a raw .pcap file (multipart 'pcap') -> extracts flow features internally
    via cicflowmeter + Scapy (no external tcpdump dependency needed - see
    the patch/workaround below, developed to work around a Windows-only
    Scapy/tcpdump subprocess issue)
  - OR a JSON body of pre-extracted flow features (for testing the
    model/scaler in isolation, using the SCHEMA column names directly)

Run with: python module3_app.py   (default port 5001)

Requires the trained model artifacts to be present relative to this file:
    models/anomaly_ocsvm.pkl
    models/anomaly_scaler.pkl
    models/anomaly_feature_columns.json
"""

import os
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from scapy.sendrecv import AsyncSniffer
from scapy.layers.inet import IP
from scapy.config import conf

from cicflowmeter.flow_session import FlowSession
import cicflowmeter.features.flow_bytes as flow_bytes_module

# ---------------------------------------------------------------------------
# Patch: cicflowmeter's flow_bytes module crashes with ValueError when a
# flow has packets in only one direction (min()/max() called on an empty
# list) - common for short captures, one-sided flows, or connections that
# never got a reply (e.g. a bare SYN). Make these safe, defaulting to 0
# instead of crashing the whole extraction run. Verified against a real
# 99-packet capture from the emulator - all flows processed successfully
# with this patch in place.
# ---------------------------------------------------------------------------
_orig_min = min
_orig_max = max


def _safe_min(*args, **kwargs):
    try:
        return _orig_min(*args, **kwargs)
    except ValueError:
        return 0


def _safe_max(*args, **kwargs):
    try:
        return _orig_max(*args, **kwargs)
    except ValueError:
        return 0


flow_bytes_module.min = _safe_min
flow_bytes_module.max = _safe_max

# If a working tcpdump substitute exists (renamed tshark.exe, kept inside
# the Wireshark install folder so it can find its sibling DLLs), point
# Scapy at it in case any codepath still checks conf.prog.tcpdump. Not
# strictly required for offline pcap reading with filter=None below, but
# harmless to set.
TCPDUMP_PATH = r"C:\Program Files\Wireshark\tcpdump.exe"
if os.path.exists(TCPDUMP_PATH):
    conf.prog.tcpdump = TCPDUMP_PATH

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join('models', 'anomaly_ocsvm.pkl')
SCALER_PATH = os.path.join('models', 'anomaly_scaler.pkl')
METADATA_PATH = os.path.join('models', 'anomaly_feature_columns.json')

model = None
scaler = None
feature_columns = []

# ---------------------------------------------------------------------------
# Column mapping: cicflowmeter (Python package) output names -> the exact
# schema names the model was trained on (CICFlowMeter Java-tool style,
# lowercase with spaces/slashes). Built by hand and verified against a
# real captured/extracted sample - all 66 trained features have a 1:1
# match, nothing missing or ambiguous.
# ---------------------------------------------------------------------------
CICFLOW_TO_SCHEMA = {
    'bwd_pkt_len_mean': 'bwd packet length mean',
    'bwd_iat_mean': 'bwd iat mean',
    'pkt_len_min': 'min packet length',
    'psh_flag_cnt': 'psh flag count',
    'bwd_pkt_len_max': 'bwd packet length max',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'bwd_iat_min': 'bwd iat min',
    'active_std': 'active std',
    'bwd_seg_size_avg': 'avg bwd segment size',
    'fwd_pkt_len_mean': 'fwd packet length mean',
    'tot_fwd_pkts': 'total fwd packets',
    'tot_bwd_pkts': 'total backward packets',
    'flow_pkts_s': 'flow packets/s',
    'down_up_ratio': 'down/up ratio',
    'ack_flag_cnt': 'ack flag count',
    'flow_duration': 'flow duration',
    'flow_iat_max': 'flow iat max',
    'fwd_pkt_len_std': 'fwd packet length std',
    'idle_max': 'idle max',
    'urg_flag_cnt': 'urg flag count',
    'pkt_len_max': 'max packet length',
    'idle_mean': 'idle mean',
    'fwd_seg_size_avg': 'avg fwd segment size',
    'fwd_pkt_len_max': 'fwd packet length max',
    'idle_min': 'idle min',
    'fwd_iat_mean': 'fwd iat mean',
    'totlen_bwd_pkts': 'total length of bwd packets',
    'flow_iat_min': 'flow iat min',
    'init_bwd_win_byts': 'init_win_bytes_backward',
    'syn_flag_cnt': 'syn flag count',
    'init_fwd_win_byts': 'init_win_bytes_forward',
    'fwd_psh_flags': 'fwd psh flags',
    'totlen_fwd_pkts': 'total length of fwd packets',
    'fwd_iat_max': 'fwd iat max',
    'subflow_bwd_byts': 'subflow bwd bytes',
    'subflow_fwd_pkts': 'subflow fwd packets',
    'src_port': 'source port',
    'flow_byts_s': 'flow bytes/s',
    'bwd_iat_max': 'bwd iat max',
    'fwd_iat_std': 'fwd iat std',
    'fwd_seg_size_min': 'min_seg_size_forward',
    'fin_flag_cnt': 'fin flag count',
    'active_mean': 'active mean',
    'fwd_pkts_s': 'fwd packets/s',
    'pkt_len_mean': 'packet length mean',
    'subflow_bwd_pkts': 'subflow bwd packets',
    'active_max': 'active max',
    'idle_std': 'idle std',
    'fwd_pkt_len_min': 'fwd packet length min',
    'flow_iat_mean': 'flow iat mean',
    'flow_iat_std': 'flow iat std',
    'bwd_pkt_len_min': 'bwd packet length min',
    'dst_port': 'destination port',
    'bwd_iat_tot': 'bwd iat total',
    'fwd_iat_tot': 'fwd iat total',
    'pkt_len_std': 'packet length std',
    'active_min': 'active min',
    'pkt_len_var': 'packet length variance',
    'pkt_size_avg': 'average packet size',
    'fwd_iat_min': 'fwd iat min',
    'bwd_pkt_len_std': 'bwd packet length std',
    'bwd_pkts_s': 'bwd packets/s',
    'bwd_header_len': 'bwd header length',
    'fwd_header_len': 'fwd header length',
    'subflow_fwd_byts': 'subflow fwd bytes',
    'bwd_iat_std': 'bwd iat std',
}


def load_model():
    """Load the trained OCSVM + scaler + feature schema once at startup"""
    global model, scaler, feature_columns

    for path in (MODEL_PATH, SCALER_PATH, METADATA_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Copy the models/ folder (anomaly_ocsvm.pkl, "
                f"anomaly_scaler.pkl, anomaly_feature_columns.json) from "
                f"where Module 3 was trained into the same folder as this "
                f"script, under a 'models' subfolder."
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    feature_columns = metadata['feature_columns']

    print(f"Module 3 model loaded: {len(feature_columns)} features")
    print(f"Training metrics: {metadata.get('metrics', {})}")


# ---------------------------------------------------------------------------
def extract_flows_from_pcap(pcap_path):
    """
    Run cicflowmeter's flow-building logic directly against a pcap file
    without shelling out to tcpdump. Manually replicates cicflowmeter's
    own "ip and (tcp or udp)" BPF filter in pure Python instead, since
    passing that filter string to Scapy forces it through an external
    tcpdump subprocess call that fails on this Windows setup.

    Returns a pandas DataFrame, one row per extracted flow, using
    cicflowmeter's native column names (not yet renamed to the schema).
    """
    tmp_csv = tempfile.NamedTemporaryFile(suffix='.csv', delete=False).name

    session = FlowSession(
        output_mode='csv',
        output=tmp_csv,
        fields=None,
        verbose=False,
    )

    def filtered_process(pkt):
        if IP in pkt and pkt[IP].proto in (6, 17):  # TCP=6, UDP=17
            session.process(pkt)

    sniffer = AsyncSniffer(
        offline=pcap_path,
        filter=None,  # avoid tcpdump dependency; filtering done manually above
        prn=filtered_process,
        store=False,
    )
    sniffer.start()
    sniffer.join()
    session.flush_flows()

    if os.path.exists(tmp_csv) and os.path.getsize(tmp_csv) > 0:
        df = pd.read_csv(tmp_csv)
    else:
        df = pd.DataFrame()

    if os.path.exists(tmp_csv):
        os.remove(tmp_csv)

    return df


def map_and_prepare_features(flow_df):
    """
    Rename cicflowmeter's native columns to the exact schema names the
    model was trained on, then select+reorder to match feature_columns
    exactly. Raises a clear error if anything required is missing rather
    than silently producing a misaligned feature vector (which would
    make the model score garbage without any obvious error).
    """
    df = flow_df.rename(columns=CICFLOW_TO_SCHEMA)

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features after mapping: {missing}")

    return df[feature_columns]


def score_flows(X_df):
    log_cols = [c for c in feature_columns if c not in ('source port', 'destination port')]
    X_log = X_df.copy()
    X_log[log_cols] = X_log[log_cols].clip(lower=0)
    X_log[log_cols] = np.log1p(X_log[log_cols])
    X_scaled = scaler.transform(X_log)

    raw_decision = model.decision_function(X_scaled)  # +ve=normal, -ve=anomaly
    predictions = model.predict(X_scaled)  # +1=normal, -1=anomaly

    steepness = 1.0
    anomaly_score = 100 / (1 + np.exp(raw_decision / steepness))

    results = []
    for i in range(len(X_df)):
        results.append({
            'anomaly_score': round(float(anomaly_score[i]), 2),
            'is_anomaly': bool(predictions[i] == -1),
            'raw_decision_value': round(float(raw_decision[i]), 4),
        })
    return results


def summarize_app_score(per_flow_results):
    """
    Combine multiple flow-level scores into one app-level anomaly score
    for the dashboard's RiskBreakdown card. Uses the MAX anomaly score
    across flows - a single suspicious flow is enough to flag the app,
    consistent with how a real analyst would treat one malicious
    connection amid otherwise-benign traffic.
    """
    if not per_flow_results:
        return {
            'anomaly_score': None,
            'is_anomaly': False,
            'flows_analyzed': 0,
            'flows_flagged': 0,
        }

    scores = [r['anomaly_score'] for r in per_flow_results]
    flagged = sum(1 for r in per_flow_results if r['is_anomaly'])

    return {
        'anomaly_score': round(max(scores), 2),
        'is_anomaly': flagged > 0,
        'flows_analyzed': len(per_flow_results),
        'flows_flagged': flagged,
    }


# ---------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'n_features': len(feature_columns),
    })


@app.route('/api/analyze/flow', methods=['POST'])
def analyze_flow_pcap():
    """
    Real pcap upload endpoint - this is what Module 7's orchestration
    logic should call once an APK's network traffic has been captured
    (e.g. via `emulator -tcpdump capture.pcap` during a sandbox run).

    Request: multipart/form-data with a file field named 'pcap'
    Response: per-flow scores + an app-level summary score, shaped for
              the dashboard's Anomaly Score card.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'pcap' not in request.files:
        return jsonify({'error': "Request must include a file field named 'pcap'"}), 400

    file = request.files['pcap']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        flow_df = extract_flows_from_pcap(tmp_path)

        if flow_df.empty:
            return jsonify({
                'anomaly_score': None,
                'is_anomaly': False,
                'flows_analyzed': 0,
                'flows_flagged': 0,
                'note': 'No TCP/UDP flows extracted from this capture',
            })

        X_df = map_and_prepare_features(flow_df)
        X_df.to_csv('debug_mapped_features.csv', index=False)
        per_flow = score_flows(X_df)
        summary = summarize_app_score(per_flow)

        return jsonify({
            **summary,
            'per_flow_detail': per_flow,
            # Raw mapped features (schema column names, pre-log-transform,
            # pre-scaling) - NOT used by this endpoint beyond scoring, but
            # exposed so Module 7's orchestrator can feed the same flows into
            # Module 5's family-clustering model without re-running pcap
            # extraction here. This is a separate consumer from score_flows()
            # above and does its OWN log-transform/scaling internally
            # (family_predictor.py) - it does not reuse this endpoint's
            # anomaly-detector preprocessing.
            'flow_features': X_df.to_dict(orient='records'),
        })

    except Exception as e:
        return jsonify({'error': f'Failed to analyze pcap: {str(e)}'}), 500

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.route('/api/analyze/flow/features', methods=['POST'])
def analyze_flow_features():
    """
    Testing/debug endpoint: accepts pre-extracted flow features directly
    as JSON, using the SCHEMA column names (not cicflowmeter's native
    names), instead of a raw pcap. Useful for testing the model/scaler
    in isolation without re-running the extraction pipeline each time.

    Request body: { "flows": [ { "flow duration": 1.66, "source port": 5353, ... }, ... ] }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    data = request.get_json(silent=True)
    if not data or 'flows' not in data:
        return jsonify({'error': "Request must include 'flows': [list of feature dicts]"}), 400

    try:
        X_df = pd.DataFrame(data['flows'])
        missing = [c for c in feature_columns if c not in X_df.columns]
        if missing:
            return jsonify({'error': f'Missing required features: {missing}'}), 400
        X_df = X_df[feature_columns]

        per_flow = score_flows(X_df)
        summary = summarize_app_score(per_flow)

        return jsonify({
            **summary,
            'per_flow_detail': per_flow,
        })
    except Exception as e:
        return jsonify({'error': f'Failed to score features: {str(e)}'}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5001, threaded=True)