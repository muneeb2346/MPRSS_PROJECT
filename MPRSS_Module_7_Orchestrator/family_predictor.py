"""
MPRSS Module 5 - Family Classification Inference
Loads the trained static and flow K-Means clustering models and predicts
which cluster a NEW APK's static features / captured flows fall into, for
display on the dashboard's Family Classification card.

IMPORTANT CAVEAT - CLUSTER IDS ARE NOT STABLE ACROSS RETRAINING:
K-Means cluster IDs (0, 1, 2...) are arbitrary labels assigned at training
time - they carry no inherent meaning and are NOT guaranteed to represent
the same behavior if the trainer is ever rerun (e.g. cluster 0 could be
"SMS-trojan-like" in one training run and "rootkit-like" in the next,
purely due to random initialization order). The STATIC_CLUSTER_LABELS and
FLOW_CLUSTER_LABELS dictionaries below were manually written by inspecting
each cluster's top_distinguishing_features in the metadata JSON produced by
static_family_cluster_trainer.py / flow_family_cluster_trainer.py at the
time of writing this file. If either trainer is rerun, these labels MUST be
manually re-checked against the new metadata before trusting them again -
this file does not do that automatically.

CONFIDENCE IS A HEURISTIC, NOT A CALIBRATED PROBABILITY:
K-Means has no native notion of prediction confidence (unlike e.g. a
Gaussian Mixture Model). "confidence" here is a softmax over the NEGATIVE
distances to each cluster center (closer center -> higher weight) - a
reasonable relative measure of "how decisively does this sample belong to
its assigned cluster vs. the alternatives", but not a true probability.
Documented the same way as the honesty notes elsewhere in this project
(Module 3's FPR/detection tradeoff, Module 6's non-empirically-tuned
weights).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Manually-assigned human-readable labels, based on the cluster
# interpretation analysis done alongside training (see project notes /
# report). Keyed by cluster_id as assigned by the CURRENTLY TRAINED models
# in models/. Re-verify against static_family_metadata.json /
# flow_family_metadata.json's cluster_interpretations if the trainers are
# ever rerun.
# ----------------------------------------------------------------------
STATIC_CLUSTER_LABELS = {
    0: {
        'name': 'SMS-Trojan',
        'description': ('Elevated SEND_SMS / RECEIVE_SMS / SmsManager usage - '
                         'consistent with premium-SMS fraud or SMS-based trojans.'),
    },
    1: {
        'name': 'SMS-Trojan (Background Service)',
        'description': ('Binder/onBind service-binding APIs combined with SMS '
                         'permissions - consistent with SMS abuse operating via '
                         'a background bound service rather than a foreground app.'),
    },
    2: {
        'name': 'Rooting / Privileged Backdoor',
        'description': ('System file manipulation (/system/bin, chmod), native '
                         'library loading, and cryptographic API usage - consistent '
                         'with privilege escalation or an advanced backdoor.'),
    },
    3: {
        'name': 'Browser-History Spyware',
        'description': ('Browser history read/write combined with reflection-based '
                         'class loading (ClassLoader/findClass) - consistent with '
                         'spyware harvesting browsing data, obfuscated via reflection.'),
    },
    4: {
        'name': 'Device-Fingerprinting Spyware',
        'description': ('SIM/subscriber ID collection plus shell command execution '
                         '(Runtime.exec) - consistent with spyware that fingerprints '
                         'the device and can run arbitrary commands.'),
    },
    5: {
        'name': 'Invasive System-Control Trojan',
        'description': ('System settings manipulation, broadcast suppression '
                         '(abortBroadcast), and SMS writing - consistent with a '
                         'trojan that takes invasive control and can hide its own traces.'),
    },
    6: {
        'name': 'Delayed-Payload Trojan',
        'description': ('Triggers on ACTION_POWER_CONNECTED with contact access and '
                         'dynamic dex loading - consistent with a trojan that waits '
                         'for an opportune moment (e.g. charging) before loading its '
                         'real payload.'),
    },
}

FLOW_CLUSTER_LABELS = {
    0: {
        'name': 'Beaconing (Spaced Outbound Signaling)',
        'description': ('Handshakes reliably complete, but forward-direction packets '
                         'are spaced apart rather than sent in a burst - consistent '
                         'with periodic C2 check-ins.'),
    },
    1: {
        'name': 'Generic Background Traffic',
        'description': ('No feature shows a strong deviation from baseline - this is '
                         'the default bucket for routine, non-distinctive malicious flows.'),
    },
    2: {
        'name': 'Minimal / Probing Packets',
        'description': ('Packets are consistently tiny (near-empty payloads) with an '
                         'elevated URG flag - consistent with control signaling or '
                         'connection probing rather than data transfer.'),
    },
    3: {
        'name': 'Beaconing (Long Idle Gaps)',
        'description': ('Long idle periods punctuate each flow rather than continuous '
                         'activity - consistent with a connect-idle-checkback C2 pattern.'),
    },
    4: {
        'name': 'Payload Download (Heavy Backward Volume)',
        'description': ('Far more data flows device-ward than baseline - consistent '
                         'with downloading additional payloads or receiving bulk content.'),
    },
    5: {
        'name': 'Burst Exfiltration',
        'description': ('Extremely high packet/byte rate, far beyond any other cluster - '
                         'consistent with a rapid data-exfiltration burst.'),
    },
}

_GENERIC_FALLBACK = {
    'name': 'Unclassified Pattern',
    'description': 'This cluster ID has no manually-assigned label yet - check the metadata JSON.',
}


class FamilyPredictor:
    """Loads Module 5's trained artifacts once and predicts cluster membership
    for new static feature sets / captured flows."""

    def __init__(self):
        self.static_kmeans = None
        self.static_pca = None
        self.static_feature_columns = []

        self.flow_kmeans = None
        self.flow_scaler = None
        self.flow_feature_columns = []

    # ------------------------------------------------------------------
    def load(self, models_dir='models'):
        static_kmeans_path = os.path.join(models_dir, 'static_family_kmeans.pkl')
        static_pca_path = os.path.join(models_dir, 'static_family_pca.pkl')
        static_meta_path = os.path.join(models_dir, 'static_family_metadata.json')

        flow_kmeans_path = os.path.join(models_dir, 'flow_family_kmeans.pkl')
        flow_scaler_path = os.path.join(models_dir, 'flow_family_scaler.pkl')
        flow_meta_path = os.path.join(models_dir, 'flow_family_metadata.json')

        for path in (static_kmeans_path, static_pca_path, static_meta_path,
                     flow_kmeans_path, flow_scaler_path, flow_meta_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Module 5 artifact not found: {path}\n"
                    f"Copy all 6 Module 5 model files into '{models_dir}/' "
                    f"relative to this script."
                )

        self.static_kmeans = joblib.load(static_kmeans_path)
        self.static_pca = joblib.load(static_pca_path)
        with open(static_meta_path) as f:
            self.static_feature_columns = json.load(f)['feature_columns']

        self.flow_kmeans = joblib.load(flow_kmeans_path)
        self.flow_scaler = joblib.load(flow_scaler_path)
        with open(flow_meta_path) as f:
            self.flow_feature_columns = json.load(f)['feature_columns']

        print(f"Module 5 family models loaded: "
              f"static={self.static_kmeans.n_clusters} clusters "
              f"({len(self.static_feature_columns)} features -> "
              f"{self.static_pca.n_components_} PCA components), "
              f"flow={self.flow_kmeans.n_clusters} clusters "
              f"({len(self.flow_feature_columns)} features)")

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_confidence(distances, chosen_idx):
        """Convert K-Means center distances into a relative confidence score
        for the chosen cluster. See module docstring - heuristic, not a
        calibrated probability."""
        neg = -np.asarray(distances)
        neg = neg - neg.max()  # numerical stability
        weights = np.exp(neg)
        weights = weights / weights.sum()
        return float(weights[chosen_idx])

    # ------------------------------------------------------------------
    def predict_static_family(self, active_features):
        """
        active_features : list[str] - feature names that are ACTIVE for this
            APK, in the same format apk_static_parser.py produces (Module 2).
            Any name not in self.static_feature_columns is silently ignored
            (e.g. a feature that only exists in one Drebin version).
        """
        if self.static_kmeans is None:
            raise RuntimeError("Models not loaded - call load() first.")

        active_set = set(active_features or [])
        vector = np.array([[1 if f in active_set else 0
                             for f in self.static_feature_columns]])

        X_pca = self.static_pca.transform(vector)
        cluster_id = int(self.static_kmeans.predict(X_pca)[0])
        distances = self.static_kmeans.transform(X_pca)[0]
        confidence = self._softmax_confidence(distances, cluster_id)

        label = STATIC_CLUSTER_LABELS.get(cluster_id, _GENERIC_FALLBACK)

        return {
            'cluster_id': cluster_id,
            'name': label['name'],
            'description': label['description'],
            'confidence': round(confidence, 4),
        }

    # ------------------------------------------------------------------
    def predict_flow_family(self, flow_records):
        """
        flow_records : list[dict] - one dict per captured flow, using the
            SAME schema column names as Module 3's mapped feature dataframe
            (see app.py's CICFLOW_TO_SCHEMA / map_and_prepare_features).
            Missing columns are filled with 0 (with a printed warning) so a
            partially-mismatched schema doesn't crash the whole request.
        """
        if self.flow_kmeans is None:
            raise RuntimeError("Models not loaded - call load() first.")

        if not flow_records:
            return {
                'cluster_id': None,
                'name': None,
                'description': None,
                'confidence': None,
                'flows_analyzed': 0,
                'cluster_distribution': {},
                'note': 'No flows available to classify.',
            }

        df = pd.DataFrame(flow_records)

        missing = [c for c in self.flow_feature_columns if c not in df.columns]
        if missing:
            print(f"[family_predictor] WARNING: {len(missing)} expected flow "
                  f"feature(s) missing, filling with 0: {missing}")
            for c in missing:
                df[c] = 0

        X = df[self.flow_feature_columns].copy()
        X = X.clip(lower=0)
        X_log = np.log1p(X)
        X_scaled = self.flow_scaler.transform(X_log)

        cluster_ids = self.flow_kmeans.predict(X_scaled)
        distances = self.flow_kmeans.transform(X_scaled)

        # App-level family = majority cluster across all captured flows
        counts = np.bincount(cluster_ids, minlength=self.flow_kmeans.n_clusters)
        majority_cluster = int(np.argmax(counts))

        # Average confidence across flows assigned to the majority cluster
        majority_mask = (cluster_ids == majority_cluster)
        majority_confidences = [
            self._softmax_confidence(distances[i], majority_cluster)
            for i in range(len(cluster_ids)) if majority_mask[i]
        ]
        avg_confidence = float(np.mean(majority_confidences)) if majority_confidences else 0.0

        label = FLOW_CLUSTER_LABELS.get(majority_cluster, _GENERIC_FALLBACK)

        return {
            'cluster_id': majority_cluster,
            'name': label['name'],
            'description': label['description'],
            'confidence': round(avg_confidence, 4),
            'flows_analyzed': len(cluster_ids),
            'cluster_distribution': {
                str(cid): int(counts[cid]) for cid in range(len(counts)) if counts[cid] > 0
            },
        }


# ----------------------------------------------------------------------
# Module-level singleton, mirroring the load-once-at-startup pattern used
# in Module 3's app.py (load_model()).
# ----------------------------------------------------------------------
_predictor = FamilyPredictor()


def load_family_models(models_dir='models'):
    _predictor.load(models_dir)


def predict_static_family(active_features):
    return _predictor.predict_static_family(active_features)


def predict_flow_family(flow_records):
    return _predictor.predict_flow_family(flow_records)