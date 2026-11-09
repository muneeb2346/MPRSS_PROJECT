"""
MPRSS Malware Family Clustering - FLOW/behavioral side (Module 5, Function 1 & 2)
Groups MALICIOUS network flows (merged_dataset.csv) into K-Means clusters
based on CICFlowMeter-style flow statistics - a BEHAVIORAL counterpart to
static_family_cluster_trainer.py's permission/API-based clustering.

WHY A SEPARATE SCALER FROM MODULE 3's FlowFeatureExtractor:
Module 3's scaler is fit on BENIGN flows only (it's a one-class anomaly
detector - see flow_feature_extractor.py). Reusing that fitted scaler here
to transform MALICIOUS flows for clustering would silently apply the wrong
distribution - the exact class of bug found and fixed in Module 3's
saturation issue (train/inference preprocessing must always match; a
scaler fit on one distribution must never be applied to a different one).
This trainer fits its OWN scaler on the malicious subset specifically, and
saves it as a clearly separate artifact (models/flow_family_scaler.pkl,
never anomaly_scaler.pkl) so the two can never be confused or swapped.

Feature engineering matches Module 3's log1p fix for the same reason it was
needed there: flow byte/duration/packet counts are heavy-tailed, and log-
compression is needed for K-Means' Euclidean distance (like OneClassSVM's
RBF distance) to be meaningful rather than dominated by a few huge flows.

NO GROUND-TRUTH FAMILY LABELS exist (only benign/malicious 0/1) - as with
the static side, clusters are validated via silhouette score and labeled
with a feature-based INTERPRETATION, not a verified family name.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from cluster_utils import select_optimal_k, get_cluster_top_features, plot_k_selection, stratified_sample_by_cluster


class FlowFamilyClusterTrainer:
    """Train and evaluate K-Means clustering over malicious network flows"""

    def __init__(self, random_state=42, label_column='label', malicious_value=1):
        self.random_state = random_state
        self.label_column = label_column
        self.malicious_value = malicious_value

        self.scaler = StandardScaler()
        self.kmeans = None
        self.feature_columns = []
        self.dropped_columns = []
        self.X_malicious_raw = None
        self.k_selection_results = []
        self.best_k = None
        self.interpretations = {}

    # ------------------------------------------------------------------
    def _log_transform(self, df):
        df = df.copy()
        df[self.feature_columns] = df[self.feature_columns].clip(lower=0)
        df[self.feature_columns] = np.log1p(df[self.feature_columns])
        return df

    # ------------------------------------------------------------------
    def prepare_data(self, csv_path, variance_threshold=1e-8):
        """
        Load merged_dataset.csv, keep MALICIOUS rows only, drop near-zero-
        variance columns (recomputed on the malicious subset specifically -
        a column that varies overall but is constant within malware wouldn't
        help distinguish malware families), then log-transform and scale.
        """
        print(f"\n{'='*50}")
        print(f"Loading dataset: {csv_path}")
        print(f"{'='*50}")

        df = pd.read_csv(csv_path)
        all_cols = [c for c in df.columns if c != self.label_column]
        df[all_cols] = df[all_cols].fillna(0)

        malicious_df = df[df[self.label_column] == self.malicious_value].reset_index(drop=True)
        print(f"Malicious rows for clustering: {len(malicious_df)} of {len(df)} total")

        variances = malicious_df[all_cols].var()
        self.dropped_columns = variances[variances <= variance_threshold].index.tolist()
        self.feature_columns = [c for c in all_cols if c not in self.dropped_columns]

        if self.dropped_columns:
            print(f"Dropped {len(self.dropped_columns)} near-zero-variance column(s) "
                  f"(within malicious subset): {self.dropped_columns}")
        print(f"Remaining feature columns: {len(self.feature_columns)}")

        self.X_malicious_raw = malicious_df[self.feature_columns].reset_index(drop=True)

        X_log = self._log_transform(self.X_malicious_raw)
        X_scaled = self.scaler.fit_transform(X_log)

        return X_scaled

    # ------------------------------------------------------------------
    def select_k(self, X_scaled, k_range=range(2, 11)):
        print(f"\n{'='*50}")
        print("Selecting optimal k (elbow + silhouette)")
        print(f"{'='*50}")
        self.k_selection_results, self.best_k = select_optimal_k(
            X_scaled, k_range=k_range, random_state=self.random_state
        )
        print(f"\nBest k by silhouette score: {self.best_k}")
        return self.best_k

    # ------------------------------------------------------------------
    def train(self, X_scaled, k=None):
        k = k or self.best_k
        if k is None:
            raise ValueError("No k specified and select_k() has not been run.")

        print(f"\n{'='*50}")
        print(f"Training final KMeans with k={k}")
        print(f"{'='*50}")

        self.kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_labels = self.kmeans.fit_predict(X_scaled)

        print(f"Cluster sizes: {np.bincount(cluster_labels).tolist()}")

        self.interpretations = get_cluster_top_features(
            self.X_malicious_raw, cluster_labels, self.feature_columns, top_n=10
        )
        self._cluster_labels = cluster_labels

        return cluster_labels

    # ------------------------------------------------------------------
    def print_interpretations(self):
        print(f"\n{'='*50}")
        print("Cluster interpretations (feature-based, NOT verified family names)")
        print(f"{'='*50}")
        for cluster_id, info in self.interpretations.items():
            print(f"\nCluster {cluster_id} \u2014 {info['cluster_size']} samples "
                  f"({info['cluster_pct']}% of malicious flows)")
            for feat in info['top_distinguishing_features'][:5]:
                print(f"    {feat['feature']:<35} cluster={feat['cluster_mean']:.2f}  "
                      f"overall={feat['overall_mean']:.2f}  (+{feat['deviation']:.2f})")

    # ------------------------------------------------------------------
    def get_balanced_training_sample(self, n_per_cluster=None):
        """Module 5 Function 2: dataset balancing. Returns malicious flows
        sampled evenly across clusters, for retraining Module 3/6 without one
        behavioral pattern dominating."""
        return stratified_sample_by_cluster(
            self.X_malicious_raw, self._cluster_labels,
            n_per_cluster=n_per_cluster, random_state=self.random_state
        )

    # ------------------------------------------------------------------
    def save_model(self, model_path='models/flow_family_kmeans.pkl'):
        out_dir = os.path.dirname(model_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.kmeans, model_path)
        print(f"\n✅ KMeans model saved to: {os.path.abspath(model_path)}")

    def save_scaler(self, scaler_path='models/flow_family_scaler.pkl'):
        """
        IMPORTANT: this is a SEPARATE scaler from Module 3's anomaly_scaler.pkl.
        Never mix them up - this one is fit on MALICIOUS flows for clustering;
        Module 3's is fit on BENIGN flows for one-class anomaly detection.
        """
        out_dir = os.path.dirname(scaler_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to: {os.path.abspath(scaler_path)}")

    def save_metadata(self, path='models/flow_family_metadata.json'):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        metadata = {
            'feature_columns': self.feature_columns,
            'dropped_columns': self.dropped_columns,
            'best_k': self.best_k,
            'k_selection_results': self.k_selection_results,
            'cluster_interpretations': self.interpretations,
            'trained_at': datetime.now().isoformat(),
            'note': ('Cluster labels are INTERPRETATIONS based on distinguishing '
                     'flow features, not verified ground-truth family names - no '
                     'labeled family dataset was available. This scaler is '
                     'separate from Module 3\'s anomaly_scaler.pkl - do not swap them.')
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Metadata saved to: {os.path.abspath(path)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = FlowFamilyClusterTrainer(random_state=42)

    X_scaled = trainer.prepare_data(csv_path='datasets/merged_dataset.csv')

    best_k = trainer.select_k(X_scaled, k_range=range(2, 11))
    plot_k_selection(trainer.k_selection_results,
                      save_path='logs/flow_family_k_selection.png',
                      title_prefix='Flow Family Clustering - ')

    trainer.train(X_scaled, k=best_k)
    trainer.print_interpretations()

    trainer.save_model('models/flow_family_kmeans.pkl')
    trainer.save_scaler('models/flow_family_scaler.pkl')
    trainer.save_metadata('models/flow_family_metadata.json')

    print(f"\n{'='*50}")
    print("✅ Module 5 (flow clustering) training complete!")
    print(f"{'='*50}")