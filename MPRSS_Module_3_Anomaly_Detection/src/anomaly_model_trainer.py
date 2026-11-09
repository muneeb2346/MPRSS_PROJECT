"""
MPRSS Anomaly Detection Model Trainer - Module 3
Trains a One-Class SVM on BENIGN-ONLY flow data to detect anomalous
(potentially zero-day) traffic patterns, per the proposal's specification.

Why One-Class SVM trains on benign data only: it learns the boundary of
'normal' behavior from benign samples alone, then flags anything outside
that boundary as an anomaly — this is what lets it catch malware patterns
it has never seen before (zero-day), unlike a supervised classifier that
can only recognize attack types present in its training labels.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from flow_feature_extractor import FlowFeatureExtractor


class AnomalyModelTrainer:
    """Train and evaluate the One-Class SVM anomaly detector"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.extractor = None
        self.feature_columns = []
        self.metrics = {}

    # ------------------------------------------------------------------
    def prepare_data(self, csv_path, test_size=0.25, train_sample_size=8000):
        """
        Load, clean, and split into train/test sets.

        One-Class SVM (RBF kernel) has roughly O(n^2)-O(n^3) training cost,
        so training on the full benign class (~24.5k rows in the full
        dataset) can be slow. train_sample_size caps how many benign rows
        are used for TRAINING ONLY — evaluation still uses a full held-out
        test set (benign + malicious) so reported metrics aren't inflated
        by a smaller test set.
        """
        self.extractor = FlowFeatureExtractor()
        self.extractor.load_dataset(csv_path)
        self.extractor.clean_dataset()
        self.feature_columns = self.extractor.feature_columns

        benign, malicious = self.extractor.get_benign_malicious_split()

        # Split BENIGN into train/test — model only ever trains on benign
        benign_train, benign_test = train_test_split(
            benign, test_size=test_size, random_state=self.random_state
        )

        # Cap training set size for speed (see docstring)
        if len(benign_train) > train_sample_size:
            print(f"\nSubsampling benign training set: {len(benign_train)} -> "
                  f"{train_sample_size} rows (for OneClassSVM training speed)")
            benign_train = benign_train.sample(n=train_sample_size, random_state=self.random_state)

        # Malicious rows are NEVER used for training — only for evaluation
        malicious_test = malicious

        print(f"\n{'='*50}")
        print(f"Data split")
        print(f"{'='*50}")
        print(f"Benign  (train, normal-only): {len(benign_train)}")
        print(f"Benign  (test):               {len(benign_test)}")
        print(f"Malicious (test, unseen by model): {len(malicious_test)}")

        # Fit scaler on training data only
        self.extractor.fit_scaler(benign_train)

        X_train = self.extractor.transform(benign_train)
        X_benign_test    = self.extractor.transform(benign_test)
        X_malicious_test = self.extractor.transform(malicious_test)

        self._X_benign_test    = X_benign_test
        self._X_malicious_test = X_malicious_test

        return X_train

    # ------------------------------------------------------------------
    def train(self, X_train, nu=0.05, kernel='rbf', gamma=0.1):
        """
        Train the One-Class SVM.

        nu: upper bound on the fraction of training errors / lower bound on
            the fraction of support vectors.
        gamma=0.1 (not the sklearn default 'scale'): a grid search over
            nu x gamma on held-out data showed gamma is the dominant lever
            for this dataset — gamma='scale'/'auto' (~0.015 effective value
            for this data) gives poor separation (~10% detection rate at
            ~10% false-positive rate, effectively random). gamma=0.1 was
            the best-performing setting found, trading a higher false
            positive rate (~30-35%) for meaningfully better detection
            (~50%). See logs/anomaly_hyperparameter_search.json for the
            full sweep. This trade-off should be re-validated once trained
            on the FULL benign class (24.5k rows) rather than a subsample —
            more training data typically tightens the learned boundary.
        """
        print(f"\n{'='*50}")
        print(f"Training One-Class SVM (kernel={kernel}, nu={nu}, gamma={gamma})")
        print(f"{'='*50}")

        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self.model.fit(X_train)

        print(f"Support vectors: {len(self.model.support_)} of {len(X_train)} training samples")
        return self.model

    # ------------------------------------------------------------------
    def evaluate(self):
        """
        Evaluate on held-out benign + malicious test sets.

        OneClassSVM.predict() returns +1 for 'normal' (inlier) and
        -1 for 'anomaly' (outlier). We map that to the malware-detection
        framing: -1 (anomaly) == flagged as malicious/suspicious.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        pred_benign    = self.model.predict(self._X_benign_test)
        pred_malicious = self.model.predict(self._X_malicious_test)

        # +1 = normal, -1 = anomaly
        y_true = np.concatenate([
            np.zeros(len(pred_benign)),      # 0 = benign
            np.ones(len(pred_malicious))      # 1 = malicious
        ])
        y_pred = np.concatenate([
            (pred_benign == -1).astype(int),
            (pred_malicious == -1).astype(int)
        ])

        false_positive_rate = (pred_benign == -1).sum() / len(pred_benign)
        detection_rate       = (pred_malicious == -1).sum() / len(pred_malicious)

        self.metrics = {
            'benign_test_size':     len(pred_benign),
            'malicious_test_size':  len(pred_malicious),
            'false_positive_rate':  round(float(false_positive_rate), 4),
            'anomaly_detection_rate': round(float(detection_rate), 4),
        }

        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        print(f"False positive rate (benign flagged as anomaly): {false_positive_rate*100:.2f}%")
        print(f"Anomaly detection rate (malicious flagged as anomaly): {detection_rate*100:.2f}%")

        print(f"\nConfusion Matrix (rows=true, cols=predicted) [benign, malicious]:")
        print(confusion_matrix(y_true, y_pred))

        print(f"\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious (Anomaly)']))

        return self.metrics

    # ------------------------------------------------------------------
    def save_model(self, model_path='models/anomaly_ocsvm.pkl'):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        out_dir = os.path.dirname(model_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.model, model_path)
        print(f"\n✅ Model saved to: {os.path.abspath(model_path)}")

    def save_scaler(self, scaler_path='models/anomaly_scaler.pkl'):
        out_dir = os.path.dirname(scaler_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self.extractor.scaler, scaler_path)
        print(f"✅ Scaler saved to: {os.path.abspath(scaler_path)}")

    def save_feature_metadata(self, path='models/anomaly_feature_columns.json'):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        metadata = {
            'feature_columns': self.feature_columns,
            'dropped_columns': self.extractor.dropped_columns,
            'n_features':      len(self.feature_columns),
            'trained_at':      datetime.now().isoformat(),
            'metrics':         self.metrics
        }
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Feature metadata saved to: {os.path.abspath(path)}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = AnomalyModelTrainer(random_state=42)

    X_train = trainer.prepare_data(
        csv_path='datasets/merged_dataset.csv',
        test_size=0.25,
        train_sample_size=8000
    )

    trainer.train(X_train, nu=0.05, kernel='rbf', gamma=0.1)
    trainer.evaluate()

    trainer.save_model('models/anomaly_ocsvm.pkl')
    trainer.save_scaler('models/anomaly_scaler.pkl')
    trainer.save_feature_metadata('models/anomaly_feature_columns.json')

    print(f"\n{'='*50}")
    print("✅ Module 3 training complete!")
    print(f"{'='*50}")