"""
MPRSS Permission Risk Model Trainer - Module 2
Trains a Random Forest classifier on Drebin-215 static features and derives
per-permission/API-call risk weights from feature importances.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from static_feature_extractor import DrebinStaticFeatureExtractor


class PermissionRiskModelTrainer:
    """Train and evaluate the Random Forest permission-risk model"""

    def __init__(self, random_state=42):
        self.random_state   = random_state
        self.model           = None
        self.feature_columns = []
        self.feature_categories = {}
        self.metrics          = {}

    # ------------------------------------------------------------------
    def prepare_data(self, csv_path, categories_path, drop_exact_duplicates=True):
        """
        Load and clean Drebin-215 via DrebinStaticFeatureExtractor.

        drop_exact_duplicates=True (default) here: for TRAINING/EVALUATION we
        remove exact-duplicate rows before the train/test split so identical
        feature vectors can't appear in both splits and inflate test metrics.
        This is a stricter, more defensible choice for reporting accuracy.
        """
        extractor = DrebinStaticFeatureExtractor()
        extractor.load_dataset(csv_path)
        extractor.load_feature_categories(categories_path)
        extractor.clean_dataset(drop_exact_duplicates=drop_exact_duplicates)

        X, y = extractor.split_features_labels()

        self.feature_columns    = extractor.feature_columns
        self.feature_categories = extractor.feature_categories

        return X, y

    # ------------------------------------------------------------------
    def train(self, X, y, test_size=0.2, n_estimators=200, max_depth=None):
        """Split, train Random Forest, and store test set for evaluation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        print(f"\n{'='*50}")
        print(f"Training Random Forest")
        print(f"{'='*50}")
        print(f"Train samples: {len(X_train)}   Test samples: {len(X_test)}")

        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        self._X_test, self._y_test = X_test, y_test
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    def evaluate(self):
        """Evaluate on the held-out test set"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        y_pred = self.model.predict(self._X_test)

        self.metrics = {
            'accuracy':  round(accuracy_score(self._y_test, y_pred), 4),
            'precision': round(precision_score(self._y_test, y_pred), 4),
            'recall':    round(recall_score(self._y_test, y_pred), 4),
            'f1_score':  round(f1_score(self._y_test, y_pred), 4),
        }

        print(f"\n{'='*50}")
        print(f"Evaluation Results")
        print(f"{'='*50}")
        for k, v in self.metrics.items():
            print(f"{k:>10}: {v}")

        print(f"\nConfusion Matrix:")
        print(confusion_matrix(self._y_test, y_pred))

        print(f"\nClassification Report:")
        print(classification_report(self._y_test, y_pred, target_names=['Benign', 'Malware']))

        return self.metrics

    # ------------------------------------------------------------------
    def get_feature_risk_weights(self, top_n=None):
        """
        Extract feature_importances_ from the trained Random Forest as
        'permission risk weights', tagged with their category
        (Manifest Permission / API call signature / Intent / Command signature).
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        importances = self.model.feature_importances_
        weights = pd.DataFrame({
            'feature':   self.feature_columns,
            'weight':    importances,
            'category':  [self.feature_categories.get(f, 'Unknown') for f in self.feature_columns]
        }).sort_values('weight', ascending=False).reset_index(drop=True)

        if top_n:
            weights = weights.head(top_n)

        return weights

    # ------------------------------------------------------------------
    def save_model(self, model_path='models/permission_risk_rf.pkl'):
        """Save the trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        out_dir = os.path.dirname(model_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        joblib.dump(self.model, model_path)
        print(f"\n✅ Model saved to: {os.path.abspath(model_path)}")

    # ------------------------------------------------------------------
    def save_feature_metadata(self, path='models/feature_columns.json'):
        """
        Save the exact feature column order + categories. This MUST be used
        at inference time to build the input vector in the same order the
        model was trained on — the API (api/app.py) loads this file.
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        metadata = {
            'feature_columns':   self.feature_columns,
            'feature_categories': self.feature_categories,
            'n_features':        len(self.feature_columns),
            'trained_at':        datetime.now().isoformat(),
            'metrics':           self.metrics
        }

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Feature metadata saved to: {os.path.abspath(path)}")

    # ------------------------------------------------------------------
    def save_risk_weights_report(self, path='logs/permission_risk_weights.json'):
        """Save the full ranked risk-weight table for reporting / XAI use"""
        weights = self.get_feature_risk_weights()

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        weights.to_json(path, orient='records', indent=2)
        print(f"✅ Risk weights report saved to: {os.path.abspath(path)}")

        return weights


# ----------------------------------------------------------------------
if __name__ == "__main__":
    trainer = PermissionRiskModelTrainer(random_state=42)

    X, y = trainer.prepare_data(
        csv_path='datasets/drebin215/drebin-215-dataset-5560malware-9476-benign.csv',
        categories_path='datasets/drebin215/dataset-features-categories.csv',
        drop_exact_duplicates=True
    )

    trainer.train(X, y, test_size=0.2, n_estimators=200)
    trainer.evaluate()

    top_weights = trainer.get_feature_risk_weights(top_n=20)
    print(f"\n{'='*50}")
    print("Top 20 Risk-Weighted Features")
    print(f"{'='*50}")
    print(top_weights.to_string(index=False))

    trainer.save_model('models/permission_risk_rf.pkl')
    trainer.save_feature_metadata('models/feature_columns.json')
    trainer.save_risk_weights_report('logs/permission_risk_weights.json')

    print(f"\n{'='*50}")
    print("✅ Module 2 training complete!")
    print(f"{'='*50}")