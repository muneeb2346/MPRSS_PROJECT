"""
MPRSS Flow Feature Extractor - Module 3 (v2 - log-transform fix)
Loads, cleans, and prepares merged_dataset.csv (CICFlowMeter-style network
flow features) for the One-Class SVM anomaly detection model.

CHANGE FROM v1: added a log1p transform on skewed magnitude features
(byte counts, packet counts, durations, rates, IAT values, window sizes)
before scaling. This fixes a real, diagnosed problem: merged_dataset.csv's
training flows are long, sustained network sessions with very different
byte/duration magnitudes than short, real-world mobile-app flows captured
by Module 7's sandbox. Without log-compression, real captured flows land
so far outside the RBF kernel's support-vector neighborhood that the
exponential term in decision_function() underflows to exactly 0 for every
support vector - causing every real capture to receive the SAME saturated
score regardless of how different the underlying traffic actually is (this
was confirmed empirically: 36 wildly different flows all produced the
identical decision_function value -0.8971). Log-compressing heavy-tailed
features before scaling brings both distributions into comparable ranges,
which should let the model actually discriminate between real captures
instead of saturating on all of them identically.

Label semantics confirmed: 1 = malicious, 0 = benign.
Benign is the minority class (~7% of the full dataset) — this is expected
and fine for one-class SVM, which trains only on the 'normal' (benign) class.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class FlowFeatureExtractor:
    """Load and clean the merged flow-based dataset for anomaly detection"""

    def __init__(self, label_column='label', malicious_value=1, benign_value=0):
        self.label_column   = label_column
        self.malicious_value = malicious_value
        self.benign_value    = benign_value

        self.raw_df   = None
        self.clean_df = None
        self.feature_columns   = []
        self.dropped_columns   = []
        self.scaler = StandardScaler()
        self.is_fitted = False

        # Columns that are identifiers/categorical-ish rather than magnitude
        # measures - log-transforming these wouldn't help and could distort
        # them. Everything else in feature_columns gets log1p applied.
        self.no_log_columns = set()  # log-transform everything, including ports — see saturation debugging notes

    # ------------------------------------------------------------------
    def load_dataset(self, csv_path):
        """Load the raw merged_dataset.csv"""
        print(f"\n{'='*50}")
        print(f"Loading dataset: {csv_path}")
        print(f"{'='*50}")

        self.raw_df = pd.read_csv(csv_path)
        print(f"Loaded {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")

        return self.raw_df

    # ------------------------------------------------------------------
    def clean_dataset(self, variance_threshold=1e-8):
        """
        Clean the raw dataset:
          1. Fill any missing numeric values with 0
          2. Drop near-zero-variance columns (computed from the actual data,
             not hardcoded — e.g. 'protocol' was constant in initial checks,
             but this is verified fresh each run in case the full dataset
             has more variety than the sample)
        """
        if self.raw_df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        df = self.raw_df.copy()

        all_cols = [c for c in df.columns if c != self.label_column]

        # --- Missing values ---
        n_missing = df[all_cols].isnull().sum().sum()
        if n_missing > 0:
            df[all_cols] = df[all_cols].fillna(0)
            print(f"Filled {n_missing} missing values with 0")
        else:
            print("No missing values found")

        # --- Drop near-zero-variance columns ---
        variances = df[all_cols].var()
        self.dropped_columns = variances[variances <= variance_threshold].index.tolist()
        self.feature_columns = [c for c in all_cols if c not in self.dropped_columns]

        if self.dropped_columns:
            print(f"Dropped {len(self.dropped_columns)} near-zero-variance column(s): "
                  f"{self.dropped_columns}")
        print(f"Remaining feature columns: {len(self.feature_columns)}")

        # --- Exact duplicates (report only, keep by default — see note in
        # static_feature_extractor.py from Module 2 for the same reasoning) ---
        n_dupes = df.duplicated().sum()
        print(f"Duplicate rows present: {n_dupes} of {len(df)} "
              f"({n_dupes/len(df)*100:.1f}%)")

        print(f"\nLabel distribution:")
        print(df[self.label_column].value_counts().rename({
            self.malicious_value: 'Malicious', self.benign_value: 'Benign'
        }))

        self.clean_df = df
        return df

    # ------------------------------------------------------------------
    def _log_transform(self, df):
        """
        Apply log1p to skewed magnitude columns (everything except ports),
        clipping any stray negative values to 0 first (shouldn't occur in
        well-formed flow stats, but log1p is undefined below -1).
        """
        df = df.copy()
        log_cols = [c for c in self.feature_columns if c not in self.no_log_columns]
        df[log_cols] = df[log_cols].clip(lower=0)
        df[log_cols] = np.log1p(df[log_cols])
        return df

    # ------------------------------------------------------------------
    def get_benign_malicious_split(self):
        """Return (benign_df, malicious_df) using confirmed label semantics"""
        if self.clean_df is None:
            raise ValueError("No cleaned dataset. Call clean_dataset() first.")

        benign    = self.clean_df[self.clean_df[self.label_column] == self.benign_value]
        malicious = self.clean_df[self.clean_df[self.label_column] == self.malicious_value]

        return benign, malicious

    # ------------------------------------------------------------------
    def fit_scaler(self, benign_train_df):
        """
        Fit the StandardScaler on BENIGN TRAINING data only (standard ML
        practice — never fit on test data, and for one-class SVM the model
        only ever trains on the 'normal' class anyway). Now fits on
        LOG-TRANSFORMED features (see _log_transform) rather than raw
        magnitudes.
        """
        transformed = self._log_transform(benign_train_df[self.feature_columns])
        self.scaler.fit(transformed)
        self.is_fitted = True
        print(f"Scaler fitted on {len(benign_train_df)} benign training rows (log-transformed)")

    def transform(self, df):
        """Log-transform then scale a dataframe's feature columns using the fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")
        transformed = self._log_transform(df[self.feature_columns])
        return self.scaler.transform(transformed)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    extractor = FlowFeatureExtractor()

    extractor.load_dataset('datasets/merged_dataset.csv')
    extractor.clean_dataset()

    benign, malicious = extractor.get_benign_malicious_split()
    print(f"\nBenign rows:    {len(benign)}")
    print(f"Malicious rows: {len(malicious)}")