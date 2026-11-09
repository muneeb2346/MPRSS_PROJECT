import pandas as pd
import numpy as np
import os


class DrebinStaticFeatureExtractor:
    """Load and clean the Drebin-215 static feature dataset"""

    def __init__(self):
        self.raw_df           = None
        self.clean_df         = None
        self.feature_columns  = []
        self.feature_categories = {}   # feature_name -> category
        self.label_column     = 'class'

    # ------------------------------------------------------------------
    def load_dataset(self, csv_path):
        """Load the raw Drebin-215 CSV"""
        print(f"\n{'='*50}")
        print(f"Loading Drebin-215 dataset: {csv_path}")
        print(f"{'='*50}")

        self.raw_df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {self.raw_df.shape[0]} rows, {self.raw_df.shape[1]} columns")

        self.feature_columns = [c for c in self.raw_df.columns if c != self.label_column]
        return self.raw_df

    # ------------------------------------------------------------------
    def load_feature_categories(self, categories_csv_path):
        """
        Load dataset-features-categories.csv and build a feature -> category map.
        Drops the stray row that describes the label column itself (not a feature).
        """
        # NOTE: this file has NO header row — its first line is real data
        # ('transact,API call signature'). Reading with default header=0
        # would silently swallow the 'transact' feature as a column name.
        cats_df = pd.read_csv(categories_csv_path, header=None, names=['feature', 'category'])
        name_col, cat_col = 'feature', 'category'

        # Drop the row that documents the label encoding (class / B=Benign;S=Malware)
        cats_df = cats_df[cats_df[name_col] != self.label_column]

        self.feature_categories = dict(zip(cats_df[name_col], cats_df[cat_col]))

        print(f"\nFeature categories loaded: {len(self.feature_categories)} features")
        print(pd.Series(self.feature_categories).value_counts())

        return self.feature_categories

    # ------------------------------------------------------------------
    def clean_dataset(self, drop_exact_duplicates=False):
        """
        Clean the raw dataset:
          1. Fix 'TelephonyManager.getSimCountryIso' — replace '?' (unknown) with 0, cast to int
          2. Encode label column: 'S' (malware) -> 1, 'B' (benign) -> 0
          3. Optionally drop exact duplicate rows (off by default — see note below)

        Note on duplicates: ~52% of Drebin-215 rows are exact duplicates. This is
        expected for a sparse binary feature space (many apps share identical
        permission/API-call fingerprints), not corrupted data. Duplicates are kept
        by default so class prevalence isn't distorted; dedup, if used, should
        happen consistently with how the train/test split is performed
        (see risk_model_trainer.py) to avoid data leakage.
        """
        if self.raw_df is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")

        df = self.raw_df.copy()

        # --- Fix the one mixed-type column ---
        problem_col = 'TelephonyManager.getSimCountryIso'
        if problem_col in df.columns:
            n_unknown = (df[problem_col] == '?').sum()
            df[problem_col] = df[problem_col].replace('?', 0).astype(int)
            print(f"Fixed '{problem_col}': replaced {n_unknown} '?' values with 0")

        # --- Ensure every other feature column is numeric ---
        for col in self.feature_columns:
            if col != problem_col:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # --- Encode label ---
        df[self.label_column] = df[self.label_column].map({'S': 1, 'B': 0})

        n_before = len(df)
        if drop_exact_duplicates:
            df = df.drop_duplicates().reset_index(drop=True)
            print(f"Dropped exact duplicates: {n_before} -> {len(df)} rows")
        else:
            n_dupes = df.duplicated().sum()
            print(f"Duplicates present but kept: {n_dupes} of {n_before} rows "
                  f"({n_dupes/n_before*100:.1f}%)")

        print(f"\nClass distribution after cleaning:")
        print(df[self.label_column].value_counts().rename({1: 'Malware (S)', 0: 'Benign (B)'}))

        self.clean_df = df
        return df

    # ------------------------------------------------------------------
    def split_features_labels(self):
        """Return (X, y) from the cleaned dataset"""
        if self.clean_df is None:
            raise ValueError("No cleaned dataset. Call clean_dataset() first.")

        X = self.clean_df[self.feature_columns].copy()
        y = self.clean_df[self.label_column].copy()
        return X, y

    # ------------------------------------------------------------------
    def get_category_summary(self):
        """Return counts of features per category (Permission / API call / Intent / Command)"""
        if not self.feature_categories:
            raise ValueError("No categories loaded. Call load_feature_categories() first.")
        return pd.Series(self.feature_categories).value_counts()

    # ------------------------------------------------------------------
    def features_by_category(self, category):
        """Return list of feature column names belonging to a given category
        e.g. features_by_category('Manifest Permission')"""
        return [f for f, c in self.feature_categories.items() if c == category]


# ----------------------------------------------------------------------
if __name__ == "__main__":
    extractor = DrebinStaticFeatureExtractor()

    extractor.load_dataset('datasets/drebin215/drebin-215-dataset-5560malware-9476-benign.csv')
    extractor.load_feature_categories('datasets/drebin215/dataset-features-categories.csv')
    extractor.clean_dataset(drop_exact_duplicates=False)

    X, y = extractor.split_features_labels()
    print(f"\nFinal feature matrix: {X.shape}")
    print(f"Labels: {y.shape}")

    print("\nPermission features (sample):", extractor.features_by_category('Manifest Permission')[:5])
    print("API call features (sample):   ", extractor.features_by_category('API call signature')[:5])