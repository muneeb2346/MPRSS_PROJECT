"""
MPRSS Dataset Integrator - Merge two CSV datasets and save to disk
"""

import pandas as pd
import os
import json
from datetime import datetime

from data_loader import AndroidMalwareDataLoader
from feature_extractor import MPRSSFeatureExtractor


class MPRSSDatasetIntegrator:
    """Integrate two Android malware CSV datasets and save the merged result"""

    def __init__(self):
        self.loader     = AndroidMalwareDataLoader()
        self.extractor  = MPRSSFeatureExtractor()
        self.combined_X = None
        self.combined_y = None

    # ------------------------------------------------------------------
    def load_and_preprocess_dataset(self, filepath, **kwargs):
        """Load and preprocess a single CSV dataset"""
        print(f"\n{'='*50}")
        print(f"Loading dataset: {filepath}")
        print(f"{'='*50}")

        df, label_col = self.loader.load_csv_dataset(filepath, **kwargs)

        if df is None:
            return None, None

        # Handle missing values
        df = self.extractor.handle_missing_values(df)

        # Separate features and labels
        # Use 'labeled' column directly — it already contains 1/0
        if 'labeled' in df.columns:
            y = df['labeled'].copy()
            drop_cols = ['labeled', 'label'] if 'label' in df.columns else ['labeled']
            X = df.drop(columns=drop_cols)
        elif label_col in df.columns:
            y = df[label_col].copy()
            X = df.drop(columns=[label_col])
            y = y.apply(
                lambda x: 1 if str(x).lower() in
                ['malware', 'malicious', '1', 'yes', 'true', 'bad'] else 0
            )
        else:
            print(f"Label column not found in {filepath}")
            return None, None

        # Encode any remaining categorical feature columns
        X = self.extractor.encode_categorical_features(X)

        print(f"Features shape:  {X.shape}")
        print(f"Malware samples: {int(sum(y))}")
        print(f"Benign samples:  {int(len(y) - sum(y))}")

        return X, y

    # ------------------------------------------------------------------
    def integrate_datasets(self, dataset_configs):
        """
        Merge two CSV datasets — keeps only common feature columns.

        dataset_configs: list of 2 dicts, each with:
            - 'path'   : str   — path to the CSV file
            - 'weight' : float — repeat factor (default 1.0, optional)
            - 'kwargs' : dict  — extra args passed to pd.read_csv (optional)
        """
        all_features    = []
        all_labels      = []
        dataset_weights = []

        for config in dataset_configs:
            X, y = self.load_and_preprocess_dataset(
                config['path'],
                **config.get('kwargs', {})
            )

            if X is not None and y is not None:
                all_features.append(X)
                all_labels.append(y)
                dataset_weights.append(config.get('weight', 1.0))

        if not all_features:
            print("No datasets loaded successfully!")
            return None, None

        # Find common columns across both datasets
        common_cols = set(all_features[0].columns)
        for X in all_features[1:]:
            common_cols = common_cols.intersection(set(X.columns))

        common_cols = list(common_cols)
        print(f"\nCommon features across datasets: {len(common_cols)}")

        # Align each dataset to common columns and apply weight if set
        aligned_features = []
        for i, X in enumerate(all_features):
            X_aligned = X[common_cols].copy()

            if dataset_weights[i] != 1.0:
                weight = dataset_weights[i]
                if weight > 1:
                    n_copies      = int(weight)
                    X_aligned     = pd.concat([X_aligned]     * n_copies, ignore_index=True)
                    all_labels[i] = pd.concat([all_labels[i]] * n_copies, ignore_index=True)

            aligned_features.append(X_aligned)

        # Combine into one DataFrame
        self.combined_X = pd.concat(aligned_features, ignore_index=True)
        self.combined_y = pd.concat(all_labels,       ignore_index=True)

        total = len(self.combined_y)
        mal   = int(sum(self.combined_y))

        print(f"\n{'='*50}")
        print(f"Combined Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total samples:   {total}")
        print(f"Total features:  {self.combined_X.shape[1]}")
        print(f"Malware samples: {mal} ({mal/total*100:.1f}%)")
        print(f"Benign samples:  {total - mal} ({(total-mal)/total*100:.1f}%)")

        return self.combined_X, self.combined_y

    # ------------------------------------------------------------------
    def save_merged_dataset(self, output_path):
        """
        Save the merged dataset (features + label column) to a CSV file.

        Parameters
        ----------
        output_path : str
            Full path where the merged CSV will be written.
            Examples:
                'datasets/merged_dataset.csv'
                'C:/Users/numl-/Desktop/MPRSS/merged_dataset.csv'
                'C:/Users/numl-/OneDrive/Desktop/FYP/Datasets/Merge Dataset/merged_dataset.csv'
        """
        if self.combined_X is None or self.combined_y is None:
            print("No merged dataset found. Run integrate_datasets() first.")
            return None

        # Create output directory if it does not exist
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Attach label column and save
        merged_df = self.combined_X.copy()
        merged_df['label'] = self.combined_y.values
        merged_df.to_csv(output_path, index=False)

        print(f"\n✅ Merged dataset saved to: {os.path.abspath(output_path)}")
        print(f"   Rows   : {merged_df.shape[0]}")
        print(f"   Columns: {merged_df.shape[1]}  (features + label)")

        return merged_df

    # ------------------------------------------------------------------
    def save_integration_report(self, output_path='logs/integration_report.json'):
        """Save a JSON summary report of the merged dataset"""
        if self.combined_X is None or self.combined_y is None:
            print("No merged dataset found. Run integrate_datasets() first.")
            return

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        total = len(self.combined_y)
        mal   = int(sum(self.combined_y))

        report = {
            'integration_date': datetime.now().isoformat(),
            'total_samples':    total,
            'total_features':   self.combined_X.shape[1],
            'malware_count':    mal,
            'benign_count':     total - mal,
            'malware_pct':      round(mal / total * 100, 2),
            'feature_names':    list(self.combined_X.columns)
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Integration report saved to {output_path}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    integrator = MPRSSDatasetIntegrator()

    configs = [
        {'path': 'datasets/Android_Malware.csv',           'weight': 1.0},
        {'path': 'datasets/CIC-AndMal2017_preprocess.csv', 'weight': 1.0}
    ]

    X, y = integrator.integrate_datasets(configs)

    if X is not None:
        integrator.save_merged_dataset(
            'C:/Users/numl-/OneDrive/Desktop/FYP/Datasets/Merge Dataset/merged_dataset.csv'
        )
        integrator.save_integration_report('logs/integration_report.json')