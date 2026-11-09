"""
MPRSS Integration Pipeline - Merge two CSV datasets and save the result
No ML training in this pipeline.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

from data_loader import AndroidMalwareDataLoader
from feature_extractor import MPRSSFeatureExtractor
from dataset_integrator import MPRSSDatasetIntegrator


class MPRSSIntegrationPipeline:
    """Pipeline that merges two CSV datasets and saves the result"""

    def __init__(self):
        self.loader     = AndroidMalwareDataLoader()
        self.extractor  = MPRSSFeatureExtractor()
        self.integrator = MPRSSDatasetIntegrator()

    def run(self, dataset_configs, output_path):
        """
        Run the full integration pipeline.

        Parameters
        ----------
        dataset_configs : list of dict
            Each dict must have:
                - 'path'   : str   — path to the CSV file
                - 'weight' : float — repeat factor (default 1.0, optional)
                - 'kwargs' : dict  — extra args for pd.read_csv (optional)

        output_path : str
            Where to save the final merged CSV.
        """
        print("\n" + "=" * 60)
        print("MPRSS - Dataset Integration Pipeline")
        print("=" * 60)

        # Step 1: Load and integrate
        print("\n[1/3] Loading and integrating datasets...")
        X, y = self.integrator.integrate_datasets(dataset_configs)

        if X is None:
            print("Integration failed!")
            return None

        # Step 2: Save merged dataset
        print("\n[2/3] Saving merged dataset...")
        merged_df = self.integrator.save_merged_dataset(output_path)

        # Step 3: Save summary report
        print("\n[3/3] Saving integration report...")
        self.integrator.save_integration_report('logs/integration_report.json')

        print("\n" + "=" * 60)
        print("✅ Integration complete!")
        print(f"📁 Merged CSV : {os.path.abspath(output_path)}")
        print(f"📋 Report     : {os.path.abspath('logs/integration_report.json')}")
        print(f"📊 Total rows : {merged_df.shape[0]}")
        print(f"📊 Total cols : {merged_df.shape[1]}  (features + label)")
        print("=" * 60)

        return merged_df


if __name__ == "__main__":
    pipeline = MPRSSIntegrationPipeline()

    dataset_configs = [
        {'path': 'datasets/Android_Malware.csv',           'weight': 1.0, 'kwargs': {}},
        {'path': 'datasets/CIC-AndMal2017_preprocess.csv', 'weight': 1.0, 'kwargs': {}}
    ]

    pipeline.run(
        dataset_configs=dataset_configs,
        output_path='datasets/merged_dataset.csv'
    )