#!/usr/bin/env python
"""
MPRSS - Dataset Integration Runner
Run this file to merge your two CSV datasets and save the result.

Usage:
    python run.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     MPRSS - Malware Prediction and Risk Scoring System   ║
    ║                                                          ║
    ║     Dataset Integration  (No ML Training)                ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    from src.main_integration_pipeline import MPRSSIntegrationPipeline

    # Create required folders
    for folder in ('datasets', 'logs'):
        os.makedirs(folder, exist_ok=True)

    # ──────────────────────────────────────────────────────────
    # STEP 1 — Set your two dataset paths here
    # ──────────────────────────────────────────────────────────
    dataset_configs = [
        {
            'path': 'datasets/Android_Malware.csv',
            'weight': 1.0,
            'kwargs': {}
        },
        {
            'path': 'datasets/CIC-AndMal2017_preprocess.csv',
            'weight': 1.0,
            'kwargs': {}
        },
    ]

    # ──────────────────────────────────────────────────────────
    # STEP 2 — Set where you want the merged CSV saved
    #
    # Examples:
    #   'datasets/merged_dataset.csv'
    #   'C:/Users/numl-/Desktop/merged_dataset.csv'
    #   'D:/MPRSS_Project/data/merged_dataset.csv'
    # ──────────────────────────────────────────────────────────
    OUTPUT_PATH = 'C:/Users/numl-/OneDrive/Desktop/FYP/Datasets/Merge Dataset/merged_dataset.csv'

    # Check datasets exist before running
    missing = [c['path'] for c in dataset_configs if not os.path.exists(c['path'])]
    if missing:
        print("❌ The following dataset files were not found:")
        for path in missing:
            print(f"   - {os.path.abspath(path)}")
        print("\n   Please place your CSV files in the datasets/ folder and try again.")
        sys.exit(1)

    # Run pipeline
    pipeline = MPRSSIntegrationPipeline()
    merged_df = pipeline.run(
        dataset_configs=dataset_configs,
        output_path=OUTPUT_PATH
    )

    if merged_df is not None:
        print("\n🎉 Success! Your merged dataset is ready.")
        print("\nFiles generated:")
        print(f"  - {OUTPUT_PATH}")
        print(f"  - logs/integration_report.json")
    else:
        print("\n❌ Integration failed. Please check the errors above.")


if __name__ == "__main__":
    main()