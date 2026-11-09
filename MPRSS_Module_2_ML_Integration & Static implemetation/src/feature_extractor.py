"""
MPRSS Feature Extractor - Extract and clean features from CSV datasets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class MPRSSFeatureExtractor:
    """Extract and clean features for dataset integration"""

    def __init__(self):
        self.label_encoders = {}

    def handle_missing_values(self, df, strategy='zero'):
        """Handle missing values in dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if strategy == 'zero':
            df[numeric_cols] = df[numeric_cols].fillna(0)
        elif strategy == 'mean':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == 'median':
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        return df

    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical (text) feature columns to numeric"""
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(df[col].astype(str))
            df[col] = self.label_encoders[col].transform(df[col].astype(str))

        return df