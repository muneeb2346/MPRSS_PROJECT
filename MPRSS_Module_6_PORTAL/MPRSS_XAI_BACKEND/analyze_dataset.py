"""
MPRSS - Dataset Analysis Script
Run this first to understand your malware dataset structure
"""

import pandas as pd
import os
import sys

print("=" * 60)
print("MPRSS - Dataset Analysis Tool")
print("=" * 60)

# ============================================================================
# STEP 1: Ask for dataset path
# ============================================================================

print("\n📁 Please enter the path to your malware dataset file")
print("   Supported formats: .csv, .json, .arff")
print("   Example: C:/Users/YourName/Downloads/malware_dataset.csv")
print("   Or press ENTER to use a sample dataset")

dataset_path = input("\n📂 Dataset path: ").strip()

# If user pressed Enter, use sample data
if not dataset_path:
    print("\n⚠️ No dataset provided. Creating sample dataset for testing...")
    
    # Create sample data
    import numpy as np
    np.random.seed(42)
    
    n_samples = 100
    
    # Features
    features = {
        'READ_SMS': np.random.randint(0, 100, n_samples),
        'SEND_SMS': np.random.randint(0, 100, n_samples),
        'RECEIVE_SMS': np.random.randint(0, 100, n_samples),
        'READ_CONTACTS': np.random.randint(0, 100, n_samples),
        'ACCESS_FINE_LOCATION': np.random.randint(0, 100, n_samples),
        'CAMERA': np.random.randint(0, 100, n_samples),
        'RECORD_AUDIO': np.random.randint(0, 100, n_samples),
        'WRITE_EXTERNAL_STORAGE': np.random.randint(0, 100, n_samples),
        'INTERNET': np.random.randint(0, 100, n_samples),
        'SUSPICIOUS_API_CALLS': np.random.randint(0, 100, n_samples),
        'OBFUSCATED_CODE': np.random.randint(0, 100, n_samples),
        'NETWORK_ACTIVITY': np.random.randint(0, 100, n_samples),
        'FILE_ACCESS': np.random.randint(0, 100, n_samples),
        'CRYPTO_OPERATIONS': np.random.randint(0, 100, n_samples),
        'ROOT_DETECTION': np.random.randint(0, 100, n_samples)
    }
    
    df = pd.DataFrame(features)
    
    # Add labels (0 = Benign, 1 = Malicious)
    # Malicious apps have higher scores on dangerous features
    dangerous_features = ['READ_SMS', 'SEND_SMS', 'RECEIVE_SMS', 'READ_CONTACTS', 
                          'SUSPICIOUS_API_CALLS', 'OBFUSCATED_CODE', 'CRYPTO_OPERATIONS']
    
    df['label'] = 0
    for i in range(n_samples):
        score = 0
        for f in dangerous_features:
            score += df.loc[i, f]
        if score > 300:  # Threshold for malicious
            df.loc[i, 'label'] = 1
    
    print(f"✅ Created sample dataset with {n_samples} samples")
    print(f"   Benign: {len(df[df['label']==0])}")
    print(f"   Malicious: {len(df[df['label']==1])}")
    
else:
    # ============================================================================
    # STEP 2: Load the dataset
    # ============================================================================
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        print(f"\n❌ File not found: {dataset_path}")
        print("   Please check the path and try again.")
        sys.exit(1)
    
    print(f"\n📊 Loading dataset from: {dataset_path}")
    
    # Try different formats
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
        print("✅ Loaded as CSV file")
    elif dataset_path.endswith('.json'):
        df = pd.read_json(dataset_path)
        print("✅ Loaded as JSON file")
    elif dataset_path.endswith('.arff'):
        from scipy.io import arff
        data = arff.loadarff(dataset_path)
        df = pd.DataFrame(data[0])
        print("✅ Loaded as ARFF file")
    else:
        print("❌ Unsupported format. Please use CSV, JSON, or ARFF.")
        sys.exit(1)

# ============================================================================
# STEP 3: Display dataset information
# ============================================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(f"\n📈 Shape: {df.shape[0]} rows, {df.shape[1]} columns")

print(f"\n📋 Column Names (first 20):")
for i, col in enumerate(df.columns[:20]):
    print(f"   {i+1:3d}. {col}")

if df.shape[1] > 20:
    print(f"   ... and {df.shape[1] - 20} more columns")

# ============================================================================
# STEP 4: Identify label column
# ============================================================================

print("\n" + "=" * 60)
print("LABEL COLUMN DETECTION")
print("=" * 60)

# Common label column names
possible_labels = ['label', 'class', 'type', 'malicious', 'benign', 
                   'Category', 'family', 'is_malicious', 'Label', 'malware']

label_column = None
for col in possible_labels:
    if col in df.columns:
        label_column = col
        break

# Also check case-insensitive
if not label_column:
    for col in df.columns:
        if col.lower() in [name.lower() for name in possible_labels]:
            label_column = col
            break

if label_column:
    print(f"\n✅ Found label column: '{label_column}'")
    unique_values = df[label_column].unique()
    print(f"   Unique values: {unique_values}")
    
    # Count distribution
    value_counts = df[label_column].value_counts()
    print(f"\n   Distribution:")
    for val, count in value_counts.items():
        percentage = (count / len(df)) * 100
        print(f"      {val}: {count} ({percentage:.1f}%)")
else:
    print("\n⚠️ Could not automatically detect label column.")
    print("   Please specify which column contains malware/benign labels:")
    for i, col in enumerate(df.columns):
        print(f"   {i}: {col}")
    
    try:
        label_idx = int(input("\nEnter column number: "))
        label_column = df.columns[label_idx]
        print(f"\n✅ Selected label column: '{label_column}'")
    except:
        print("   Using 'label' as default")
        label_column = 'label'

# ============================================================================
# STEP 5: Feature analysis
# ============================================================================

print("\n" + "=" * 60)
print("FEATURE ANALYSIS")
print("=" * 60)

# Separate features and labels
feature_columns = [col for col in df.columns if col != label_column]

# Check data types
numeric_cols = df[feature_columns].select_dtypes(include=['number']).columns.tolist()
categorical_cols = df[feature_columns].select_dtypes(include=['object']).columns.tolist()

print(f"\n📊 Total Features: {len(feature_columns)}")
print(f"   Numeric features: {len(numeric_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Show sample of numeric features
if len(numeric_cols) > 0:
    print(f"\n📈 Sample Numeric Features (first 10):")
    for col in numeric_cols[:10]:
        print(f"   - {col}")
        print(f"     Range: {df[col].min():.2f} to {df[col].max():.2f}")
        print(f"     Mean: {df[col].mean():.2f}")

# Show sample of categorical features
if len(categorical_cols) > 0:
    print(f"\n🏷️ Sample Categorical Features (first 10):")
    for col in categorical_cols[:10]:
        unique_count = df[col].nunique()
        print(f"   - {col}: {unique_count} unique values")
        if unique_count <= 10:
            print(f"     Values: {df[col].unique()}")

# ============================================================================
# STEP 6: Check for missing values
# ============================================================================

print("\n" + "=" * 60)
print("MISSING VALUES CHECK")
print("=" * 60)

missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

if len(missing_cols) > 0:
    print(f"\n⚠️ Found {len(missing_cols)} columns with missing values:")
    for col, count in missing_cols.head(10).items():
        percentage = (count / len(df)) * 100
        print(f"   - {col}: {count} ({percentage:.1f}%)")
else:
    print("\n✅ No missing values found!")

# ============================================================================
# STEP 7: Save feature information
# ============================================================================

print("\n" + "=" * 60)
print("SAVING FEATURE INFORMATION")
print("=" * 60)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save feature list
feature_list = feature_columns
with open('models/feature_names.txt', 'w') as f:
    for feature in feature_list:
        f.write(f"{feature}\n")

print(f"\n💾 Feature list saved to: models/feature_names.txt")
print(f"   Total features: {len(feature_list)}")

# Save dataset summary
summary = {
    'total_samples': len(df),
    'total_features': len(feature_columns),
    'numeric_features': len(numeric_cols),
    'categorical_features': len(categorical_cols),
    'label_column': label_column
}

import json
with open('models/dataset_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"💾 Dataset summary saved to: models/dataset_summary.json")

# ============================================================================
# STEP 8: Summary
# ============================================================================

print("\n" + "=" * 60)
print("✅ DATASET ANALYSIS COMPLETE!")
print("=" * 60)

print("\n📋 Summary:")
print(f"   Total samples: {len(df)}")
print(f"   Total features: {len(feature_columns)}")
print(f"   Label column: {label_column}")

if label_column in df.columns:
    print(f"\n   Class distribution:")
    for val, count in df[label_column].value_counts().items():
        print(f"      {val}: {count} ({count/len(df)*100:.1f}%)")

print("\n🚀 Next steps:")
print("   1. Run: python train_model_real.py")
print("   2. Then: python app.py")
print("=" * 60)