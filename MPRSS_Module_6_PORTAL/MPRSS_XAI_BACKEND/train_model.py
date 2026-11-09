"""
MPRSS - Model Training Script
Trains Random Forest model for malware detection
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("MPRSS - Model Training")
print("=" * 60)

# ============================================================================
# STEP 1: Create or Load Dataset
# ============================================================================

print("\n📊 Preparing dataset...")

# Create sample training data (since you don't have real dataset yet)
np.random.seed(42)
n_samples = 2000

# Define features (permissions and behaviors)
features = [
    'READ_SMS', 'SEND_SMS', 'RECEIVE_SMS', 'READ_CONTACTS',
    'ACCESS_FINE_LOCATION', 'CAMERA', 'RECORD_AUDIO', 'WRITE_EXTERNAL_STORAGE',
    'INTERNET', 'SUSPICIOUS_API_CALLS', 'OBFUSCATED_CODE', 'NETWORK_ACTIVITY',
    'FILE_ACCESS', 'CRYPTO_OPERATIONS', 'ROOT_DETECTION'
]

print(f"   Features: {len(features)}")
print(f"   Samples: {n_samples}")

# Generate random feature values (0-100 scale)
X = np.random.randint(0, 100, size=(n_samples, len(features)))

# Generate labels (0 = Benign, 1 = Malicious)
y = np.zeros(n_samples)

# Malicious apps have higher scores on dangerous features
dangerous_indices = [0, 1, 2, 3, 4, 5, 6, 9, 10, 13, 14]

for i in range(n_samples):
    score = 0
    for idx in dangerous_indices:
        if idx < len(features):
            score += X[i, idx] * 2
    if score > 500:
        y[i] = 1

print(f"   Benign samples: {np.sum(y == 0)}")
print(f"   Malicious samples: {np.sum(y == 1)}")

# ============================================================================
# STEP 2: Split Data
# ============================================================================

print("\n🔀 Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# ============================================================================
# STEP 3: Train Model
# ============================================================================

print("\n🌲 Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# ============================================================================
# STEP 4: Evaluate Model
# ============================================================================

print("\n📈 Evaluating model...")

train_accuracy = model.score(X_train, y_train) * 100
test_accuracy = model.score(X_test, y_test) * 100

print(f"   Training accuracy: {train_accuracy:.2f}%")
print(f"   Testing accuracy: {test_accuracy:.2f}%")

# Get feature importance
importance = model.feature_importances_
print(f"\n🏆 Top 10 Most Important Features:")

# Sort features by importance
sorted_indices = np.argsort(importance)[::-1]
for i in range(min(10, len(features))):
    idx = sorted_indices[i]
    print(f"   {i+1:2d}. {features[idx]}: {importance[idx]*100:.1f}%")

# ============================================================================
# STEP 5: Save Model
# ============================================================================

print("\n💾 Saving model...")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/malware_model.pkl')
print(f"   ✅ Model saved to: models/malware_model.pkl")

# Save features
joblib.dump(features, 'models/features.pkl')
print(f"   ✅ Features saved to: models/features.pkl")

print("\n" + "=" * 60)
print("✅ Training Complete!")
print("=" * 60)
print("\n🚀 Next: Run 'python app.py' to start the XAI backend")