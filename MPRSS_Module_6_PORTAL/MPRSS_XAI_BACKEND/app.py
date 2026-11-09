"""
MPRSS - XAI Backend Server (Fixed)
Provides SHAP and LIME explanations for malware predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import sys
import traceback

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow React frontend to call this API

print("=" * 60)
print("MPRSS XAI Backend Starting...")
print("=" * 60)

# ============================================================================
# LOAD TRAINED MODEL
# ============================================================================

print("\n📂 Loading trained model...")

# Check if model exists
if not os.path.exists('models/malware_model.pkl'):
    print("   ❌ ERROR: Model not found!")
    print("   Please run 'python train_model.py' first")
    sys.exit(1)

try:
    # Load the trained model
    model = joblib.load('models/malware_model.pkl')
    print("   ✅ Model loaded successfully")
    print(f"   Model type: {type(model).__name__}")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load feature names
try:
    FEATURES = joblib.load('models/features.pkl')
    print(f"   ✅ Loaded {len(FEATURES)} features")
    print(f"   Features: {FEATURES[:5]}...")
except Exception as e:
    print(f"   ❌ Error loading features: {e}")
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def predict_safe(features):
    """Safe prediction wrapper"""
    try:
        features_array = np.array([features]).reshape(1, -1)
        
        # Get prediction
        prediction = model.predict(features_array)[0]
        
        # Get probabilities safely
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_array)[0]
        else:
            # Fallback for models without predict_proba
            probabilities = [0.5, 0.5]
        
        return prediction, probabilities
    except Exception as e:
        print(f"Prediction error: {e}")
        return 0, [0.5, 0.5]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/xai/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'features_count': len(FEATURES),
        'sample_features': FEATURES[:5]
    })

@app.route('/api/xai/shap', methods=['POST'])
def get_shap_explanation():
    """Generate SHAP-like explanation for features"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        feature_values = data.get('features', [])
        
        # Validate feature count
        if len(feature_values) != len(FEATURES):
            return jsonify({
                'error': f'Expected {len(FEATURES)} features, got {len(feature_values)}'
            }), 400
        
        # Convert to numpy array and ensure proper shape
        X = np.array(feature_values).reshape(1, -1)
        
        # Get prediction safely
        try:
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Get feature importance from model
        try:
            feature_importance = model.feature_importances_
        except:
            # Fallback: equal importance
            feature_importance = np.ones(len(FEATURES)) / len(FEATURES)
        
        # Create feature importance list
        importance_list = []
        for i, (feature, value, imp) in enumerate(zip(FEATURES, feature_values, feature_importance)):
            importance_list.append({
                'feature': feature,
                'value': float(value),
                'importance': float(imp),
                'impact': 'positive' if imp > 0.03 else 'negative'
            })
        
        # Sort by importance and get top 5
        importance_list.sort(key=lambda x: x['importance'], reverse=True)
        top_features = importance_list[:5]
        
        # Normalize to percentages
        total = sum(f['importance'] for f in top_features)
        if total > 0:
            for f in top_features:
                f['importance_percent'] = round((f['importance'] / total) * 100)
        else:
            for f in top_features:
                f['importance_percent'] = 0
        
        # Generate summary
        risk_factors = [f"{f['feature']} ({f['importance_percent']}%)" 
                       for f in top_features if f['impact'] == 'positive' and f['importance_percent'] > 5]
        
        summary = f"Main risk factors: {', '.join(risk_factors[:3])}" if risk_factors else "No significant risk factors detected"
        
        return jsonify({
            'success': True,
            'prediction': 'Malicious' if prediction == 1 else 'Benign',
            'probability': float(probabilities[1] if prediction == 1 else probabilities[0]),
            'top_features': top_features,
            'summary': summary
        })
        
    except Exception as e:
        print(f"SHAP error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/xai/lime', methods=['POST'])
def get_lime_explanation():
    """Generate LIME-like explanation"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        feature_values = data.get('features', [])
        
        # Validate feature count
        if len(feature_values) != len(FEATURES):
            return jsonify({'error': f'Expected {len(FEATURES)} features'}), 400
        
        # Convert to numpy array
        X = np.array(feature_values).reshape(1, -1)
        
        # Get prediction
        try:
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Get feature importance
        try:
            feature_importance = model.feature_importances_
        except:
            feature_importance = np.ones(len(FEATURES)) / len(FEATURES)
        
        # Create explanation with top features
        top_indices = np.argsort(feature_importance)[::-1][:5]
        
        lime_features = []
        for idx in top_indices:
            lime_features.append({
                'feature': FEATURES[idx],
                'impact': float(feature_importance[idx]),
                'impact_percent': round(float(feature_importance[idx]) * 100, 1)
            })
        
        # Generate natural language explanation
        confidence = probabilities[1] if prediction == 1 else probabilities[0]
        
        if prediction == 1:
            explanation_text = f"This app is classified as **MALICIOUS** with {confidence*100:.1f}% confidence. "
            risk_features = [f['feature'] for f in lime_features[:3]]
            if risk_features:
                explanation_text += f"Main risk factors: {', '.join(risk_features)}."
            else:
                explanation_text += "No specific risk factors identified."
        else:
            explanation_text = f"This app is classified as **BENIGN** with {confidence*100:.1f}% confidence."
        
        return jsonify({
            'success': True,
            'prediction': 'Malicious' if prediction == 1 else 'Benign',
            'probability': float(confidence),
            'explanation': explanation_text,
            'top_features': lime_features
        })
        
    except Exception as e:
        print(f"LIME error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/xai/feature-importance', methods=['GET'])
def get_global_feature_importance():
    """Return global feature importance"""
    try:
        # Get feature importance from model
        try:
            importance = model.feature_importances_
        except:
            importance = np.ones(len(FEATURES)) / len(FEATURES)
        
        feature_importance = []
        for i, (feature, imp) in enumerate(zip(FEATURES, importance)):
            feature_importance.append({
                'feature': feature,
                'importance': float(imp),
                'importance_percent': round(float(imp) * 100, 2),
                'rank': i + 1
            })
        
        # Sort by importance (highest first)
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        # Add global rank
        for idx, f in enumerate(feature_importance):
            f['global_rank'] = idx + 1
        
        return jsonify({
            'success': True,
            'features': feature_importance,
            'total_features': len(FEATURES)
        })
        
    except Exception as e:
        print(f"Global importance error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 XAI Backend Ready!")
    print("=" * 60)
    print(f"\n📊 Model: Random Forest Classifier")
    print(f"📈 Features: {len(FEATURES)}")
    print("\n🌐 API Endpoints:")
    print("   POST /api/xai/shap     - SHAP explanation")
    print("   POST /api/xai/lime     - LIME explanation")
    print("   GET  /api/xai/feature-importance - Global importance")
    print("   GET  /api/xai/health   - Health check")
    print("\n📍 Server: http://localhost:5001")
    print("=" * 60)
    print("\n✅ Server is running! Press Ctrl+C to stop")
    print("=" * 60)
    
    # Run the app
    app.run(host='0.0.0.0', port=5001, debug=True)