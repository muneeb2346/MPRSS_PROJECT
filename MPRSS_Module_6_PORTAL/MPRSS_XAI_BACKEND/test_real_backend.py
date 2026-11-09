"""
MPRSS - XAI Backend Test Script
Tests all API endpoints to ensure they're working
"""

import requests
import json
import time

# API base URL
API_URL = 'http://localhost:5001/api/xai'

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def test_health():
    """Test the health check endpoint"""
    print_section("TEST 1: Health Check")
    
    try:
        response = requests.get(f'{API_URL}/health', timeout=5)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Model Loaded: {data.get('model_loaded', False)}")
            print(f"   Features Count: {data.get('features_count', 0)}")
            print(f"   Sample Features: {data.get('sample_features', [])[:3]}")
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend!")
        print("   Make sure the backend is running: python app.py")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_shap():
    """Test SHAP explanation endpoint"""
    print_section("TEST 2: SHAP Explanation")
    
    # Create test features (all medium values)
    test_features = [50] * 15
    # Make it slightly malicious by increasing dangerous permissions
    test_features[0] = 85   # READ_SMS - high
    test_features[1] = 90   # SEND_SMS - high
    test_features[9] = 80   # SUSPICIOUS_API_CALLS - high
    
    try:
        response = requests.post(
            f'{API_URL}/shap',
            json={'features': test_features},
            timeout=10
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   Prediction: {data.get('prediction', 'unknown')}")
                print(f"   Probability: {data.get('probability', 0):.2%}")
                print(f"   Summary: {data.get('summary', 'N/A')}")
                print("\n   Top 5 Features:")
                for i, f in enumerate(data.get('top_features', [])[:5], 1):
                    print(f"      {i}. {f.get('feature', 'unknown')}: {f.get('importance_percent', 0)}% ({f.get('impact', 'unknown')})")
                return True
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_lime():
    """Test LIME explanation endpoint"""
    print_section("TEST 3: LIME Explanation")
    
    # Create test features (benign app)
    test_features = [50] * 15
    test_features[0] = 10   # READ_SMS - low
    test_features[1] = 5    # SEND_SMS - low
    test_features[9] = 5    # SUSPICIOUS_API_CALLS - low
    
    try:
        response = requests.post(
            f'{API_URL}/lime',
            json={'features': test_features},
            timeout=10
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   Prediction: {data.get('prediction', 'unknown')}")
                print(f"   Probability: {data.get('probability', 0):.2%}")
                print(f"   Explanation: {data.get('explanation', 'N/A')}")
                return True
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_global_importance():
    """Test global feature importance endpoint"""
    print_section("TEST 4: Global Feature Importance")
    
    try:
        response = requests.get(f'{API_URL}/feature-importance', timeout=10)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"   Total Features: {data.get('total_features', 0)}")
                print("\n   Top 10 Features (Global):")
                for i, f in enumerate(data.get('features', [])[:10], 1):
                    print(f"      {i:2d}. {f.get('feature', 'unknown')}: {f.get('importance_percent', 0)}%")
                return True
            else:
                print(f"❌ Error: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid input"""
    print_section("TEST 5: Error Handling")
    
    # Send wrong number of features (10 instead of 15)
    invalid_features = [50] * 10
    
    try:
        response = requests.post(
            f'{API_URL}/shap',
            json={'features': invalid_features},
            timeout=10
        )
        
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 400:
            data = response.json()
            print(f"   Error Message: {data.get('error', 'Unknown error')}")
            print("   ✅ Proper error handling works!")
            return True
        else:
            print(f"⚠️ Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  MPRSS XAI Backend Test Suite")
    print("=" * 60)
    print("\n⚠️ Make sure the backend is running: python app.py")
    print("   The backend should be on http://localhost:5001")
    
    # Wait for user confirmation
    input("\nPress Enter to start tests...")
    
    # Run all tests
    results = []
    
    results.append(("Health Check", test_health()))
    
    # Only run other tests if health check passes
    if results[0][1]:
        results.append(("SHAP Explanation", test_shap()))
        results.append(("LIME Explanation", test_lime()))
        results.append(("Global Feature Importance", test_global_importance()))
        results.append(("Error Handling", test_error_handling()))
    else:
        print("\n❌ Health check failed. Please start the backend first.")
        print("   Run: python app.py")
    
    # Summary
    print("\n" + "=" * 60)
    print("  TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        if result:
            passed += 1
        else:
            failed += 1
        print(f"   {status}: {name}")
    
    print("-" * 60)
    print(f"   Total: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All tests passed! Backend is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the backend logs for details.")
    
    print("=" * 60)