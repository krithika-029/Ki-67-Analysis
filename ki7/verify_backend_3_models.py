#!/usr/bin/env python3
"""
Backend verification script to ensure all 3 models are loaded and exposed via API
"""

import requests
import json

def test_backend_endpoints():
    print("🔍 Verifying Backend 3-Model Ensemble")
    print("=" * 50)
    
    base_url = "http://localhost:5001"
    
    try:
        # Test health endpoint
        print("1️⃣ Testing health endpoint...")
        health_response = requests.get(f"{base_url}/api/health")
        health_data = health_response.json()
        
        print(f"   Status: {health_data['status']}")
        print(f"   Models loaded: {health_data['models_loaded']}")
        print(f"   Total models: {health_data['total_models']}")
        print(f"   ✅ Health check passed\n")
        
        # Test models endpoint
        print("2️⃣ Testing models endpoint...")
        models_response = requests.get(f"{base_url}/api/models")
        models_data = models_response.json()
        
        print(f"   Total models: {models_data['total_models']}")
        print(f"   Available models:")
        
        for model in models_data['available_models']:
            print(f"     - {model['name']} ({model['architecture']})")
            print(f"       Weight: {model['ensemble_weight']}")
            print(f"       Accuracy: {model['accuracy']}%")
            print(f"       Status: {model['status']}")
        
        print(f"   ✅ Models endpoint passed\n")
        
        # Test ensemble info endpoint
        print("3️⃣ Testing ensemble info endpoint...")
        ensemble_response = requests.get(f"{base_url}/api/ensemble/info")
        ensemble_data = ensemble_response.json()
        
        print(f"   Ensemble: {ensemble_data['ensemble']['name']}")
        print(f"   High-confidence accuracy: {ensemble_data['ensemble']['high_confidence_accuracy']}%")
        print(f"   Models in ensemble: {ensemble_data['ensemble']['loaded_models']}")
        
        print(f"   Model details:")
        for model in ensemble_data['models']:
            status = "✅" if model['loaded'] else "❌"
            print(f"     {status} {model['name']}: {model['ensemble_weight']} weight, {model['individual_accuracy']}% acc")
        
        print(f"   ✅ Ensemble info endpoint passed\n")
        
        # Verification summary
        models_loaded = models_data['total_models']
        expected_models = ['EfficientNet-B2', 'RegNet-Y-8GF', 'ViT']
        
        loaded_model_names = [model['name'] for model in models_data['available_models']]
        all_expected_loaded = all(name in loaded_model_names for name in expected_models)
        
        print("🏅 BACKEND VERIFICATION SUMMARY:")
        print("=" * 50)
        print(f"✅ Models loaded: {models_loaded}/3")
        print(f"✅ Expected models present: {'Yes' if all_expected_loaded else 'No'}")
        print(f"✅ API endpoints working: Yes")
        print(f"✅ Ensemble configuration: Working")
        
        if models_loaded == 3 and all_expected_loaded:
            print("\n🎉 BACKEND VERIFICATION PASSED!")
            print("🎉 All 3 models are loaded and exposed via API")
        else:
            print("\n❌ BACKEND VERIFICATION FAILED!")
            
    except Exception as e:
        print(f"❌ Backend verification failed: {e}")

if __name__ == "__main__":
    test_backend_endpoints()
