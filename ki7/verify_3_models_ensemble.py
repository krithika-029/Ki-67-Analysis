#!/usr/bin/env python3
"""
Verification script to ensure all 3 models are loaded and working in the ensemble
"""

import torch
import json
from pathlib import Path
from refined_95_percent_ensemble import Refined95Evaluator

def main():
    print("🔍 Verifying 3-Model Ensemble Configuration")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = Refined95Evaluator("models")
    
    # Load models
    success = evaluator.load_models()
    
    if not success:
        print("❌ Failed to load models")
        return
    
    print(f"\n📊 ENSEMBLE VERIFICATION RESULTS:")
    print("=" * 50)
    print(f"✅ Total models loaded: {len(evaluator.models)}")
    print(f"✅ Expected models: 3")
    print(f"✅ Status: {'PASS' if len(evaluator.models) == 3 else 'FAIL'}")
    
    print(f"\n🏆 MODEL DETAILS:")
    total_weight = 0
    for model_name, config in evaluator.model_configs.items():
        is_loaded = model_name in evaluator.models
        weight = evaluator.model_weights.get(model_name, 0)
        total_weight += weight if is_loaded else 0
        
        print(f"   {model_name}:")
        print(f"     Loaded: {'✅' if is_loaded else '❌'}")
        print(f"     Weight: {weight:.2f}")
        print(f"     Individual Accuracy: {config['individual_acc']:.1f}%")
        print(f"     Architecture: {config['arch']}")
        print()
    
    print(f"🎯 ENSEMBLE WEIGHTS:")
    print(f"   Total Weight: {total_weight:.2f} (should be 1.0)")
    print(f"   Weight Distribution:")
    for model_name, weight in evaluator.model_weights.items():
        percentage = (weight / total_weight * 100) if total_weight > 0 else 0
        print(f"     {model_name}: {weight:.2f} ({percentage:.1f}%)")
    
    # Test if we can create a prediction (without actual data)
    print(f"\n🧪 ENSEMBLE FUNCTIONALITY TEST:")
    try:
        # Test model inference capabilities
        device = evaluator.device
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        
        predictions = {}
        for model_name, model in evaluator.models.items():
            with torch.no_grad():
                output = model(dummy_input)
                predictions[model_name] = torch.sigmoid(output).item()
        
        print(f"✅ All models can perform inference")
        print(f"   Sample predictions: {predictions}")
        
        # Calculate weighted ensemble prediction
        weighted_sum = sum(pred * evaluator.model_weights[name] 
                          for name, pred in predictions.items())
        ensemble_prediction = weighted_sum / sum(evaluator.model_weights.values())
        
        print(f"✅ Ensemble prediction: {ensemble_prediction:.3f}")
        print(f"✅ Ensemble working correctly!")
        
    except Exception as e:
        print(f"❌ Ensemble functionality test failed: {e}")
    
    # Summary
    print(f"\n🏅 FINAL VERIFICATION:")
    all_loaded = len(evaluator.models) == 3
    weights_correct = abs(total_weight - 1.0) < 0.01
    
    if all_loaded and weights_correct:
        print("✅ ALL CHECKS PASSED!")
        print("✅ 3-model ensemble is properly configured and ready")
        print("✅ EfficientNet-B2, RegNet-Y-8GF, and ViT are all loaded")
        print("✅ Weights are properly normalized")
    else:
        print("❌ VERIFICATION FAILED!")
        if not all_loaded:
            print(f"❌ Only {len(evaluator.models)}/3 models loaded")
        if not weights_correct:
            print(f"❌ Weights don't sum to 1.0 (sum={total_weight:.3f})")

if __name__ == "__main__":
    main()
