#!/usr/bin/env python3
"""
Simple Improved Confidence Calculation for Ki-67 Ensemble
=========================================================

This script provides a simple improvement to the confidence calculation 
by using multiple methods and avoiding the overconfidence issue.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Import timm for model architectures
try:
    import timm
    print(f"✅ timm version: {timm.__version__}")
except ImportError:
    print("❌ timm not found. Installing...")
    os.system("pip install timm")
    import timm

class ImprovedKi67Ensemble:
    """
    Ki-67 ensemble with improved confidence calculation
    """
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.transform = self._create_transform()
        
        # Model configurations (matching the refined model manager)
        self.model_configs = {
            'EfficientNet-B2': {
                'file': 'Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth',
                'arch': 'efficientnet_b2',
                'weight': 0.70,
                'individual_acc': 92.5
            },
            'RegNet-Y-8GF': {
                'file': 'Ki67_Advanced_RegNet-Y-8GF_best_model_20250619_111223.pth', 
                'arch': 'regnety_008',
                'weight': 0.20,
                'individual_acc': 89.3
            },
            'ViT': {
                'file': 'Ki67_ViT_best_model_20250619_071454.pth',
                'arch': 'vit_base_patch16_224',
                'weight': 0.10,
                'individual_acc': 87.8
            }
        }
        
        print(f"🚀 Initializing Improved Ki-67 Ensemble")
        self.load_models()
        
    def _create_transform(self):
        """Create image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self):
        """Load all ensemble models"""
        print("🔄 Loading ensemble models...")
        
        for model_name, config in self.model_configs.items():
            model_path = self.models_dir / config['file']
            
            if not model_path.exists():
                print(f"❌ Model file not found: {model_path}")
                continue
            
            try:
                # Create model architecture with correct number of classes
                if config['arch'].startswith('efficientnet'):
                    model = timm.create_model(config['arch'], pretrained=False, num_classes=1)
                elif config['arch'].startswith('regnety'):
                    model = timm.create_model(config['arch'], pretrained=False, num_classes=1)
                elif config['arch'].startswith('vit'):
                    model = timm.create_model(config['arch'], pretrained=False, num_classes=1, img_size=224)
                else:
                    print(f"❌ Unknown architecture: {config['arch']}")
                    continue
                
                # Load weights
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(self.device)
                model.eval()
                
                self.models[model_name] = model
                self.model_weights[model_name] = config['weight']
                
                print(f"✅ {model_name}: {config['arch']} (weight: {config['weight']:.2f})")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {str(e)}")
                continue
        
        print(f"📊 Successfully loaded {len(self.models)} models")
    
    def predict_with_improved_confidence(self, image_path):
        """
        Predict with improved confidence calculation
        """
        if len(self.models) == 0:
            return {
                'error': 'No models loaded',
                'prediction': None,
                'confidence': 0.0
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions from each model
            model_probs = []
            model_outputs = []
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    # Get raw model output (single value with sigmoid)
                    raw_output = model(image_tensor)
                    prob_positive = torch.sigmoid(raw_output).cpu().item()
                    
                    model_probs.append(prob_positive)
                    model_outputs.append(raw_output.cpu().item())
            
            # Weighted ensemble probability
            weighted_prob = sum(prob * self.model_weights[name] 
                              for prob, name in zip(model_probs, self.models.keys()))
            
            prediction = int(weighted_prob > 0.5)
            
            # Calculate multiple confidence metrics
            
            # 1. Raw confidence (original method - for comparison)
            raw_confidence = abs(weighted_prob - 0.5) * 2
            
            # 2. Agreement-based confidence (how much do models agree?)
            model_predictions = [p > 0.5 for p in model_probs]
            agreement_ratio = sum(model_predictions) / len(model_predictions)
            agreement_confidence = max(agreement_ratio, 1 - agreement_ratio)  # Distance from 50/50 split
            
            # 3. Variance-based confidence (low variance = high confidence)
            prob_variance = np.var(model_probs)
            max_variance = 0.25  # Maximum possible variance for probabilities
            variance_confidence = 1.0 - (prob_variance / max_variance)
            
            # 4. Ensemble entropy confidence
            ensemble_entropy = -(weighted_prob * np.log(weighted_prob + 1e-8) + 
                               (1 - weighted_prob) * np.log(1 - weighted_prob + 1e-8))
            max_entropy = np.log(2)  # Maximum entropy for binary classification
            entropy_confidence = 1.0 - (ensemble_entropy / max_entropy)
            
            # 5. Output magnitude confidence (how far from decision boundary?)
            weighted_output = sum(output * self.model_weights[name] 
                                for output, name in zip(model_outputs, self.models.keys()))
            magnitude_confidence = min(1.0, abs(weighted_output) / 2.0)  # Normalize to [0, 1]
            
            # Combined improved confidence (weighted combination)
            improved_confidence = (
                agreement_confidence * 0.35 +      # Model agreement is very important
                variance_confidence * 0.25 +       # Low variance indicates consistency
                entropy_confidence * 0.20 +        # Entropy-based uncertainty
                magnitude_confidence * 0.20        # Distance from decision boundary
            )
            
            # Classification of confidence level
            if improved_confidence >= 0.85:
                confidence_level = "Very High"
                clinical_recommendation = "Suitable for autonomous clinical decision making"
            elif improved_confidence >= 0.75:
                confidence_level = "High"
                clinical_recommendation = "Good confidence, suitable for most clinical uses"
            elif improved_confidence >= 0.65:
                confidence_level = "Moderate"
                clinical_recommendation = "May benefit from expert review"
            elif improved_confidence >= 0.55:
                confidence_level = "Low"
                clinical_recommendation = "Recommend expert review"
            else:
                confidence_level = "Very Low"
                clinical_recommendation = "Expert review required"
            
            return {
                'prediction': prediction,
                'probability': float(weighted_prob),
                'prediction_label': 'Positive' if prediction == 1 else 'Negative',
                
                # Confidence metrics
                'raw_confidence': float(raw_confidence),
                'improved_confidence': float(improved_confidence),
                'agreement_confidence': float(agreement_confidence),
                'variance_confidence': float(variance_confidence),
                'entropy_confidence': float(entropy_confidence),
                'magnitude_confidence': float(magnitude_confidence),
                
                # Clinical interpretation
                'confidence_level': confidence_level,
                'clinical_recommendation': clinical_recommendation,
                
                # Model details
                'models_used': list(self.models.keys()),
                'individual_probabilities': {
                    name: float(prob) for name, prob in zip(self.models.keys(), model_probs)
                },
                'model_agreement': f"{sum(model_predictions)}/{len(model_predictions)} models agree",
                'probability_variance': float(prob_variance)
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }

def main():
    """Test the improved confidence calculation"""
    print("🎯 Testing Improved Ki-67 Confidence Calculation")
    print("=" * 55)
    
    # Initialize ensemble
    ensemble = ImprovedKi67Ensemble()
    
    if len(ensemble.models) == 0:
        print("❌ No models loaded. Cannot test.")
        return
    
    # Test images
    test_images = [
        "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test/6.png",
        "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test/1.png",
        "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test/10.png"
    ]
    
    print(f"\n🔍 Testing improved confidence on sample images:")
    print("-" * 55)
    
    all_results = []
    
    for test_image in test_images:
        if not Path(test_image).exists():
            print(f"❌ Test image not found: {test_image}")
            continue
        
        print(f"\n📁 Image: {Path(test_image).name}")
        result = ensemble.predict_with_improved_confidence(test_image)
        
        if 'error' not in result:
            print(f"   Prediction: {result['prediction_label']} (p={result['probability']:.3f})")
            print(f"   Raw Confidence: {result['raw_confidence']:.3f}")
            print(f"   Improved Confidence: {result['improved_confidence']:.3f} ({result['confidence_level']})")
            print(f"   Agreement: {result['model_agreement']}")
            print(f"   Variance: {result['probability_variance']:.4f}")
            print(f"   Clinical: {result['clinical_recommendation']}")
            
            # Show individual model contributions
            print(f"   Individual model probabilities:")
            for model_name, prob in result['individual_probabilities'].items():
                print(f"     {model_name}: {prob:.3f}")
            
            all_results.append(result)
        else:
            print(f"   ❌ Error: {result['error']}")
    
    # Save results
    if all_results:
        output_file = "improved_confidence_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'method': 'improved_confidence_calculation',
                'description': 'Results using improved confidence calculation methods',
                'results': all_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
    
    print("\n✅ Improved confidence testing complete!")

if __name__ == "__main__":
    main()
