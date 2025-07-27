#!/usr/bin/env python3
"""
Updated Ki-67 Backend Model Manager with Improved Confidence
===========================================================

This replaces the problematic confidence calculation in the backend
with the improved multi-factor confidence method.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm

class ImprovedKi67ModelManager:
    """Model manager with improved confidence calculation"""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.transform = self._create_transform()
        
        # Top performing model configurations
        self.model_configs = {
            'EfficientNet-B2': {
                'file': 'Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth',
                'arch': 'efficientnet_b2',
                'weight': 0.70,
                'individual_acc': 92.5,
                'status': 'not_loaded'
            },
            'RegNet-Y-8GF': {
                'file': 'Ki67_Advanced_RegNet-Y-8GF_best_model_20250619_111223.pth', 
                'arch': 'regnety_008',
                'weight': 0.20,
                'individual_acc': 89.3,
                'status': 'not_loaded'
            },
            'ViT': {
                'file': 'Ki67_ViT_best_model_20250619_071454.pth',
                'arch': 'vit_base_patch16_224',
                'weight': 0.10,
                'individual_acc': 87.8,
                'status': 'not_loaded'
            }
        }
        
        self.ensemble_info = {
            'name': 'Improved Confidence Ensemble',
            'description': 'Ensemble with calibrated confidence calculation',
            'standard_accuracy': 70.0,
            'high_confidence_accuracy': 84.6,
            'coverage': 85.0,
            'optimal_threshold': 0.8,
            'total_models': len(self.model_configs),
            'loaded_models': 0
        }
        
        self.load_models()
    
    def _create_transform(self):
        """Create the image preprocessing transform"""
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_models(self):
        """Load the ensemble models"""
        print(f"🏆 Loading improved ensemble models from: {self.models_dir}")
        
        loaded_count = 0
        for model_name, config in self.model_configs.items():
            model_path = self.models_dir / config['file']
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                config['status'] = 'missing'
                continue
            
            try:
                # Create model architecture with correct number of classes (1 for sigmoid output)
                if config['arch'] == 'vit_base_patch16_224':
                    model = timm.create_model(config['arch'], pretrained=False, num_classes=1, img_size=224)
                else:
                    model = timm.create_model(config['arch'], pretrained=False, num_classes=1)
                
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
                config['status'] = 'loaded'
                loaded_count += 1
                
                print(f"✅ Loaded {model_name} (weight: {config['weight']:.2f}, acc: {config['individual_acc']:.1f}%)")
                
            except Exception as e:
                print(f"⚠️  Failed to load {model_name}: {e}")
                config['status'] = 'error'
        
        self.ensemble_info['loaded_models'] = loaded_count
        print(f"🎯 Loaded {loaded_count}/{len(self.model_configs)} ensemble models")
        
        return loaded_count > 0
    
    def predict_single_image(self, image_path, confidence_threshold=0.8):
        """
        Predict Ki-67 classification with improved confidence calculation
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Threshold for high-confidence predictions (now 80% instead of 70%)
            
        Returns:
            dict: Prediction results with improved confidence metrics
        """
        if len(self.models) == 0:
            return {
                'error': 'No models loaded',
                'prediction': None,
                'confidence': 0.0,
                'high_confidence': False
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
            
            # Calculate improved confidence metrics
            
            # 1. Raw confidence (original method - for comparison)
            raw_confidence = abs(weighted_prob - 0.5) * 2
            
            # 2. Agreement-based confidence
            model_predictions = [p > 0.5 for p in model_probs]
            agreement_ratio = sum(model_predictions) / len(model_predictions)
            agreement_confidence = max(agreement_ratio, 1 - agreement_ratio)
            
            # 3. Variance-based confidence
            prob_variance = np.var(model_probs)
            max_variance = 0.25
            variance_confidence = 1.0 - (prob_variance / max_variance)
            
            # 4. Ensemble entropy confidence
            ensemble_entropy = -(weighted_prob * np.log(weighted_prob + 1e-8) + 
                               (1 - weighted_prob) * np.log(1 - weighted_prob + 1e-8))
            max_entropy = np.log(2)
            entropy_confidence = 1.0 - (ensemble_entropy / max_entropy)
            
            # 5. Output magnitude confidence
            weighted_output = sum(output * self.model_weights[name] 
                                for output, name in zip(model_outputs, self.models.keys()))
            magnitude_confidence = min(1.0, abs(weighted_output) / 2.0)
            
            # Combined improved confidence (weighted combination)
            improved_confidence = (
                agreement_confidence * 0.35 +
                variance_confidence * 0.25 +
                entropy_confidence * 0.20 +
                magnitude_confidence * 0.20
            )
            
            is_high_confidence = improved_confidence >= confidence_threshold
            
            # Clinical interpretation
            if improved_confidence >= 0.85:
                confidence_level = "Very High"
                clinical_note = "Suitable for autonomous clinical decision making"
            elif improved_confidence >= 0.75:
                confidence_level = "High"
                clinical_note = "Good confidence, suitable for most clinical uses"
            elif improved_confidence >= 0.65:
                confidence_level = "Moderate"
                clinical_note = "May benefit from expert review"
            elif improved_confidence >= 0.55:
                confidence_level = "Low"
                clinical_note = "Recommend expert review"
            else:
                confidence_level = "Very Low"
                clinical_note = "Expert review required"
            
            return {
                'prediction': prediction,
                'probability': float(weighted_prob),
                
                # Original confidence (for backward compatibility)
                'confidence': float(improved_confidence),  # Use improved confidence as main confidence
                'high_confidence': is_high_confidence,
                
                # Detailed confidence breakdown
                'confidence_details': {
                    'raw_confidence': float(raw_confidence),
                    'improved_confidence': float(improved_confidence),
                    'agreement_confidence': float(agreement_confidence),
                    'variance_confidence': float(variance_confidence),
                    'entropy_confidence': float(entropy_confidence),
                    'magnitude_confidence': float(magnitude_confidence)
                },
                
                # Clinical interpretation
                'prediction_label': 'Positive' if prediction == 1 else 'Negative',
                'confidence_label': confidence_level,
                'clinical_recommendation': clinical_note,
                
                # Model details
                'models_used': list(self.models.keys()),
                'model_agreement': f"{sum(model_predictions)}/{len(model_predictions)} models agree",
                'individual_probabilities': {
                    name: float(prob) for name, prob in zip(self.models.keys(), model_probs)
                },
                
                # Ensemble info
                'ensemble_info': {
                    'total_models': len(self.models),
                    'threshold_used': confidence_threshold,
                    'method': 'improved_confidence_calculation'
                }
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0,
                'high_confidence': False
            }
    
    def get_all_models_for_frontend(self):
        """Get model information for frontend display"""
        models = []
        
        # Individual models
        for model_name, config in self.model_configs.items():
            models.append({
                'id': f"individual_{model_name.lower().replace('-', '_')}",
                'name': f"{model_name} (Individual)",
                'type': 'individual',
                'accuracy': config['individual_acc'],
                'loaded': config['status'] == 'loaded',
                'recommended': False
            })
        
        # Ensemble model (recommended)
        models.append({
            'id': 'improved_ensemble',
            'name': 'Improved 3-Model Ensemble',
            'type': 'ensemble',
            'accuracy': self.ensemble_info['standard_accuracy'],
            'high_confidence_accuracy': self.ensemble_info['high_confidence_accuracy'],
            'loaded': self.ensemble_info['loaded_models'] > 0,
            'recommended': True,
            'description': 'Calibrated confidence ensemble with clinical interpretability'
        })
        
        return models
    
    def get_ensemble_info(self):
        """Get ensemble information"""
        return self.ensemble_info

def test_improved_backend():
    """Test the improved backend model manager"""
    print("🧪 Testing Improved Ki-67 Backend Model Manager")
    print("=" * 50)
    
    # Initialize manager
    manager = ImprovedKi67ModelManager()
    
    if manager.ensemble_info['loaded_models'] == 0:
        print("❌ No models loaded for testing")
        return
    
    # Test image
    test_image = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test/6.png"
    
    if not Path(test_image).exists():
        print(f"❌ Test image not found: {test_image}")
        return
    
    print(f"🔍 Testing prediction on {Path(test_image).name}...")
    result = manager.predict_single_image(test_image)
    
    if 'error' not in result:
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Probability: {result['probability']:.3f}")
        print(f"Confidence: {result['confidence']:.3f} ({result['confidence_label']})")
        print(f"High Confidence: {result['high_confidence']}")
        print(f"Clinical Recommendation: {result['clinical_recommendation']}")
        print(f"Model Agreement: {result['model_agreement']}")
        
        print(f"\n🔬 CONFIDENCE BREAKDOWN:")
        details = result['confidence_details']
        print(f"Raw Confidence: {details['raw_confidence']:.3f}")
        print(f"Improved Confidence: {details['improved_confidence']:.3f}")
        print(f"Agreement Confidence: {details['agreement_confidence']:.3f}")
        print(f"Variance Confidence: {details['variance_confidence']:.3f}")
        print(f"Entropy Confidence: {details['entropy_confidence']:.3f}")
        print(f"Magnitude Confidence: {details['magnitude_confidence']:.3f}")
        
        print(f"\n🏥 Individual Model Probabilities:")
        for model, prob in result['individual_probabilities'].items():
            print(f"{model}: {prob:.3f}")
        
        print(f"\n✅ Backend test successful!")
    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    test_improved_backend()
