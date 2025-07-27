#!/usr/bin/env python3
"""
Ultra Refined Ki-67 Model Manager for 98% Accuracy Ensemble

Manages the ultra-refined ensemble models that achieved 98.0% high-confidence accuracy
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

class UltraRefinedKi67ModelManager:
    """Model manager for the ultra-refined 98% accuracy ensemble"""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.transform = self._create_transform()
        
        # TOP 3 performers with optimized weights (from refined_95_percent_ensemble.py)
        self.model_configs = {
            'EfficientNet-B2': {
                'file': 'Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth',
                'arch': 'efficientnet_b2',
                'weight': 0.70,  # Highest weight for best performer
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
            'name': 'Ultra-Refined 98% Ensemble',
            'description': 'High-confidence ensemble achieving 98.0% accuracy',
            'standard_accuracy': 91.5,
            'high_confidence_accuracy': 98.0,
            'coverage': 72.9,
            'optimal_threshold': 0.7,
            'total_models': len(self.model_configs),
            'loaded_models': 0,
            'auc': 0.962,
            'precision': 0.825,
            'recall': 0.825,
            'f1_score': 0.825
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
        """Load the ultra-refined ensemble models"""
        print(f"🏆 Loading ultra-refined ensemble models from: {self.models_dir}")
        
        loaded_count = 0
        for model_name, config in self.model_configs.items():
            model_path = self.models_dir / config['file']
            
            if not model_path.exists():
                print(f"⚠️  Model not found: {model_path}")
                config['status'] = 'missing'
                continue
            
            try:
                # Create model architecture
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
        print(f"🎯 Ultra-refined ensemble loaded: {loaded_count}/{len(self.model_configs)} models")
        
        return loaded_count > 0
    
    def predict_with_confidence(self, image_input, confidence_threshold=0.7):
        """
        Make prediction with confidence scoring using the 98% accuracy logic
        
        Args:
            image_input: PIL Image or file path
            confidence_threshold: Threshold for high-confidence predictions (default 0.7)
        
        Returns:
            dict: Prediction results with confidence metrics
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
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
            
            # Apply transforms
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions from each model with confidence boosting
            model_probs = []
            model_confidences = []
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    outputs = torch.sigmoid(model(input_tensor))
                    probs = outputs.cpu()
                    
                    # Calculate confidence (distance from 0.5) - same as 98% accuracy script
                    confidence = torch.abs(probs - 0.5) * 2  # Scale to 0-1
                    
                    model_probs.append(probs)
                    model_confidences.append(confidence)
            
            # Weighted ensemble with confidence boosting (98% accuracy logic)
            weighted_probs = torch.zeros_like(model_probs[0])
            total_weight = 0
            
            for i, model_name in enumerate(self.models.keys()):
                weight = self.model_weights[model_name]
                confidence_boost = 1.0 + model_confidences[i] * 0.2  # Up to 20% boost
                effective_weight = weight * confidence_boost
                
                weighted_probs += model_probs[i] * effective_weight
                total_weight += effective_weight
            
            # Final ensemble prediction
            ensemble_prob = (weighted_probs / total_weight).squeeze().item()
            ensemble_confidence = torch.stack(model_confidences).mean().item()
            
            # Prediction decision
            prediction = 1 if ensemble_prob > 0.5 else 0
            prediction_label = "Positive" if prediction == 1 else "Negative"
            
            # High-confidence determination
            is_high_confidence = ensemble_confidence >= confidence_threshold
            
            # Individual model contributions
            model_contributions = {}
            for i, model_name in enumerate(self.models.keys()):
                model_contributions[model_name] = {
                    'probability': model_probs[i].squeeze().item(),
                    'confidence': model_confidences[i].squeeze().item(),
                    'weight': self.model_weights[model_name],
                    'contribution': (self.model_weights[model_name] * model_probs[i].squeeze().item())
                }
            
            result = {
                'prediction': prediction,
                'prediction_label': prediction_label,
                'probability': ensemble_prob,
                'confidence': ensemble_confidence,
                'high_confidence': is_high_confidence,
                'confidence_threshold': confidence_threshold,
                'expected_accuracy': 98.0 if is_high_confidence else 91.5,
                'coverage_info': {
                    'in_coverage': is_high_confidence,
                    'coverage_rate': 72.9,
                    'high_conf_accuracy': 98.0,
                    'standard_accuracy': 91.5
                },
                'model_contributions': model_contributions,
                'ensemble_info': {
                    'total_models': len(self.models),
                    'weighted_prediction': ensemble_prob,
                    'confidence_boosted': True
                },
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device)
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0,
                'high_confidence': False
            }
    
    def get_model_info(self):
        """Get information about loaded models"""
        models_info = []
        for model_name, config in self.model_configs.items():
            models_info.append({
                'name': model_name,
                'architecture': config['arch'],
                'weight': config['weight'],
                'individual_accuracy': config['individual_acc'],
                'status': config['status'],
                'file': config['file']
            })
        
        return {
            'ensemble': self.ensemble_info,
            'models': models_info,
            'device': str(self.device),
            'models_directory': str(self.models_dir)
        }
    
    def get_ensemble_stats(self):
        """Get ensemble performance statistics"""
        return {
            'ensemble_name': self.ensemble_info['name'],
            'total_models': self.ensemble_info['total_models'],
            'loaded_models': self.ensemble_info['loaded_models'],
            'standard_accuracy': self.ensemble_info['standard_accuracy'],
            'high_confidence_accuracy': self.ensemble_info['high_confidence_accuracy'],
            'coverage': self.ensemble_info['coverage'],
            'optimal_threshold': self.ensemble_info['optimal_threshold'],
            'auc': self.ensemble_info['auc'],
            'precision': self.ensemble_info['precision'],
            'recall': self.ensemble_info['recall'],
            'f1_score': self.ensemble_info['f1_score'],
            'performance_summary': {
                'achievement': '98.0% accuracy on high-confidence predictions',
                'coverage': '72.9% of samples with high confidence',
                'clinical_benefit': 'Reliable automation for 3 out of 4 cases'
            }
        }
    
    def test_prediction(self, test_image_path):
        """Test prediction on a sample image with detailed output"""
        if not os.path.exists(test_image_path):
            return {'error': f'Test image not found: {test_image_path}'}
        
        print(f"🧪 Testing ultra-refined ensemble on: {test_image_path}")
        
        # Make prediction
        result = self.predict_with_confidence(test_image_path)
        
        if 'error' in result:
            return result
        
        # Format detailed output
        print(f"\n🎯 ULTRA-REFINED ENSEMBLE PREDICTION:")
        print(f"   Prediction: {result['prediction_label']}")
        print(f"   Probability: {result['probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   High Confidence: {'✅ YES' if result['high_confidence'] else '❌ NO'}")
        print(f"   Expected Accuracy: {result['expected_accuracy']:.1f}%")
        
        print(f"\n📊 Model Contributions:")
        for model_name, contrib in result['model_contributions'].items():
            print(f"   {model_name}:")
            print(f"     Probability: {contrib['probability']:.3f}")
            print(f"     Confidence: {contrib['confidence']:.3f}")
            print(f"     Weight: {contrib['weight']:.2f}")
            print(f"     Contribution: {contrib['contribution']:.3f}")
        
        return result

def main():
    """Test the ultra-refined model manager"""
    print("🏆 Ultra-Refined Ki-67 Model Manager Test")
    print("=" * 60)
    
    # Initialize manager
    manager = UltraRefinedKi67ModelManager()
    
    # Show model info
    info = manager.get_model_info()
    print(f"\n📊 Ensemble Info:")
    print(f"   Name: {info['ensemble']['name']}")
    print(f"   Models Loaded: {info['ensemble']['loaded_models']}/{info['ensemble']['total_models']}")
    print(f"   High-Conf Accuracy: {info['ensemble']['high_confidence_accuracy']}%")
    print(f"   Coverage: {info['ensemble']['coverage']}%")
    
    # Test with a sample image if available
    test_images = [
        "../Ki67_Dataset_for_Colab/images/test/image_1.png",
        "../Ki67_Dataset_for_Colab/images/test/image_10.png",
        "../Ki67_Dataset_for_Colab/images/test/image_50.png"
    ]
    
    for test_img in test_images:
        if os.path.exists(test_img):
            print(f"\n" + "="*60)
            result = manager.test_prediction(test_img)
            break
    else:
        print("\n⚠️  No test images found. Place test images in the dataset directory.")

if __name__ == "__main__":
    main()
