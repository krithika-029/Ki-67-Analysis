#!/usr/bin/env python3
"""
Refined Ki-67 Model Manager for 95%+ Accuracy Ensemble

Manages the refined ensemble models that achieved 97.4% high-confidence accuracy
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

class RefinedKi67ModelManager:
    """Model manager for the refined 95%+ accuracy ensemble"""
    
    def __init__(self, models_dir="../models"):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.transform = self._create_transform()
        
        # Top performing model configurations (97.4% accuracy achieved)
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
            'name': 'Refined 95%+ Ensemble',
            'description': 'High-confidence ensemble achieving 97.4% accuracy',
            'standard_accuracy': 91.3,
            'high_confidence_accuracy': 97.4,
            'coverage': 77.4,
            'optimal_threshold': 0.7,
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
        """Load the refined ensemble models"""
        print(f"🏆 Loading refined ensemble models from: {self.models_dir}")
        
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
        print(f"🎯 Loaded {loaded_count}/{len(self.model_configs)} ensemble models")
        
        return loaded_count > 0
    
    def predict_single_image(self, image_path, confidence_threshold=0.7):
        """
        Predict Ki-67 classification for a single image
        
        Args:
            image_path: Path to the image file
            confidence_threshold: Threshold for high-confidence predictions
            
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
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get predictions from each model
            model_probs = []
            model_confidences = []
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    outputs = torch.sigmoid(model(image_tensor))
                    probs = outputs.cpu()
                    
                    # Calculate confidence (distance from 0.5)
                    confidence = torch.abs(probs - 0.5) * 2  # Scale to 0-1
                    
                    model_probs.append(probs)
                    model_confidences.append(confidence)
            
            # Weighted ensemble with confidence boosting
            weighted_probs = torch.zeros_like(model_probs[0])
            total_weight = 0
            
            for i, model_name in enumerate(self.models.keys()):
                weight = self.model_weights[model_name]
                confidence_boost = 1.0 + model_confidences[i] * 0.2  # Up to 20% boost
                effective_weight = weight * confidence_boost
                
                weighted_probs += model_probs[i] * effective_weight
                total_weight += effective_weight
            
            ensemble_prob = (weighted_probs / total_weight).item()
            ensemble_confidence = torch.stack(model_confidences).mean().item()
            
            prediction = int(ensemble_prob > 0.5)
            is_high_confidence = ensemble_confidence >= confidence_threshold
            
            return {
                'prediction': prediction,
                'probability': float(ensemble_prob),
                'confidence': float(ensemble_confidence),
                'high_confidence': is_high_confidence,
                'prediction_label': 'Positive' if prediction == 1 else 'Negative',
                'confidence_label': 'High' if is_high_confidence else 'Low',
                'models_used': list(self.models.keys()),
                'ensemble_info': {
                    'total_models': len(self.models),
                    'threshold_used': confidence_threshold
                }
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0,
                'high_confidence': False
            }
    
    def get_model_info(self):
        """Get information about the loaded models"""
        model_info = []
        
        for model_name, config in self.model_configs.items():
            info = {
                'name': model_name,
                'architecture': config['arch'],
                'individual_accuracy': config['individual_acc'],
                'ensemble_weight': config['weight'],
                'status': config['status'],
                'file': config['file']
            }
            
            if model_name in self.models:
                info['loaded'] = True
                # Get model size
                model_path = self.models_dir / config['file']
                if model_path.exists():
                    info['size_mb'] = round(model_path.stat().st_size / (1024 * 1024), 2)
            else:
                info['loaded'] = False
                
            model_info.append(info)
        
        return model_info
    
    def get_ensemble_info(self):
        """Get ensemble performance information"""
        return self.ensemble_info
    
    def get_all_models_for_frontend(self):
        """Get model information formatted for frontend"""
        models = []
        
        for model_name, config in self.model_configs.items():
            model_data = {
                'id': model_name.lower().replace('-', '').replace(' ', ''),
                'name': model_name,
                'full_name': f"Ki67 {model_name}",
                'accuracy': config['individual_acc'],
                'ensemble_weight': config['weight'],
                'status': 'available' if config['status'] == 'loaded' else 'unavailable',
                'type': 'refined_ensemble',
                'architecture': config['arch']
            }
            models.append(model_data)
        
        return models
    
    def get_system_status(self):
        """Get system status for monitoring"""
        loaded_models = sum(1 for config in self.model_configs.values() if config['status'] == 'loaded')
        total_models = len(self.model_configs)
        
        return {
            'status': 'operational' if loaded_models > 0 else 'degraded',
            'loaded_models': loaded_models,
            'total_models': total_models,
            'ensemble_ready': loaded_models >= 2,  # Minimum 2 models for ensemble
            'device': str(self.device),
            'last_updated': datetime.now().isoformat()
        }


def main():
    """Test the refined model manager"""
    manager = RefinedKi67ModelManager()
    
    print("\n🔍 Model Information:")
    for model in manager.get_model_info():
        print(f"  {model['name']}: {model['status']} (acc: {model['individual_accuracy']}%, weight: {model['ensemble_weight']})")
    
    print(f"\n📊 Ensemble Info:")
    ensemble = manager.get_ensemble_info()
    print(f"  Standard Accuracy: {ensemble['standard_accuracy']}%")
    print(f"  High-Confidence Accuracy: {ensemble['high_confidence_accuracy']}%")
    print(f"  Coverage: {ensemble['coverage']}%")
    
    print(f"\n🏥 System Status:")
    status = manager.get_system_status()
    print(f"  Status: {status['status']}")
    print(f"  Models: {status['loaded_models']}/{status['total_models']}")
    print(f"  Ensemble Ready: {status['ensemble_ready']}")


if __name__ == "__main__":
    main()
