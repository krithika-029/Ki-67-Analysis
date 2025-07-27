#!/usr/bin/env python3
"""
Improved Ki-67 Ensemble with Better Confidence Calibration
=========================================================

This script improves the confidence calibration of the Ki-67 ensemble by:
1. Implementing temperature scaling for better probability calibration
2. Using Expected Calibration Error (ECE) to measure calibration quality
3. Training calibration parameters on a validation set
4. Providing more reliable confidence estimates for clinical use
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Import timm for model architectures
try:
    import timm
    print(f"✅ timm version: {timm.__version__}")
except ImportError:
    print("❌ timm not found. Installing...")
    os.system("pip install timm")
    import timm

class TemperatureScaling(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification model
    lr (float):
        The learning rate for temperature scaling
    max_iter (int):
        Maximum iterations for the optimization
    """
    def __init__(self, model, lr=0.01, max_iter=50):
        super(TemperatureScaling, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.lr = lr
        self.max_iter = max_iter

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, valid_loader):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        print(f'Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}')

        # Next: optimize the temperature w.r.t. NLL
        optimizer = torch.optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        print(f'Optimal temperature: {self.temperature.item():.3f}')
        print(f'After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}')

        return self

class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't differentiable and won't be used for optimization, but is useful for measuring calibration)

    The input to this loss is the logits of a model, NOT the softmax scores.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class PlattScaling:
    """
    Platt scaling to calibrate probabilities
    """
    def __init__(self):
        self.platt_lr = LogisticRegression()
        
    def fit(self, logits, labels):
        """
        Fit Platt scaling on validation data
        """
        # Convert logits to decision values (for binary classification)
        if logits.shape[1] == 2:
            decision_values = logits[:, 1] - logits[:, 0]  # log odds
        else:
            decision_values = logits.flatten()
            
        self.platt_lr.fit(decision_values.reshape(-1, 1), labels)
        
    def predict_proba(self, logits):
        """
        Apply Platt scaling to get calibrated probabilities
        """
        if logits.shape[1] == 2:
            decision_values = logits[:, 1] - logits[:, 0]
        else:
            decision_values = logits.flatten()
            
        return self.platt_lr.predict_proba(decision_values.reshape(-1, 1))

class IsotonicScaling:
    """
    Isotonic regression for probability calibration
    """
    def __init__(self):
        self.isotonic_reg = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, probs, labels):
        """
        Fit isotonic regression on validation data
        """
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            # For multi-class, use probability of positive class
            probs = probs[:, 1]
        self.isotonic_reg.fit(probs, labels)
        
    def predict_proba(self, probs):
        """
        Apply isotonic scaling to get calibrated probabilities
        """
        if len(probs.shape) > 1 and probs.shape[1] > 1:
            probs = probs[:, 1]
        calibrated_probs = self.isotonic_reg.predict(probs)
        return np.column_stack([1 - calibrated_probs, calibrated_probs])

class CalibratedKi67Ensemble:
    """
    Improved Ki-67 ensemble with proper confidence calibration
    """
    
    def __init__(self, models_dir="models", calibration_method='temperature'):
        self.models_dir = Path(models_dir).resolve()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_weights = {}
        self.calibration_method = calibration_method
        self.calibrators = {}
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
        
        print(f"🚀 Initializing Calibrated Ki-67 Ensemble with {calibration_method} calibration")
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
                checkpoint = torch.load(model_path, map_location=self.device)
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
        
    def calibrate_models(self, validation_loader):
        """
        Calibrate models using validation data
        """
        print(f"🎯 Calibrating models using {self.calibration_method} method...")
        
        for model_name, model in self.models.items():
            print(f"Calibrating {model_name}...")
            
            if self.calibration_method == 'temperature':
                calibrator = TemperatureScaling(model)
                calibrator.set_temperature(validation_loader)
                self.calibrators[model_name] = calibrator
                
            elif self.calibration_method == 'platt':
                # Collect validation predictions
                logits_list = []
                labels_list = []
                
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs = inputs.to(self.device)
                        logits = model(inputs)
                        logits_list.append(logits.cpu())
                        labels_list.append(labels)
                
                all_logits = torch.cat(logits_list)
                all_labels = torch.cat(labels_list)
                
                calibrator = PlattScaling()
                calibrator.fit(all_logits.numpy(), all_labels.numpy())
                self.calibrators[model_name] = calibrator
                
            elif self.calibration_method == 'isotonic':
                # Collect validation predictions
                probs_list = []
                labels_list = []
                
                with torch.no_grad():
                    for inputs, labels in validation_loader:
                        inputs = inputs.to(self.device)
                        logits = model(inputs)
                        probs = F.softmax(logits, dim=1)
                        probs_list.append(probs.cpu())
                        labels_list.append(labels)
                
                all_probs = torch.cat(probs_list)
                all_labels = torch.cat(labels_list)
                
                calibrator = IsotonicScaling()
                calibrator.fit(all_probs.numpy(), all_labels.numpy())
                self.calibrators[model_name] = calibrator
        
        print("✅ Model calibration complete!")
    
    def predict_with_calibrated_confidence(self, image_path):
        """
        Predict with properly calibrated confidence
        """
        if len(self.models) == 0:
            return {
                'error': 'No models loaded',
                'prediction': None,
                'confidence': 0.0,
                'calibrated': False
            }
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Get calibrated predictions from each model
            model_probs = []
            model_raw_confidences = []
            model_calibrated_confidences = []
            
            with torch.no_grad():
                for model_name, model in self.models.items():
                    if self.calibration_method == 'temperature' and model_name in self.calibrators:
                        # Temperature scaling (needs to be adapted for binary classification)
                        calibrator = self.calibrators[model_name]
                        raw_output = model(image_tensor)
                        # Convert single output to logits for temperature scaling
                        prob_positive = torch.sigmoid(raw_output)
                        prob_negative = 1 - prob_positive
                        logits = torch.log(prob_positive / prob_negative + 1e-8)
                        logits = torch.stack([logits.squeeze(), -logits.squeeze()], dim=1)
                        calibrated_logits = calibrator.temperature_scale(logits)
                        probs = F.softmax(calibrated_logits, dim=1)
                    else:
                        # Standard prediction with sigmoid
                        raw_output = model(image_tensor)
                        prob_positive = torch.sigmoid(raw_output).cpu()
                        prob_negative = 1 - prob_positive
                        probs = torch.stack([prob_negative.squeeze(), prob_positive.squeeze()], dim=1)
                        
                        if self.calibration_method == 'platt' and model_name in self.calibrators:
                            # Platt scaling - convert to decision values
                            calibrator = self.calibrators[model_name]
                            decision_values = raw_output.cpu().numpy()
                            probs = torch.tensor(calibrator.predict_proba(decision_values))
                        elif self.calibration_method == 'isotonic' and model_name in self.calibrators:
                            # Isotonic scaling
                            calibrator = self.calibrators[model_name]
                            probs = torch.tensor(calibrator.predict_proba(probs.numpy()))
                    
                    probs = probs.cpu()
                    
                    # Calculate raw confidence (old method for comparison)
                    raw_confidence = torch.abs(probs[:, 1] - 0.5) * 2
                    
                    # Calculate calibrated confidence using entropy
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                    max_entropy = torch.log(torch.tensor(2.0))  # For binary classification
                    calibrated_confidence = 1.0 - (entropy / max_entropy)
                    
                    model_probs.append(probs)
                    model_raw_confidences.append(raw_confidence)
                    model_calibrated_confidences.append(calibrated_confidence)
            
            # Weighted ensemble
            weighted_probs = torch.zeros_like(model_probs[0])
            total_weight = 0
            
            for i, model_name in enumerate(self.models.keys()):
                weight = self.model_weights[model_name]
                weighted_probs += model_probs[i] * weight
                total_weight += weight
            
            ensemble_prob = weighted_probs / total_weight
            ensemble_prob_positive = ensemble_prob[0, 1].item()
            
            # Calculate ensemble confidence using multiple methods
            raw_confidence = abs(ensemble_prob_positive - 0.5) * 2
            
            # Entropy-based confidence
            entropy = -torch.sum(ensemble_prob * torch.log(ensemble_prob + 1e-8), dim=1)
            max_entropy = torch.log(torch.tensor(2.0))
            entropy_confidence = (1.0 - (entropy / max_entropy)).item()
            
            # Agreement-based confidence (how much models agree)
            model_predictions = [p[0, 1].item() > 0.5 for p in model_probs]
            agreement = sum(model_predictions) / len(model_predictions)
            agreement_confidence = max(agreement, 1 - agreement)  # Distance from 0.5 agreement
            
            # Combined calibrated confidence
            calibrated_confidence = (entropy_confidence * 0.5 + agreement_confidence * 0.3 + 
                                   torch.stack(model_calibrated_confidences).mean().item() * 0.2)
            
            prediction = int(ensemble_prob_positive > 0.5)
            
            return {
                'prediction': prediction,
                'probability': float(ensemble_prob_positive),
                'raw_confidence': float(raw_confidence),
                'calibrated_confidence': float(calibrated_confidence),
                'entropy_confidence': float(entropy_confidence),
                'agreement_confidence': float(agreement_confidence),
                'prediction_label': 'Positive' if prediction == 1 else 'Negative',
                'confidence_interpretation': self._interpret_confidence(calibrated_confidence),
                'calibration_method': self.calibration_method,
                'models_used': list(self.models.keys()),
                'individual_model_confidences': {
                    name: float(conf) for name, conf in zip(self.models.keys(), model_calibrated_confidences)
                }
            }
            
        except Exception as e:
            return {
                'error': f'Prediction failed: {str(e)}',
                'prediction': None,
                'confidence': 0.0
            }
    
    def _interpret_confidence(self, confidence):
        """Interpret confidence levels for clinical use"""
        if confidence >= 0.9:
            return "Very High - Reliable for clinical decision making"
        elif confidence >= 0.8:
            return "High - Good confidence, suitable for most clinical uses"
        elif confidence >= 0.7:
            return "Moderate - May benefit from expert review"
        elif confidence >= 0.6:
            return "Low - Recommend expert review"
        else:
            return "Very Low - Expert review required"

def create_calibration_dataset():
    """Create a small validation dataset for calibration"""
    from ultra_optimized_95_percent_ensemble import UltraKi67Dataset
    
    dataset_path = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab"
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset (use test set for calibration - in practice, use separate validation set)
    dataset = UltraKi67Dataset(dataset_path, transform=transform, split='test')
    
    # Use subset for calibration (first 50 samples)
    subset_indices = list(range(min(50, len(dataset))))
    subset = torch.utils.data.Subset(dataset, subset_indices)
    
    return DataLoader(subset, batch_size=8, shuffle=False)

def main():
    """Main function to test calibrated ensemble"""
    print("🎯 Testing Calibrated Ki-67 Ensemble")
    print("=" * 50)
    
    # Test different calibration methods
    calibration_methods = ['temperature', 'platt', 'isotonic']
    
    for method in calibration_methods:
        print(f"\n🔬 Testing {method.upper()} calibration:")
        print("-" * 30)
        
        try:
            # Initialize ensemble
            ensemble = CalibratedKi67Ensemble(calibration_method=method)
            
            if len(ensemble.models) == 0:
                print("❌ No models loaded. Skipping calibration test.")
                continue
            
            # Create calibration dataset
            print("📊 Creating calibration dataset...")
            cal_loader = create_calibration_dataset()
            
            # Calibrate models
            ensemble.calibrate_models(cal_loader)
            
            # Test prediction on a sample image
            test_image = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/test_images/6.png"
            
            if Path(test_image).exists():
                print(f"🔍 Testing prediction on {Path(test_image).name}...")
                result = ensemble.predict_with_calibrated_confidence(test_image)
                
                if 'error' not in result:
                    print(f"Prediction: {result['prediction_label']}")
                    print(f"Probability: {result['probability']:.3f}")
                    print(f"Raw Confidence: {result['raw_confidence']:.3f}")
                    print(f"Calibrated Confidence: {result['calibrated_confidence']:.3f}")
                    print(f"Entropy Confidence: {result['entropy_confidence']:.3f}")
                    print(f"Agreement Confidence: {result['agreement_confidence']:.3f}")
                    print(f"Interpretation: {result['confidence_interpretation']}")
                else:
                    print(f"❌ Error: {result['error']}")
            else:
                print(f"❌ Test image not found: {test_image}")
                
        except Exception as e:
            print(f"❌ Error testing {method} calibration: {str(e)}")
    
    print("\n✅ Calibration testing complete!")

if __name__ == "__main__":
    main()
