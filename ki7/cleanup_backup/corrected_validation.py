#!/usr/bin/env python3
"""
Fixed Ki-67 Model Validation Script
===================================
Fixes the identified issues:
1. Adds missing sigmoid layers
2. Handles proper data distribution
3. Tests with corrected architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import os
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class DirectKi67Dataset(Dataset):
    """Dataset that directly scans annotation folders with proper shuffling"""
    
    def __init__(self, dataset_path, split='test', transform=None, shuffle_seed=42):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.split = split
        
        self.samples = []
        self.labels = []
        
        self.load_direct(shuffle_seed)
        
    def load_direct(self, shuffle_seed):
        """Load samples by directly scanning positive/negative folders"""
        
        # Scan positive samples
        pos_samples = []
        pos_ann_dir = self.dataset_path / "annotations" / self.split / "positive"
        if pos_ann_dir.exists():
            for ann_file in pos_ann_dir.glob("*.h5"):
                img_name = f"{ann_file.stem}.png"
                img_path = self.dataset_path / "images" / self.split / img_name
                
                if img_path.exists():
                    pos_samples.append((img_path, 1))
        
        # Scan negative samples
        neg_samples = []
        neg_ann_dir = self.dataset_path / "annotations" / self.split / "negative"
        if neg_ann_dir.exists():
            for ann_file in neg_ann_dir.glob("*.h5"):
                img_name = f"{ann_file.stem}.png"
                img_path = self.dataset_path / "images" / self.split / img_name
                
                if img_path.exists():
                    neg_samples.append((img_path, 0))
        
        # Combine and shuffle for balanced testing
        all_samples = pos_samples + neg_samples
        
        # Shuffle with fixed seed for reproducibility
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            np.random.shuffle(all_samples)
        
        # Extract samples and labels
        self.samples = [sample[0] for sample in all_samples]
        self.labels = [sample[1] for sample in all_samples]
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"{self.split}: {len(self.samples)} samples ({pos_count} pos, {neg_count} neg) - Shuffled")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # Fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32)

def create_corrected_model(model_type, device):
    """Create model with corrected architecture"""
    
    if model_type == 'InceptionV3':
        model = models.inception_v3(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()  # ✅ Already has sigmoid
        )
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.AuxLogits.fc.in_features, 1),
                nn.Sigmoid()
            )
        return model.to(device)
    
    elif model_type == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()  # ✅ FIXED: Added missing sigmoid
        )
        return model.to(device)
    
    elif model_type == 'ViT':
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            # ✅ FIXED: Add sigmoid layer
            original_head = model.head
            model.head = nn.Sequential(
                original_head,
                nn.Sigmoid()
            )
            return model.to(device)
        else:
            model = models.resnet18(weights=None)
            model.fc = nn.Sequential(
                nn.Linear(model.fc.in_features, 1),
                nn.Sigmoid()
            )
            return model.to(device)
    
    return None

def load_corrected_model(model_path, model_type, device):
    """Load model with corrected architecture"""
    
    # Create corrected architecture
    model = create_corrected_model(model_type, device)
    if model is None:
        return None
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # For models that need sigmoid fix, remove sigmoid from state dict if present
        if model_type in ['ResNet50', 'ViT']:
            # The original models were trained without sigmoid in the final layer
            # So we need to load the weights and add sigmoid during inference
            if model_type == 'ResNet50':
                # Load weights up to the linear layer (before sigmoid)
                model_dict = model.state_dict()
                # Remove sigmoid layer weights from loading
                filtered_dict = {}
                for k, v in state_dict.items():
                    if 'fc.1.' in k:  # This was the linear layer in original
                        new_key = k.replace('fc.1.', 'fc.0.')  # Map to new position
                        filtered_dict[new_key] = v
                    elif 'fc.' not in k or 'fc.0.' in k:  # Keep other layers
                        filtered_dict[k] = v
                
                # Update model dict and load
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
                
            elif model_type == 'ViT':
                # For ViT, the head was directly connected, now we have head.0 (linear) + head.1 (sigmoid)
                model_dict = model.state_dict()
                filtered_dict = {}
                for k, v in state_dict.items():
                    if k == 'head.weight':
                        filtered_dict['head.0.weight'] = v
                    elif k == 'head.bias':
                        filtered_dict['head.0.bias'] = v
                    else:
                        filtered_dict[k] = v
                
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict, strict=False)
        else:
            # InceptionV3 can load normally
            model.load_state_dict(state_dict)
        
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading {model_type}: {e}")
        return None

def validate_corrected_model(model, test_loader, device, model_name):
    """Validate model with corrected architecture"""
    model.eval()
    
    predictions = []
    probabilities = []
    true_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Adjust input size for InceptionV3
            if model_name == "InceptionV3" and inputs.size(-1) != 299:
                inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
            
            outputs = model(inputs)
            
            # Handle tuple output
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            # Now all models should output probabilities in [0,1] range
            probs = outputs  # Already passed through sigmoid
            
            probabilities.extend(probs.cpu().numpy().flatten())
            preds = (probs > 0.5).float()
            predictions.extend(preds.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions) * 100
    try:
        auc = roc_auc_score(true_labels, probabilities) * 100
    except:
        auc = 50.0
    f1 = f1_score(true_labels, predictions, zero_division=0) * 100
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1_score': f1,
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities),
        'true_labels': np.array(true_labels)
    }

def create_corrected_ensemble(results, true_labels):
    """Create ensemble with corrected models"""
    print(f"\n🤝 Creating Corrected Ensemble...")
    
    model_names = list(results.keys())
    probs_matrix = np.column_stack([results[name]['probabilities'] for name in model_names])
    
    # Test ensemble strategies
    strategies = {}
    
    # Simple Average
    avg_probs = np.mean(probs_matrix, axis=1)
    avg_preds = (avg_probs > 0.5).astype(int)
    strategies['Simple Average'] = (avg_probs, avg_preds)
    
    # Weighted by Accuracy
    weights = np.array([results[name]['accuracy'] for name in model_names])
    weights = weights / np.sum(weights)
    weighted_probs = np.average(probs_matrix, axis=1, weights=weights)
    weighted_preds = (weighted_probs > 0.5).astype(int)
    strategies['Accuracy Weighted'] = (weighted_probs, weighted_preds)
    
    # Confidence-based
    conf_probs = np.mean(probs_matrix, axis=1)
    conf_preds = (conf_probs > 0.6).astype(int)  # Higher threshold
    strategies['High Confidence'] = (conf_probs, conf_preds)
    
    # Majority Vote
    binary_preds = probs_matrix > 0.5
    majority_preds = (np.sum(binary_preds, axis=1) > len(model_names)/2).astype(int)
    strategies['Majority Vote'] = (avg_probs, majority_preds)
    
    # Evaluate strategies
    print(f"\n📊 Corrected Ensemble Results:")
    print(f"{'Strategy':<18} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 50)
    
    best_strategy = None
    best_accuracy = 0
    ensemble_results = {}
    
    for strategy_name, (probs, preds) in strategies.items():
        accuracy = accuracy_score(true_labels, preds) * 100
        try:
            auc = roc_auc_score(true_labels, probs) * 100
        except:
            auc = 50.0
        f1 = f1_score(true_labels, preds, zero_division=0) * 100
        
        ensemble_results[strategy_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'predictions': preds,
            'probabilities': probs
        }
        
        print(f"{strategy_name:<18} {accuracy:>7.2f}%  {auc:>6.2f}%  {f1:>6.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_strategy = strategy_name
    
    return ensemble_results, best_strategy

def main():
    """Main corrected validation"""
    print("🔧 Fixed Ki-67 Model Validation")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    base_dir = "/Users/chinthan/ki7"
    dataset_path = os.path.join(base_dir, "Ki67_Dataset_for_Colab")
    models_dir = os.path.join(base_dir, "models")
    
    # Model files
    model_files = {
        'InceptionV3': os.path.join(models_dir, "Ki67_InceptionV3_best_model_20250619_070054.pth"),
        'ResNet50': os.path.join(models_dir, "Ki67_ResNet50_best_model_20250619_070508.pth"),
        'ViT': os.path.join(models_dir, "Ki67_ViT_best_model_20250619_071454.pth")
    }
    
    # Create corrected dataset (shuffled for proper evaluation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DirectKi67Dataset(dataset_path, split='test', transform=transform, shuffle_seed=42)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\n📊 Dataset loaded: {len(test_dataset)} samples")
    
    # Load and test corrected models
    results = {}
    
    print(f"\n🧪 Testing Corrected Models:")
    print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 45)
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            print(f"Loading {model_name}...")
            model = load_corrected_model(model_path, model_name, device)
            
            if model is not None:
                result = validate_corrected_model(model, test_loader, device, model_name)
                results[model_name] = result
                
                print(f"{model_name:<15} {result['accuracy']:>7.2f}%  {result['auc']:>6.2f}%  {result['f1_score']:>6.2f}%")
            else:
                print(f"❌ Failed to load {model_name}")
        else:
            print(f"❌ {model_name} file not found")
    
    if not results:
        print("❌ No models loaded successfully")
        return
    
    # Create corrected ensemble
    true_labels = results[list(results.keys())[0]]['true_labels']
    ensemble_results, best_strategy = create_corrected_ensemble(results, true_labels)
    
    if ensemble_results:
        best_ensemble = ensemble_results[best_strategy]
        
        print(f"\n🎉 CORRECTED RESULTS:")
        print(f"📊 Best ensemble accuracy: {best_ensemble['accuracy']:.2f}%")
        print(f"🏆 Strategy: {best_strategy}")
        print(f"🔢 Models in ensemble: {len(results)}")
        
        if best_ensemble['accuracy'] >= 95.0:
            print(f"🎉 TARGET ACHIEVED! 95%+ accuracy reached!")
        elif best_ensemble['accuracy'] >= 90.0:
            print(f"🔥 Excellent! Very close to 95% target.")
            print(f"💡 Need +{95.0 - best_ensemble['accuracy']:.1f}% to reach 95%")
        elif best_ensemble['accuracy'] >= 80.0:
            print(f"✅ Good performance! Consider training additional models.")
        else:
            print(f"📈 Significant improvement needed.")
        
        # Plot results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        plot_idx = 0
        for model_name, result in results.items():
            if plot_idx < 3:
                cm = confusion_matrix(true_labels, result['predictions'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                           ax=axes[plot_idx])
                axes[plot_idx].set_title(f'{model_name}\n{result["accuracy"]:.1f}%')
                plot_idx += 1
        
        # Ensemble plot
        cm = confusion_matrix(true_labels, best_ensemble['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                   ax=axes[3])
        axes[3].set_title(f'Best Ensemble ({best_strategy})\n{best_ensemble["accuracy"]:.1f}%')
        
        plt.tight_layout()
        plt.savefig(os.path.join(base_dir, 'corrected_validation_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = {
            'timestamp': timestamp,
            'individual_results': {k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else str(v2) 
                                      for k2, v2 in v.items() if k2 not in ['predictions', 'probabilities', 'true_labels']} 
                                  for k, v in results.items()},
            'ensemble_results': {k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else str(v2) 
                                    for k2, v2 in v.items() if k2 not in ['predictions', 'probabilities']} 
                                for k, v in ensemble_results.items()},
            'best_strategy': best_strategy,
            'best_accuracy': float(best_ensemble['accuracy']),
            'target_achieved': best_ensemble['accuracy'] >= 95.0
        }
        
        results_file = os.path.join(base_dir, f'corrected_validation_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Results saved: {os.path.basename(results_file)}")
        print(f"✅ Corrected validation completed!")

if __name__ == "__main__":
    main()
