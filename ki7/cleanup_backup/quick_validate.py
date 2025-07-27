#!/usr/bin/env python3
"""
Quick Ki-67 Model Validation Script
===================================
Simple validation of your current 3 models by directly scanning annotation folders
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
import glob

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

# Try to import timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

class DirectKi67Dataset(Dataset):
    """Dataset that directly scans annotation folders"""
    
    def __init__(self, dataset_path, split='test', transform=None):
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.split = split
        
        self.samples = []
        self.labels = []
        
        self.load_direct()
        
    def load_direct(self):
        """Load samples by directly scanning positive/negative folders"""
        
        # Scan positive samples
        pos_ann_dir = self.dataset_path / "annotations" / self.split / "positive"
        if pos_ann_dir.exists():
            for ann_file in pos_ann_dir.glob("*.h5"):
                img_name = f"{ann_file.stem}.png"
                img_path = self.dataset_path / "images" / self.split / img_name
                
                if img_path.exists():
                    self.samples.append(img_path)
                    self.labels.append(1)
        
        # Scan negative samples
        neg_ann_dir = self.dataset_path / "annotations" / self.split / "negative"
        if neg_ann_dir.exists():
            for ann_file in neg_ann_dir.glob("*.h5"):
                img_name = f"{ann_file.stem}.png"
                img_path = self.dataset_path / "images" / self.split / img_name
                
                if img_path.exists():
                    self.samples.append(img_path)
                    self.labels.append(0)
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"{self.split}: {len(self.samples)} samples ({pos_count} pos, {neg_count} neg)")
    
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

def load_model(model_path, model_type, device):
    """Load a single model"""
    
    # Create architecture
    if model_type == 'InceptionV3':
        model = models.inception_v3(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1),
            nn.Sigmoid()
        )
        if hasattr(model, 'AuxLogits'):
            model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.AuxLogits.fc.in_features, 1),
                nn.Sigmoid()
            )
    elif model_type == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1)
        )
    elif model_type == 'ViT':
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
        else:
            # Simple fallback
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
    else:
        return None
    
    model = model.to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading {model_type}: {e}")
        return None

def validate_model(model, test_loader, device, model_name):
    """Validate a single model"""
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
            
            # Get probabilities
            if model_name == 'ResNet50':
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs
            
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

def main():
    """Quick validation"""
    print("🚀 Quick Ki-67 Model Validation")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    base_dir = "/Users/chinthan/ki7"
    dataset_path = os.path.join(base_dir, "Ki67_Dataset_for_Colab")
    
    # Model files
    model_files = {
        'InceptionV3': os.path.join(base_dir, "Ki67_InceptionV3_best_model_20250619_070054.pth"),
        'ResNet50': os.path.join(base_dir, "Ki67_ResNet50_best_model_20250619_070508.pth"),
        'ViT': os.path.join(base_dir, "Ki67_ViT_best_model_20250619_071454.pth")
    }
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DirectKi67Dataset(dataset_path, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    if len(test_dataset) == 0:
        print("❌ No test data found")
        return
    
    # Load and validate each model
    results = {}
    
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            print(f"\n📊 Testing {model_name}...")
            model = load_model(model_path, model_name, device)
            
            if model is not None:
                result = validate_model(model, test_loader, device, model_name)
                results[model_name] = result
                
                print(f"✅ {model_name}: {result['accuracy']:.2f}% accuracy, {result['auc']:.2f}% AUC")
            else:
                print(f"❌ Failed to load {model_name}")
        else:
            print(f"❌ {model_name} file not found")
    
    if not results:
        print("❌ No models validated successfully")
        return
    
    # Create ensemble
    print(f"\n🤝 Creating ensemble...")
    model_names = list(results.keys())
    all_probs = np.column_stack([results[name]['probabilities'] for name in model_names])
    ensemble_probs = np.mean(all_probs, axis=1)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    true_labels = results[model_names[0]]['true_labels']
    
    ensemble_acc = accuracy_score(true_labels, ensemble_preds) * 100
    try:
        ensemble_auc = roc_auc_score(true_labels, ensemble_probs) * 100
    except:
        ensemble_auc = 50.0
    ensemble_f1 = f1_score(true_labels, ensemble_preds, zero_division=0) * 100
    
    # Results summary
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 45)
    
    for model_name, result in results.items():
        print(f"{model_name:<15} {result['accuracy']:>7.2f}%  {result['auc']:>6.2f}%  {result['f1_score']:>6.2f}%")
    
    print(f"{'ENSEMBLE':<15} {ensemble_acc:>7.2f}%  {ensemble_auc:>6.2f}%  {ensemble_f1:>6.2f}%")
    
    # Quick confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
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
    cm = confusion_matrix(true_labels, ensemble_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
               xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
               ax=axes[3])
    axes[3].set_title(f'Ensemble\n{ensemble_acc:.1f}%')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'quick_validation_results.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 Analysis:")
    if ensemble_acc >= 90:
        print(f"🎉 Excellent performance! {ensemble_acc:.1f}% ensemble accuracy")
    elif ensemble_acc >= 80:
        print(f"✅ Good performance! {ensemble_acc:.1f}% ensemble accuracy")
        print(f"💡 To reach 95%+: Add 1-2 more models (EfficientNet, ConvNeXt)")
    else:
        print(f"⚠️  Performance: {ensemble_acc:.1f}% ensemble accuracy")
        print(f"💡 Check if models trained correctly and dataset labels are correct")
    
    print(f"✅ Quick validation completed!")

if __name__ == "__main__":
    main()
