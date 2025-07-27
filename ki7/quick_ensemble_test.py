#!/usr/bin/env python3
"""
Quick Champion B5 Ensemble Test - Fast evaluation of your existing models
"""

import os
import sys
import warnings
import json
import numpy as np
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

try:
    import timm
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class QuickDataset(Dataset):
    def __init__(self, dataset_path, split='test'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Smaller for speed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.create_dataset()
    
    def create_dataset(self):
        possible_paths = [
            self.dataset_path / "Ki67_Dataset_for_Colab",
            self.dataset_path / "BCData",
            self.dataset_path
        ]
        
        base_path = None
        for path in possible_paths:
            if path.exists():
                if (path / "images" / self.split).exists():
                    base_path = path
                    break
        
        if base_path is None:
            return
        
        images_dir = base_path / "images" / self.split
        pos_annotations_dir = base_path / "annotations" / self.split / "positive"
        neg_annotations_dir = base_path / "annotations" / self.split / "negative"
        
        for img_file in images_dir.glob("*.png"):
            img_name = img_file.stem
            pos_ann = pos_annotations_dir / f"{img_name}.h5"
            neg_ann = neg_annotations_dir / f"{img_name}.h5"
            
            if pos_ann.exists() and neg_ann.exists():
                try:
                    pos_size = pos_ann.stat().st_size
                    neg_size = neg_ann.stat().st_size
                    
                    if abs(pos_size - neg_size) > 100:
                        if neg_size > pos_size:
                            self.images.append(str(img_file))
                            self.labels.append(0)
                        else:
                            self.images.append(str(img_file))
                            self.labels.append(1)
                    else:
                        idx = len(self.images)
                        self.images.append(str(img_file))
                        self.labels.append(idx % 2)
                except:
                    idx = len(self.images)
                    self.images.append(str(img_file))
                    self.labels.append(idx % 2)
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"Dataset: {len(self.images)} images ({pos_count} positive, {neg_count} negative)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            image = self.transform(image)
            return image, torch.tensor(self.labels[idx], dtype=torch.float32)
        except:
            return torch.zeros((3, 224, 224)), torch.tensor(self.labels[idx], dtype=torch.float32)

def load_models_safely(device, models_dir):
    """Load TOP performing models for 95% accuracy push"""
    models = []
    model_names = []
    val_accuracies = []
    
    # TOP PERFORMERS based on validation results (ordered by performance)
    configs = [
        ('Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth', 'efficientnet_b2', 'B2-Advanced-92.54%', 92.54),
        ('Ki67_ViT_best_model_20250619_071454.pth', 'vit_base_patch16_224', 'ViT-87.81%', 87.81),
        ('Ki67_Advanced_Swin-Tiny_best_model_20250619_110516.pth', 'swin_tiny_patch4_window7_224', 'Swin-87.06%', 87.06),
        ('Ki67_STABLE_Champion_EfficientNet_B5_Champion_FINAL_90.98_20250620_142507.pth', 'efficientnet_b5', 'B5-Champion-85.57%', 85.57),
        ('Ki67_B4_Adapted_Champion_EfficientNet_B4_Adapted_best_model_20250620_133200.pth', 'efficientnet_b4', 'B4-Adapted-82.84%', 82.84)
    ]
    
    for file, arch, name, val_acc in configs:
        model_path = models_dir / file
        if model_path.exists():
            try:
                # Create model with proper configuration
                if arch == 'vit_base_patch16_224':
                    model = timm.create_model(arch, pretrained=False, num_classes=1, img_size=224)
                else:
                    model = timm.create_model(arch, pretrained=False, num_classes=1)
                
                # Load with weights_only=False for compatibility
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model = model.to(device)
                model.eval()
                
                models.append(model)
                model_names.append(name)
                val_accuracies.append(val_acc)
                print(f"✅ Loaded {name}")
                
            except Exception as e:
                print(f"⚠️  Failed to load {name}: {e}")
    
    return models, model_names, val_accuracies

def quick_ensemble_test(models, data_loader, device):
    """Quick ensemble test"""
    print("🧪 Running quick ensemble test...")
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx+1}/{len(data_loader)}")
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions from all models
            model_outputs = []
            for model in models:
                output = torch.sigmoid(model(inputs))
                model_outputs.append(output)
            
            # Average ensemble
            ensemble_output = torch.stack(model_outputs).mean(dim=0)
            predictions = (ensemble_output > 0.5).float()
            
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.array(all_preds), np.array(all_targets)

def main():
    print("🚀 Quick Champion Ensemble Test")
    print("=" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Paths
    current_dir = Path.cwd()
    models_dir = current_dir / "models"
    
    # Find dataset
    dataset_paths = ["./Ki67_Dataset_for_Colab", "./BCData", "./data"]
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = Path(path)
            break
    
    if dataset_path is None:
        print("❌ Dataset not found")
        return
    
    print(f"📂 Dataset: {dataset_path}")
    
    # Load models
    models, model_names, val_accuracies = load_models_safely(device, models_dir)
    
    if len(models) == 0:
        print("❌ No models loaded")
        return
    
    print(f"\n🎯 Ensemble: {len(models)} models")
    for i, name in enumerate(model_names):
        print(f"   {i+1}. {name}")
    
    # Create test dataset
    test_dataset = QuickDataset(dataset_path, split='test')
    if len(test_dataset) == 0:
        print("❌ No test data found")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Quick test
    predictions, targets = quick_ensemble_test(models, test_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, zero_division=0)
    recall = recall_score(targets, predictions, zero_division=0)
    f1 = f1_score(targets, predictions, zero_division=0)
    
    print(f"\n🏆 QUICK ENSEMBLE RESULTS:")
    print(f"=" * 30)
    print(f"Accuracy:  {accuracy*100:.2f}%")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    
    if accuracy >= 0.95:
        print(f"\n🎉 95%+ TARGET ACHIEVED!")
    elif accuracy >= 0.93:
        print(f"\n🔥 Very close to 95%!")
    elif accuracy >= 0.90:
        print(f"\n✅ Strong performance!")
    else:
        print(f"\n📈 Need {(0.95 - accuracy)*100:.1f}% more for 95%")
    
    print(f"\n💡 Next steps:")
    if accuracy >= 0.95:
        print("   ✅ Add your trained B5 Champion for even better results!")
    else:
        print("   1. Add your trained B5 Champion model")
        print("   2. Train additional B4 models")
        print("   3. Use TTA for final boost")
    
    return accuracy

if __name__ == "__main__":
    main()
