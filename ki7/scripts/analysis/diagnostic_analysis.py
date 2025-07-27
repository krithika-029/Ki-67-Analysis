#!/usr/bin/env python3
"""
Ki-67 Model Diagnostic Script
============================
Debug why models show 50% accuracy despite high training performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
from PIL import Image
from pathlib import Path
import os

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
            return image, torch.tensor(label, dtype=torch.float32), str(img_path)
        except Exception as e:
            # Fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32), str(img_path)

def load_and_inspect_model(model_path, model_type, device):
    """Load model and inspect its structure and weights"""
    print(f"\n🔍 INSPECTING {model_type} MODEL")
    print("=" * 40)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    print("📋 Checkpoint Analysis:")
    if isinstance(checkpoint, dict):
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        if 'model_state_dict' in checkpoint:
            print("  ✅ Has model_state_dict")
            state_dict = checkpoint['model_state_dict']
        else:
            print("  ⚠️  Direct state dict")
            state_dict = checkpoint
        
        # Print metadata
        for key in ['epoch', 'val_loss', 'val_acc', 'performance_summary']:
            if key in checkpoint:
                print(f"  {key}: {checkpoint[key]}")
    else:
        print("  Direct state dict without metadata")
        state_dict = checkpoint
    
    # Analyze state dict
    print(f"\n📊 Model Structure Analysis:")
    layer_count = len(state_dict)
    print(f"  Total layers: {layer_count}")
    
    # Show first few layers
    print("  First 5 layers:")
    for i, (name, tensor) in enumerate(list(state_dict.items())[:5]):
        print(f"    {name}: {tensor.shape}")
    
    # Show final layers
    print("  Last 5 layers:")
    for name, tensor in list(state_dict.items())[-5:]:
        print(f"    {name}: {tensor.shape}")
    
    # Create model architecture
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
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
    
    model = model.to(device)
    
    # Try to load weights
    try:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("  ✅ Model weights loaded successfully")
        
        # Check if model is properly loaded by examining some weights
        first_param = next(iter(model.parameters()))
        print(f"  First parameter stats: mean={first_param.mean():.6f}, std={first_param.std():.6f}")
        
        return model
        
    except Exception as e:
        print(f"  ❌ Error loading weights: {e}")
        return None

def test_model_detailed(model, test_loader, device, model_name):
    """Test model with detailed output analysis"""
    print(f"\n🧪 DETAILED TESTING: {model_name}")
    print("=" * 40)
    
    model.eval()
    
    predictions = []
    probabilities = []
    true_labels = []
    sample_outputs = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels, paths) in enumerate(test_loader):
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
            
            # Store detailed info for first batch
            if batch_idx == 0:
                print(f"📊 First batch analysis:")
                print(f"  Input shape: {inputs.shape}")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Raw outputs (first 5): {outputs.flatten()[:5].cpu().numpy()}")
                print(f"  Probabilities (first 5): {probs.flatten()[:5].cpu().numpy()}")
                print(f"  True labels (first 5): {labels.flatten()[:5].cpu().numpy()}")
                
                # Check if all outputs are the same (indicating a problem)
                output_range = outputs.max() - outputs.min()
                print(f"  Output range: {output_range:.6f}")
                if output_range < 1e-6:
                    print("  ⚠️  WARNING: All outputs are nearly identical!")
            
            probabilities.extend(probs.cpu().numpy().flatten())
            preds = (probs > 0.5).float()
            predictions.extend(preds.cpu().numpy().flatten())
            true_labels.extend(labels.cpu().numpy().flatten())
            
            # Only process first few batches for detailed analysis
            if batch_idx >= 5:
                break
    
    # Overall statistics
    predictions = np.array(predictions)
    probabilities = np.array(probabilities)
    true_labels = np.array(true_labels)
    
    print(f"\n📈 Prediction Statistics:")
    print(f"  Probability range: {probabilities.min():.4f} to {probabilities.max():.4f}")
    print(f"  Probability mean: {probabilities.mean():.4f}")
    print(f"  Prediction distribution: {np.bincount(predictions.astype(int))}")
    print(f"  True label distribution: {np.bincount(true_labels.astype(int))}")
    
    # Check if model is just predicting one class
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 1:
        print(f"  ⚠️  WARNING: Model only predicts class {unique_preds[0]}!")
    
    accuracy = np.mean(predictions == true_labels) * 100
    print(f"  Accuracy: {accuracy:.2f}%")
    
    return accuracy

def main():
    """Main diagnostic function"""
    print("🔬 Ki-67 Model Diagnostic Analysis")
    print("=" * 50)
    
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
    
    # Create test dataset (smaller for detailed analysis)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DirectKi67Dataset(dataset_path, split='test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)  # Small batch for debugging
    
    print(f"\n📊 Dataset loaded: {len(test_dataset)} samples")
    
    # Analyze each model
    for model_name, model_path in model_files.items():
        if os.path.exists(model_path):
            # Inspect model structure
            model = load_and_inspect_model(model_path, model_name, device)
            
            if model is not None:
                # Test model performance
                accuracy = test_model_detailed(model, test_loader, device, model_name)
            else:
                print(f"❌ Could not load {model_name}")
        else:
            print(f"❌ {model_name} file not found")
    
    print(f"\n🎯 DIAGNOSTIC SUMMARY:")
    print("=" * 30)
    print("If you see:")
    print("  • All outputs nearly identical → Model weights may be corrupted")
    print("  • Only predicting one class → Threshold or sigmoid issue")
    print("  • Probability range [0,1] but wrong predictions → Training/validation data mismatch")
    print("  • Probability range not [0,1] → Missing sigmoid or wrong architecture")

if __name__ == "__main__":
    main()
