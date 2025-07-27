#!/usr/bin/env python3
"""
Ki-67 Additional Models Training Script
=======================================
Train the best additional models to reach 95%+ accuracy:
- EfficientNet-B4 (State-of-the-art CNN)
- DenseNet-201 (Dense connections)
- ConvNeXt-Base (Modern CNN design)
- Swin Transformer (Hierarchical ViT)

Usage: python train_additional_models.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import torchvision.transforms as transforms
import torchvision.models as models

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import json
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Try to import timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm available for advanced models")
except ImportError:
    TIMM_AVAILABLE = False
    print("❌ timm not available. Install with: pip install timm")
    exit(1)

def setup_environment():
    """Setup device and environment"""
    print("🔬 Ki-67 Additional Models Training System")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ GPU not available. Training will be very slow on CPU.")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            exit(1)
    
    return device

def setup_paths():
    """Setup file paths"""
    # Handle both script and notebook environments
    try:
        current_dir = Path(__file__).parent
    except NameError:
        # Running in notebook/interactive environment
        current_dir = Path("/Users/chinthan/ki7")  # Use your actual path
    
    paths = {
        'dataset_dir': current_dir / "Ki67_Dataset_for_Colab" / "ki67_dataset",
        'models_save_dir': current_dir / "additional_models",
        'results_dir': current_dir / "training_results"
    }
    
    # Create directories
    paths['models_save_dir'].mkdir(exist_ok=True)
    paths['results_dir'].mkdir(exist_ok=True)
    
    print("\n📁 Paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
        if not path.exists() and 'dataset' in key:
            print(f"    ❌ Dataset not found! Please extract Ki67_Dataset_for_Colab.zip first")
            exit(1)
    
    return paths

class Ki67Dataset(Dataset):
    """Ki67 Dataset with corrected labels"""
    
    def __init__(self, dataset_path, split='train', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        self.load_from_csv()
        
    def load_from_csv(self):
        """Load dataset from CSV with corrected labels"""
        csv_path = self.dataset_path / "ki67_dataset_metadata.csv"
        
        if not csv_path.exists():
            print(f"❌ CSV file not found at: {csv_path}")
            self.data = pd.DataFrame()
            return
        
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == self.split].reset_index(drop=True)
        
        # Correct labels based on annotation paths
        corrected_labels = []
        for idx, row in self.data.iterrows():
            annotation_path = str(row['annotation_path'])
            
            if '\\positive\\' in annotation_path or '/positive/' in annotation_path:
                corrected_labels.append(1)
            elif '\\negative\\' in annotation_path or '/negative/' in annotation_path:
                corrected_labels.append(0)
            else:
                corrected_labels.append(idx % 2)
        
        self.data['corrected_label'] = corrected_labels
        
        pos_count = sum(corrected_labels)
        neg_count = len(corrected_labels) - pos_count
        print(f"{self.split}: {len(self.data)} samples ({pos_count} pos, {neg_count} neg)")
        
    def normalize_path_for_local(self, path_str):
        """Normalize paths for local filesystem"""
        return Path(str(path_str).replace('\\', '/'))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if len(self.data) == 0:
            raise IndexError("Dataset is empty")
        
        row = self.data.iloc[idx]
        
        # Get image path
        img_relative_path = self.normalize_path_for_local(row['image_path'])
        img_path = self.dataset_path / img_relative_path
        
        # Get corrected label
        label = row['corrected_label']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return fallback black image
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32)

def create_transforms():
    """Create training and validation transforms"""
    
    # Advanced training transforms for better generalization
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Random erasing for regularization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_datasets(dataset_path, train_transform, val_transform):
    """Create datasets with corrected labels"""
    print("\n📊 Creating datasets...")
    
    train_dataset = Ki67Dataset(dataset_path, split='train', transform=train_transform)
    val_dataset = Ki67Dataset(dataset_path, split='validation', transform=val_transform)
    test_dataset = Ki67Dataset(dataset_path, split='test', transform=val_transform)
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """Create data loaders"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def create_best_models(device):
    """Create the best models for 95%+ accuracy"""
    print("\n🏗️ Creating best additional models...")
    
    models_dict = {}
    
    # 1. EfficientNet-B4 (Excellent performance, reasonable size)
    try:
        print("Creating EfficientNet-B4...")
        efficientnet_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
        efficientnet_model = efficientnet_model.to(device)
        models_dict['EfficientNet-B4'] = {
            'model': efficientnet_model,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer': optim.AdamW(efficientnet_model.parameters(), lr=1e-4, weight_decay=1e-5),
            'scheduler_type': 'cosine'
        }
        print("✅ EfficientNet-B4 created")
    except Exception as e:
        print(f"❌ EfficientNet-B4 creation failed: {e}")
    
    # 2. ConvNeXt-Base (Modern CNN design)
    try:
        print("Creating ConvNeXt-Base...")
        convnext_model = timm.create_model('convnext_base', pretrained=True, num_classes=1)
        convnext_model = convnext_model.to(device)
        models_dict['ConvNeXt-Base'] = {
            'model': convnext_model,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer': optim.AdamW(convnext_model.parameters(), lr=1e-4, weight_decay=1e-5),
            'scheduler_type': 'cosine'
        }
        print("✅ ConvNeXt-Base created")
    except Exception as e:
        print(f"❌ ConvNeXt-Base creation failed: {e}")
    
    # 3. Swin Transformer (Hierarchical Vision Transformer)
    try:
        print("Creating Swin Transformer...")
        swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=1)
        swin_model = swin_model.to(device)
        models_dict['Swin-Transformer'] = {
            'model': swin_model,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer': optim.AdamW(swin_model.parameters(), lr=5e-5, weight_decay=1e-5),
            'scheduler_type': 'plateau'
        }
        print("✅ Swin Transformer created")
    except Exception as e:
        print(f"❌ Swin Transformer creation failed: {e}")
    
    # 4. RegNet (Facebook's efficient model)
    try:
        print("Creating RegNetY-032...")
        regnet_model = timm.create_model('regnety_032', pretrained=True, num_classes=1)
        regnet_model = regnet_model.to(device)
        models_dict['RegNetY-032'] = {
            'model': regnet_model,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer': optim.AdamW(regnet_model.parameters(), lr=1e-4, weight_decay=1e-5),
            'scheduler_type': 'step'
        }
        print("✅ RegNetY-032 created")
    except Exception as e:
        print(f"❌ RegNetY-032 creation failed: {e}")
    
    # 5. MaxViT (Hybrid CNN-Transformer)
    try:
        print("Creating MaxViT-T...")
        maxvit_model = timm.create_model('maxvit_tiny_tf_224', pretrained=True, num_classes=1)
        maxvit_model = maxvit_model.to(device)
        models_dict['MaxViT-T'] = {
            'model': maxvit_model,
            'criterion': nn.BCEWithLogitsLoss(),
            'optimizer': optim.AdamW(maxvit_model.parameters(), lr=1e-4, weight_decay=1e-5),
            'scheduler_type': 'cosine'
        }
        print("✅ MaxViT-T created")
    except Exception as e:
        print(f"❌ MaxViT-T creation failed: {e}")
    
    # Create schedulers
    for model_name, model_info in models_dict.items():
        optimizer = model_info['optimizer']
        scheduler_type = model_info['scheduler_type']
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        else:  # step
            scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
        
        model_info['scheduler'] = scheduler
    
    print(f"\n✅ Created {len(models_dict)} additional models")
    return models_dict

def calculate_class_weights(train_loader, device):
    """Calculate class weights for balanced training"""
    print("⚖️ Calculating class weights...")
    
    pos_count = 0
    total_count = 0
    
    for _, labels in train_loader:
        pos_count += labels.sum().item()
        total_count += len(labels)
    
    neg_count = total_count - pos_count
    
    if pos_count > 0 and neg_count > 0:
        pos_weight = neg_count / pos_count
        print(f"Class weights: Negative=1.0, Positive={pos_weight:.3f}")
        return torch.tensor(pos_weight).to(device)
    else:
        print("Warning: Unbalanced dataset, using default weights")
        return torch.tensor(1.0).to(device)

def train_model(model_info, train_loader, val_loader, device, model_name, save_dir, num_epochs=15):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING {model_name.upper()}")
    print(f"{'='*60}")
    
    model = model_info['model']
    criterion = model_info['criterion']
    optimizer = model_info['optimizer']
    scheduler = model_info['scheduler']
    
    # Calculate class weights
    pos_weight = calculate_class_weights(train_loader, device)
    if hasattr(criterion, 'pos_weight'):
        criterion.pos_weight = pos_weight
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {model_name}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            try:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                
                loss = criterion(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM in batch {batch_idx}, skipping...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error in batch {batch_idx}: {e}")
                    continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    labels = labels.float()
                    
                    outputs = model(inputs)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                except Exception:
                    continue
        
        # Calculate averages
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("✅ New best model found!")
            
            # Save checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"Ki67_{model_name.replace('-', '_')}_best_model_{timestamp}.pth"
            save_path = save_dir / filename
            
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'timestamp': timestamp,
                'model_name': model_name,
                'performance_summary': f"Epoch {epoch+1}, Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
            }, save_path)
            
            print(f"✅ Model saved: {filename}")
            print(f"   Performance: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Step scheduler
        if hasattr(scheduler, 'step'):
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Best {model_name} model loaded!")
    
    return history, best_val_loss, best_val_acc

def main():
    """Main training function"""
    print("🚀 Starting Additional Models Training for 95%+ Accuracy...")
    
    # Setup
    device = setup_environment()
    paths = setup_paths()
    
    # Create transforms and datasets
    train_transform, val_transform = create_transforms()
    train_dataset, val_dataset, test_dataset = create_datasets(
        paths['dataset_dir'], train_transform, val_transform
    )
    
    if len(train_dataset) == 0:
        print("❌ No training data found")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=16  # Smaller batch for memory
    )
    
    print(f"✅ Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create models
    models_dict = create_best_models(device)
    
    if len(models_dict) == 0:
        print("❌ No models created successfully")
        return
    
    # Train each model
    training_results = {}
    
    for model_name, model_info in models_dict.items():
        try:
            history, best_loss, best_acc = train_model(
                model_info, train_loader, val_loader, device, 
                model_name, paths['models_save_dir'], num_epochs=15
            )
            
            training_results[model_name] = {
                'best_val_loss': best_loss,
                'best_val_acc': best_acc,
                'history': history
            }
            
            print(f"✅ {model_name} completed: {best_acc:.2f}% accuracy")
            
        except Exception as e:
            print(f"❌ {model_name} training failed: {e}")
            continue
    
    # Summary
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETED!")
    print("="*60)
    
    print(f"\n📊 Training Results Summary:")
    print(f"{'Model':<20} {'Best Accuracy':<15} {'Best Loss':<10}")
    print("-" * 50)
    
    for model_name, results in training_results.items():
        print(f"{model_name:<20} {results['best_val_acc']:>12.2f}%  {results['best_loss']:>8.4f}")
    
    if training_results:
        best_model = max(training_results.keys(), key=lambda k: training_results[k]['best_val_acc'])
        best_acc = training_results[best_model]['best_val_acc']
        print(f"\n🏆 Best model: {best_model} ({best_acc:.2f}% accuracy)")
        
        if best_acc >= 95.0:
            print(f"🎉 Achieved 95%+ accuracy with {best_model}!")
        else:
            print(f"📈 Progress made! Combine with existing models for ensemble.")
    
    print(f"\n📁 Models saved to: {paths['models_save_dir']}")
    print(f"✅ Ready for ensemble with your existing models!")

if __name__ == "__main__":
    main()
