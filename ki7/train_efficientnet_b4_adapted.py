#!/usr/bin/env python3
"""
Ki-67 EfficientNet-B4 Adapted Training Script - Incorporating B2 Success Strategies

This script adapts the successful EfficientNet-B2 configuration (92.5% accuracy) 
to EfficientNet-B4 for even better performance. Key adaptations:

1. Uses the PROVEN simple training approach from B2 script
2. Scales up to EfficientNet-B4 for higher capacity
3. Maintains the stable, conservative training strategies that worked
4. Removes complex augmentations that may have caused instability

Expected: 94%+ single model accuracy by combining B2 stability with B4 capacity

Usage in Google Colab:
    Upload this script and run: exec(open('train_efficientnet_b4_adapted.py').read())
"""

import os
import sys
import warnings
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random
import zipfile
import subprocess
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')

def install_package(package):
    """Install package with error handling"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print(f"✅ {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  {package} - may already be installed")
        return False

def setup_colab_packages():
    """Install required packages for Colab"""
    print("📦 Installing required packages for Ki-67 B4 adapted training...")
    
    packages = [
        "torch", "torchvision", "timm", "scikit-learn", 
        "matplotlib", "seaborn", "pandas", "numpy", "Pillow"
    ]
    
    for package in packages:
        install_package(package)
    
    print("\n🎯 Package installation completed!")

def setup_colab_environment():
    """Setup Google Colab environment with Drive mounting"""
    try:
        from google.colab import drive
        
        # Mount Google Drive
        print("📱 Mounting Google Drive...")
        drive.mount('/content/drive')
        
        # Check if drive is mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully!")
        else:
            print("❌ Failed to mount Google Drive")
            return None, None
        
        # Create directories
        models_dir = "/content/drive/MyDrive"  # Save models directly to MyDrive
        results_dir = "/content/drive/MyDrive/Ki67_B4_Adapted_Results"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n📁 B4 adapted model will be saved to: {models_dir}")
        print(f"📁 Training results will be saved to: {results_dir}")
        
        return models_dir, results_dir
        
    except ImportError:
        print("⚠️  Not running in Google Colab")
        return None, None
    except Exception as e:
        print(f"⚠️  Error setting up Colab environment: {e}")
        return None, None

def extract_dataset_from_drive():
    """Extract Ki67 dataset from Google Drive"""
    try:
        # Common locations for the dataset
        possible_locations = [
            "/content/drive/MyDrive/Ki67_Dataset_for_Colab.zip",
            "/content/drive/MyDrive/Ki67_Dataset_for_Colab/Ki67_Dataset_for_Colab.zip",
            "/content/drive/MyDrive/BCData.zip"
        ]
        
        dataset_zip = None
        for location in possible_locations:
            if os.path.exists(location):
                dataset_zip = location
                break
        
        if not dataset_zip:
            print("❌ Ki67 dataset not found in Google Drive")
            print("📋 Please upload Ki67_Dataset_for_Colab.zip to your Google Drive")
            return None
        
        print(f"📂 Found dataset: {dataset_zip}")
        
        # Extract to local storage
        extract_path = "/content/ki67_dataset"
        
        if os.path.exists(extract_path):
            print(f"📂 Dataset already extracted at {extract_path}")
        else:
            print(f"📦 Extracting dataset to {extract_path}...")
            with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
                zip_ref.extractall("/content")
            print("✅ Dataset extracted successfully!")
        
        return extract_path
        
    except Exception as e:
        print(f"❌ Error extracting dataset: {e}")
        return None

def setup_device():
    """Setup device and import libraries - simplified from B2 success"""
    global torch, nn, optim, transforms, models, timm
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torchvision.transforms as transforms
        import torchvision.models as models
        from torch.utils.data import Dataset, DataLoader
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
        import timm
        from PIL import Image
        
        print("✅ All libraries imported successfully")
    except ImportError as e:
        print(f"❌ Missing packages: {e}")
    
    # Set device and optimize for Colab
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🚀 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Optimize CUDA settings for Colab
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set memory management
        torch.cuda.empty_cache()
        
        # Check available memory
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🚀 GPU Memory - Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
    else:
        print("⚠️  No GPU available - training will be significantly slower")
    
    return device

# Use the same proven dataset class
class CorrectedKi67Dataset(Dataset):
    """Dataset that uses annotation file size analysis (proven approach)"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32)

def create_corrected_dataset_from_directories(dataset_path, split, transform=None):
    """Create dataset from directory structure - EXACT same approach as B2 success"""
    print(f"🔧 Creating corrected {split} dataset from directory structure...")
    
    dataset_path = Path(dataset_path)
    
    # Check multiple possible directory structures
    possible_structures = [
        # Standard BCData structure
        {
            'images': dataset_path / "BCData" / "images" / split,
            'pos_annotations': dataset_path / "BCData" / "annotations" / split / "positive",
            'neg_annotations': dataset_path / "BCData" / "annotations" / split / "negative"
        },
        # Alternative structure
        {
            'images': dataset_path / "images" / split,
            'pos_annotations': dataset_path / "annotations" / split / "positive",
            'neg_annotations': dataset_path / "annotations" / split / "negative"
        },
        # Check if ki67_dataset subdirectory exists
        {
            'images': dataset_path / "ki67_dataset" / "images" / split,
            'pos_annotations': dataset_path / "ki67_dataset" / "annotations" / split / "positive",
            'neg_annotations': dataset_path / "ki67_dataset" / "annotations" / split / "negative"
        }
    ]
    
    images = []
    labels = []
    
    for structure in possible_structures:
        images_dir = structure['images']
        pos_annotations_dir = structure['pos_annotations']
        neg_annotations_dir = structure['neg_annotations']
        
        if images_dir.exists():
            print(f"Found images directory: {images_dir}")
            
            # Get all image files
            for img_file in images_dir.glob("*.png"):
                img_name = img_file.stem
                
                # Check for corresponding annotation files
                pos_annotation = pos_annotations_dir / f"{img_name}.xml"
                neg_annotation = neg_annotations_dir / f"{img_name}.xml"
                
                if pos_annotation.exists():
                    # Check annotation file size
                    if os.path.getsize(pos_annotation) > 100:  # Non-empty annotation
                        images.append(str(img_file))
                        labels.append(1)  # Positive
                elif neg_annotation.exists():
                    images.append(str(img_file))
                    labels.append(0)  # Negative
            
            if len(images) > 0:
                print(f"✅ Found {len(images)} images in {images_dir}")
                break
    
    if len(images) == 0:
        raise ValueError(f"No valid images found for {split} split in {dataset_path}")
    
    print(f"✅ Loaded {len(images)} images for {split}")
    print(f"   Distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return CorrectedKi67Dataset(images, labels, transform)

def create_b2_adapted_transforms():
    """Create simple, proven transforms adapted from successful B2 configuration"""
    print("🖼️ Creating B2-adapted transforms for EfficientNet-B4...")
    
    # B4 needs slightly larger images than B2, but keep it reasonable
    image_size = 288  # Larger than B2's 224, smaller than your 384
    
    # Simple, proven augmentations from B2 success
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Conservative rotation
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"✅ Transforms created - Image size: {image_size}x{image_size}")
    print("✅ Using conservative augmentation strategy from B2 success")
    
    return train_transform, val_transform

def create_adapted_b4_model(device, num_classes=1):
    """Create EfficientNet-B4 with B2-inspired configuration"""
    print("🏗️ Creating EfficientNet-B4 with B2-adapted configuration...")
    
    try:
        # Use EfficientNet-B4 (higher capacity than B2)
        model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=num_classes)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✅ EfficientNet-B4 created: {total_params:,} parameters")
        
        return model, 'EfficientNet-B4-Adapted'
        
    except Exception as e:
        print(f"❌ EfficientNet-B4 creation failed: {e}")
        
        # Fallback to B3 if B4 fails
        try:
            print("🔄 Falling back to EfficientNet-B3...")
            model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=num_classes)
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✅ EfficientNet-B3 created: {total_params:,} parameters")
            
            return model, 'EfficientNet-B3-Adapted'
            
        except Exception as e2:
            print(f"❌ EfficientNet-B3 also failed: {e2}")
            raise e2

def calculate_class_weights(train_dataset, device):
    """Calculate class weights - same as B2 success"""
    labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        labels.append(int(label.item()))
    
    if len(labels) == 0:
        return torch.tensor([1.0, 1.0]).to(device)
    
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    
    if pos_count > 0 and neg_count > 0:
        pos_weight = len(labels) / (2 * pos_count)
        neg_weight = len(labels) / (2 * neg_count)
        pos_weight_ratio = pos_weight / neg_weight
    else:
        pos_weight_ratio = 1.0
    
    print(f"Class distribution: {pos_count} positive, {neg_count} negative")
    print(f"Positive weight ratio: {pos_weight_ratio:.3f}")
    
    return torch.tensor(pos_weight_ratio).to(device)

def train_b4_adapted_model(model, train_loader, val_loader, device, model_name, 
                          num_epochs=20, save_path="/content/drive/MyDrive"):
    """Train EfficientNet-B4 using proven B2 strategies"""
    print(f"🚀 Training {model_name} with B2-adapted strategies...")
    
    # Calculate class weights for loss function
    pos_weight = calculate_class_weights(train_loader.dataset, device)
    
    # B2-inspired configuration (PROVEN SUCCESSFUL)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Simple Adam optimizer like B2 success (lr=0.001, weight_decay=1e-4)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,  # Same as B2 success
        weight_decay=1e-4  # Same as B2 success
    )
    
    # Conservative scheduler like B2 success
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5  # Same as B2 success
    )
    
    # Mixed precision training for T4 efficiency (like B2)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training tracking
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'learning_rates': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 7  # Same as B2 success
    
    print(f"🎯 B4 Adapted Training Configuration:")
    print(f"   Model: {model_name}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: 0.001 (B2 proven)")
    print(f"   Weight Decay: 1e-4 (B2 proven)")
    print(f"   Scheduler: ReduceLROnPlateau (B2 proven)")
    print(f"   Early Stopping: {early_stopping_patience} epochs")
    print(f"   Mixed Precision: {'Enabled' if scaler else 'Disabled'}")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} - {model_name}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                targets = targets.float()
                
                optimizer.zero_grad()
                
                # Forward pass with mixed precision (if available)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        if outputs.dim() == 1:
                            outputs = outputs.unsqueeze(1)
                        loss = criterion(outputs, targets)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    loss = criterion(outputs, targets)
                    
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy (same as B2)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💾 GPU OOM in batch {batch_idx}, clearing cache...")
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error in batch {batch_idx}: {e}")
                    optimizer.zero_grad()
                    continue
        
        # Validation phase (same as B2)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                try:
                    inputs = inputs.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    
                    if targets.dim() == 1:
                        targets = targets.unsqueeze(1)
                    targets = targets.float()
                    
                    if scaler is not None:
                        with torch.cuda.amp.autocast():
                            outputs = model(inputs)
                            if outputs.dim() == 1:
                                outputs = outputs.unsqueeze(1)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = model(inputs)
                        if outputs.dim() == 1:
                            outputs = outputs.unsqueeze(1)
                        loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
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
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model (same logic as B2)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("✅ New best model found!")
            
            # Save model checkpoint
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"ki67_{model_name.lower()}_best_model_{timestamp}.pth"
            model_path = os.path.join(save_path, model_filename)
            
            try:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'model_name': model_name
                }, model_path)
                print(f"✅ Model saved: {model_filename}")
            except Exception as e:
                print(f"⚠️  Failed to save model: {e}")
        else:
            patience_counter += 1
        
        # Step scheduler (same as B2)
        scheduler.step(val_loss)
        
        # Early stopping (same as B2)
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Performance milestones
        if val_acc >= 90.0:
            print(f"🎯 MILESTONE: 90%+ validation accuracy achieved!")
        if val_acc >= 92.5:
            print(f"🏆 MILESTONE: 92.5%+ validation accuracy (B2 level)!")
        if val_acc >= 94.0:
            print(f"🚀 MILESTONE: 94%+ validation accuracy (B4 target)!")
        
        # T4 GPU memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Best {model_name} model loaded! Final Accuracy: {best_val_acc:.2f}%")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'model_name': model_name,
        'best_val_acc': best_val_acc,
        'training_history': history,
        'timestamp': timestamp,
        'configuration': {
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'scheduler': 'ReduceLROnPlateau',
            'epochs_trained': epoch + 1,
            'early_stopping_patience': early_stopping_patience
        }
    }
    
    results_filename = f"b4_adapted_results_{timestamp}.json"
    results_path = os.path.join(save_path, results_filename)
    
    try:
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ Results saved: {results_filename}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")
    
    return history, best_val_acc

def main_b4_adapted():
    """Main function for B4 adapted training"""
    print("🚀 Ki-67 EfficientNet-B4 Adapted Training (B2 Success Strategies)")
    print("=" * 80)
    
    # Setup environment
    if 'google.colab' in sys.modules:
        setup_colab_packages()
        models_dir, results_dir = setup_colab_environment()
        dataset_path = extract_dataset_from_drive()
        
        if not dataset_path:
            print("❌ Dataset setup failed")
            return
    else:
        print("⚠️  Running outside Google Colab")
        models_dir = "."
        results_dir = "./results"
        dataset_path = input("Enter path to Ki67 dataset: ").strip()
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset path does not exist: {dataset_path}")
            return
    
    # Setup device and imports
    device = setup_device()
    
    # Create transforms
    train_transform, val_transform = create_b2_adapted_transforms()
    
    # Load datasets
    try:
        print("\n📊 Loading datasets...")
        
        train_dataset = create_corrected_dataset_from_directories(dataset_path, "train", train_transform)
        val_dataset = create_corrected_dataset_from_directories(dataset_path, "val", val_transform)
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return
    
    # Create data loaders (B2-adapted batch sizes)
    if device.type == 'cuda':
        batch_size = 16  # Conservative for B4 on T4
    else:
        batch_size = 4   # CPU fallback
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"✅ Data loaders created - Batch size: {batch_size}")
    
    # Create model
    try:
        model, model_name = create_adapted_b4_model(device)
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return
    
    # Train model
    print(f"\n🚀 Starting B4 adapted training...")
    try:
        history, best_acc = train_b4_adapted_model(
            model, train_loader, val_loader, device, model_name,
            num_epochs=20, save_path=models_dir
        )
        
        print(f"\n🎉 B4 adapted training completed!")
        print(f"🏆 Best validation accuracy: {best_acc:.2f}%")
        
        if best_acc >= 92.5:
            print(f"✅ SUCCESS: Achieved B2-level performance (92.5%+) with B4!")
        if best_acc >= 94.0:
            print(f"🚀 OUTSTANDING: Exceeded B4 target (94%+)!")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

# Run the adapted B4 training
if __name__ == "__main__":
    main_b4_adapted()
