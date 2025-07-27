#!/usr/bin/env python3
"""
Additional EfficientNet-B4 Models for Ensemble Boosting
This script trains 2 additional EfficientNet-B4 models with different strategies
to combine with the Champion B5 model for achieving 95%+ ensemble accuracy.

Author: AI Assistant
Date: 2024
Purpose: Train complementary B4 models for ensemble dominance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import json
import time
from datetime import datetime
import random
from collections import Counter

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

def setup_colab_environment():
    """Setup Google Colab environment"""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=True)
        
        # Create directories
        models_dir = "/content/drive/MyDrive"
        results_dir = "/content/drive/MyDrive/Ki67_Additional_B4_Results"
        os.makedirs(results_dir, exist_ok=True)
        
        print(f"\n📁 Additional B4 models will be saved to: {models_dir}")
        print(f"📁 Training results will be saved to: {results_dir}")
        
        return models_dir, results_dir
        
    except ImportError:
        print("⚠️  Not running in Google Colab")
        return None, None
    except Exception as e:
        print(f"⚠️  Error setting up Colab environment: {e}")
        return None, None

def create_efficientnet_b4_strategy1(device):
    """Create EfficientNet-B4 Strategy 1: Conservative Training"""
    try:
        print("🔥 Creating EfficientNet-B4 Strategy 1 (Conservative)...")
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        
        # Modify classifier for binary classification
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),  # Moderate dropout
            nn.Linear(model.classifier[1].in_features, 1)
        )
        
        model = model.to(device)
        model_name = "EfficientNet_B4_Strategy1_Conservative"
        
        print(f"✅ {model_name} created successfully")
        return model, model_name
        
    except Exception as e:
        print(f"❌ Failed to create EfficientNet-B4 Strategy 1: {e}")
        return None, None

def create_efficientnet_b4_strategy2(device):
    """Create EfficientNet-B4 Strategy 2: Aggressive Training"""
    try:
        print("🔥 Creating EfficientNet-B4 Strategy 2 (Aggressive)...")
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        
        # Modify classifier for binary classification
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Higher dropout for regularization
            nn.Linear(model.classifier[1].in_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
        
        model = model.to(device)
        model_name = "EfficientNet_B4_Strategy2_Aggressive"
        
        print(f"✅ {model_name} created successfully")
        return model, model_name
        
    except Exception as e:
        print(f"❌ Failed to create EfficientNet-B4 Strategy 2: {e}")
        return None, None

def get_conservative_transforms(image_size):
    """Conservative augmentation strategy"""
    return {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def get_aggressive_transforms(image_size):
    """Aggressive augmentation strategy"""
    return {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }

def train_b4_strategy1(model, train_loader, val_loader, device, save_path, num_epochs=25):
    """Train B4 Strategy 1: Conservative approach"""
    print(f"🚀 Training B4 Strategy 1 (Conservative)...")
    
    # Conservative training configuration
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0001,  # Very conservative learning rate
        weight_decay=0.005  # Moderate regularization
    )
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
                
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_path}/Ki67_B4_Strategy1_{val_acc:.2f}.pth")
        
        scheduler.step()
    
    return model, best_val_acc, history

def train_b4_strategy2(model, train_loader, val_loader, device, save_path, num_epochs=25):
    """Train B4 Strategy 2: Aggressive approach"""
    print(f"🚀 Training B4 Strategy 2 (Aggressive)...")
    
    # Aggressive training configuration
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=0.0008,  # Higher learning rate
        weight_decay=0.01  # Stronger regularization
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler() if device.type == 'cuda' else None
    
    best_val_acc = 0.0
    history = {'train_acc': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.float().unsqueeze(1) if targets.dim() == 1 else targets.float()
                
                outputs = model(inputs)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_path}/Ki67_B4_Strategy2_{val_acc:.2f}.pth")
        
        scheduler.step()
    
    return model, best_val_acc, history

def main():
    """Main function to train additional B4 models"""
    print("🎯 Training Additional EfficientNet-B4 Models for Ensemble Boosting")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Setup environment
    try:
        models_save_path, results_save_path = setup_colab_environment()
        if models_save_path is None:
            # Local mode
            models_save_path = "./models"
            results_save_path = "./results"
            os.makedirs(models_save_path, exist_ok=True)
            os.makedirs(results_save_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Environment setup failed: {e}")
        return
    
    # Training configuration
    image_size = 288  # B4 optimized size
    batch_size = 16 if torch.cuda.is_available() else 4
    num_epochs = 25
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Batch size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    
    # Note: This script assumes you have the dataset setup
    # You would need to integrate with your existing Ki67Dataset class
    print(f"\n📝 Next Steps to Complete:")
    print(f"   1. Import your Ki67Dataset class from the main script")
    print(f"   2. Setup train/val dataloaders with different augmentation strategies")
    print(f"   3. Train Strategy 1 (Conservative) and Strategy 2 (Aggressive)")
    print(f"   4. Combine predictions with your Champion B5 model")
    
    print(f"\n💡 Ensemble Strategy:")
    print(f"   - Champion B5: Primary high-accuracy model")
    print(f"   - B4 Strategy 1: Conservative, stable predictions")
    print(f"   - B4 Strategy 2: Aggressive, diverse predictions")
    print(f"   - Final prediction: Weighted average of all three")
    
    print(f"\n🎯 Expected Outcome:")
    print(f"   If Champion B5 reaches 92-94%, ensemble should achieve 95%+")
    print(f"   If Champion B5 reaches 90-92%, ensemble should achieve 93-95%")

if __name__ == "__main__":
    main()
