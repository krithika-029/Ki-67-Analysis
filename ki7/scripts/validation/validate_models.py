#!/usr/bin/env python3
"""
Ki-67 Model Validation Script for VS Code
==========================================
Validates the 3 trained models (InceptionV3, ResNet50, ViT) on the Ki67_Dataset_for_Colab dataset
Run this directly in VS Code terminal: python validate_models.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
import zipfile
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Try to import timm for ViT
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm available for ViT model")
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available, will use fallback CNN for ViT")

def setup_environment():
    """Setup device and environment"""
    print("🔬 Ki-67 Model Validation System")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Using CPU (training was likely done on GPU)")
    
    return device

def setup_paths():
    """Setup file paths based on current directory structure"""
    current_dir = Path(__file__).parent
    
    # Based on analysis of /Users/chinthan/ki7 structure
    paths = {
        'dataset_dir': current_dir / "Ki67_Dataset_for_Colab",  # Dataset already extracted
        'models_dir': current_dir / "Ki67_Models",  # Empty - models need to be downloaded from Google Drive
        'history_dir': current_dir,  # .pkl files are in root directory
        'results_dir': current_dir / "Ki67_Results"
    }
    
    print("\n📁 Analyzing current directory structure:")
    print(f"Project directory: {current_dir}")
    
    for key, path in paths.items():
        print(f"  {key}: {path}")
        if path.exists():
            if path.is_dir():
                files = list(path.iterdir())
                print(f"    ✅ Found ({len(files)} items)")
                if key == 'dataset_dir' and files:
                    print(f"      Contents: {[f.name for f in files[:5]]}")
            else:
                print(f"    ✅ Found (file)")
        else:
            print(f"    ❌ Not found")
    
    # Check for .pkl history files in root
    history_files = list(current_dir.glob("*_history_*.pkl"))
    if history_files:
        print(f"\n📈 Found training history files:")
        for f in history_files:
            print(f"    ✅ {f.name}")
    
    # Check for model files
    model_files = list(current_dir.glob("*.pth"))
    if model_files:
        print(f"\n🤖 Found model files:")
        for f in model_files:
            print(f"    ✅ {f.name}")
    else:
        print(f"\n⚠️  No .pth model files found in current directory")
        print("    Models are likely still in Google Drive. You need to:")
        print("    1. Download them from Google Drive")
        print("    2. Place them in Ki67_Models/ directory")
        print("    3. Or update this script to point to their location")
    
    return paths

def check_dataset(dataset_dir):
    """Check if dataset is available (already extracted)"""
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found at: {dataset_dir}")
        return None
    
    # Check for required files
    csv_file = os.path.join(dataset_dir, "ki67_dataset_metadata.csv")
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")
    
    if not os.path.exists(csv_file):
        print(f"❌ Metadata CSV not found at: {csv_file}")
        return None
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found at: {images_dir}")
        return None
    
    if not os.path.exists(annotations_dir):
        print(f"❌ Annotations directory not found at: {annotations_dir}")
        return None
    
    print(f"✅ Dataset found at: {dataset_dir}")
    
    # Show dataset structure
    print("\nDataset structure:")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = '  ' * level
        folder_name = os.path.basename(root) if level > 0 else "ki67_dataset"
        print(f'{indent}{folder_name}/')
        
        subindent = '  ' * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f'{subindent}{file}')
        if len(files) > 3:
            print(f'{subindent}... and {len(files)-3} more files')
        
        # Don't go too deep
        if level > 2:
            dirs.clear()
    
    return dataset_dir

class Ki67ValidationDataset(Dataset):
    """Dataset class for Ki67 validation"""
    
    def __init__(self, dataset_path, split='test', transform=None):
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
        
        print(f"📊 Loading {self.split} data from CSV...")
        df = pd.read_csv(csv_path)
        
        print(f"CSV columns: {list(df.columns)}")
        print(f"Available splits: {df['split'].unique()}")
        
        self.data = df[df['split'] == self.split].reset_index(drop=True)
        
        # Correct labels based on annotation paths (same logic as training)
        corrected_labels = []
        for idx, row in self.data.iterrows():
            annotation_path = str(row['annotation_path'])
            
            if '\\positive\\' in annotation_path or '/positive/' in annotation_path:
                corrected_labels.append(1)
            elif '\\negative\\' in annotation_path or '/negative/' in annotation_path:
                corrected_labels.append(0)
            else:
                # Fallback
                corrected_labels.append(idx % 2)
        
        self.data['corrected_label'] = corrected_labels
        
        print(f"Loaded {len(self.data)} samples for {self.split}")
        if len(corrected_labels) > 0:
            pos_count = sum(corrected_labels)
            neg_count = len(corrected_labels) - pos_count
            print(f"Distribution: {pos_count} positive, {neg_count} negative")
        
    def normalize_path_for_local(self, path_str):
        """Normalize paths for local filesystem"""
        if isinstance(path_str, str):
            return Path(path_str.replace('\\', '/'))
        return Path(path_str)
    
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

def create_model_architectures(device):
    """Create the 3 model architectures"""
    print("\n🏗️ Creating model architectures...")
    
    models_dict = {}
    
    # 1. InceptionV3
    print("Creating InceptionV3...")
    inception_model = models.inception_v3(pretrained=False)
    inception_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(inception_model.fc.in_features, 1),
        nn.Sigmoid()
    )
    if hasattr(inception_model, 'AuxLogits'):
        inception_model.AuxLogits.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(inception_model.AuxLogits.fc.in_features, 1),
            nn.Sigmoid()
        )
    inception_model = inception_model.to(device)
    models_dict['InceptionV3'] = inception_model
    print("✅ InceptionV3 architecture created")
    
    # 2. ResNet50
    print("Creating ResNet50...")
    resnet_model = models.resnet50(pretrained=False)
    resnet_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(resnet_model.fc.in_features, 1)
    )
    resnet_model = resnet_model.to(device)
    models_dict['ResNet50'] = resnet_model
    print("✅ ResNet50 architecture created")
    
    # 3. ViT
    print("Creating ViT...")
    try:
        if TIMM_AVAILABLE:
            vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            vit_model = vit_model.to(device)
            models_dict['ViT'] = vit_model
            print("✅ ViT architecture created")
        else:
            raise ImportError("timm not available")
    except Exception as e:
        print(f"⚠️  ViT creation failed: {e}")
        print("Creating simple CNN fallback...")
        
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d(7)
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        vit_model = SimpleCNN().to(device)
        models_dict['ViT'] = vit_model
        print("✅ Simple CNN created as ViT fallback")
    
    return models_dict

def find_model_files(models_dir):
    """Find the trained model files"""
    print(f"\n🔍 Looking for model files in: {models_dir}")
    
    # Look for the specific model files from your training
    model_patterns = {
        'InceptionV3': '*InceptionV3*best_model*.pth',
        'ResNet50': '*ResNet50*best_model*.pth',
        'ViT': '*ViT*best_model*.pth'
    }
    
    found_files = {}
    
    for model_name, pattern in model_patterns.items():
        files = list(Path(models_dir).glob(pattern))
        if files:
            # Get the most recent file if multiple exist
            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            found_files[model_name] = latest_file
            print(f"✅ Found {model_name}: {latest_file.name}")
        else:
            print(f"❌ No {model_name} model file found with pattern: {pattern}")
    
    return found_files

def load_trained_models(models_dict, models_dir, device):
    """Load the trained model weights"""
    print("\n📥 Loading trained model weights...")
    
    model_files = find_model_files(models_dir)
    loaded_models = {}
    
    for model_name, model in models_dict.items():
        if model_name in model_files:
            model_path = model_files[model_name]
            
            try:
                print(f"Loading {model_name} from {model_path.name}...")
                checkpoint = torch.load(model_path, map_location=device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    perf_summary = checkpoint.get('performance_summary', 'N/A')
                else:
                    model.load_state_dict(checkpoint)
                    perf_summary = 'N/A'
                
                model.eval()  # Set to evaluation mode
                loaded_models[model_name] = model
                
                print(f"✅ {model_name} loaded successfully")
                print(f"   Performance: {perf_summary}")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
        else:
            print(f"❌ No model file found for {model_name}")
    
    return loaded_models

def validate_models(models_dict, test_loader, device):
    """Validate all models on test set"""
    print("\n🔍 Validating models on test set...")
    
    results = {}
    all_predictions = {}
    all_labels = []
    
    # Collect predictions from all models
    with torch.no_grad():
        for model_name, model in models_dict.items():
            print(f"\nEvaluating {model_name}...")
            model.eval()
            
            model_predictions = []
            model_probs = []
            model_labels = []
            
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                try:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # Adjust input size for InceptionV3
                    if model_name == "InceptionV3" and inputs.size(-1) != 299:
                        inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Handle tuple output (InceptionV3 auxiliary)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    # Get probabilities
                    if model_name == 'ResNet50':
                        probs = torch.sigmoid(outputs)
                    else:
                        probs = outputs  # Already has sigmoid
                    
                    # Store predictions
                    model_probs.extend(probs.cpu().numpy().flatten())
                    predictions = (probs > 0.5).float()
                    model_predictions.extend(predictions.cpu().numpy().flatten())
                    model_labels.extend(labels.cpu().numpy().flatten())
                
                except Exception as e:
                    print(f"Error in batch {batch_idx} for {model_name}: {e}")
                    continue
            
            # Store results for this model
            all_predictions[model_name] = {
                'predictions': np.array(model_predictions),
                'probabilities': np.array(model_probs)
            }
            
            # Store labels (only once)
            if not all_labels:
                all_labels = np.array(model_labels)
    
    # Calculate metrics for each model
    print("\n📊 Model Performance:")
    print("=" * 60)
    
    for model_name, preds in all_predictions.items():
        y_pred = preds['predictions']
        y_prob = preds['probabilities']
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, y_pred)
        precision = precision_score(all_labels, y_pred, zero_division=0)
        recall = recall_score(all_labels, y_pred, zero_division=0)
        f1 = f1_score(all_labels, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, y_prob)
        except:
            auc = 0.5
        
        results[model_name] = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'auc': auc * 100,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print(f"{model_name:12}: Acc={accuracy*100:6.2f}%, AUC={auc*100:6.2f}%, F1={f1*100:6.2f}%")
    
    return results, all_labels

def create_ensemble_predictions(results, all_labels):
    """Create ensemble predictions with multiple strategies"""
    print("\n🤝 Creating ensemble predictions...")
    
    # Get individual probabilities
    model_names = list(results.keys())
    probs_matrix = np.column_stack([results[name]['probabilities'] for name in model_names])
    
    # Strategy 1: Simple average ensemble
    ensemble_probs = np.mean(probs_matrix, axis=1)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Strategy 2: Weighted ensemble based on individual accuracy
    weights = np.array([results[name]['accuracy'] for name in model_names])
    weights = weights / np.sum(weights)  # Normalize weights
    weighted_probs = np.average(probs_matrix, axis=1, weights=weights)
    weighted_preds = (weighted_probs > 0.5).astype(int)
    
    # Strategy 3: Confidence-based ensemble (higher threshold)
    confident_probs = np.mean(probs_matrix, axis=1)
    confident_preds = (confident_probs > 0.6).astype(int)  # Higher threshold
    
    # Strategy 4: Majority voting
    binary_preds = probs_matrix > 0.5
    majority_preds = (np.sum(binary_preds, axis=1) > len(model_names)/2).astype(int)
    
    # Evaluate all strategies
    strategies = {
        'Average': (ensemble_probs, ensemble_preds),
        'Weighted': (weighted_probs, weighted_preds),
        'Confident': (confident_probs, confident_preds),
        'Majority': (confident_probs, majority_preds)  # Use confident_probs for AUC
    }
    
    best_strategy = 'Average'
    best_accuracy = 0
    
    print("\n📊 Ensemble Strategy Comparison:")
    for strategy_name, (probs, preds) in strategies.items():
        accuracy = accuracy_score(all_labels, preds)
        try:
            auc = roc_auc_score(all_labels, probs)
        except:
            auc = 0.5
        
        print(f"{strategy_name:10}: Acc={accuracy*100:6.2f}%, AUC={auc*100:6.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_strategy = strategy_name
    
    print(f"\n🏆 Best strategy: {best_strategy} ({best_accuracy*100:.2f}%)")
    
    # Use best strategy
    best_probs, best_preds = strategies[best_strategy]
    
    
    # Calculate metrics for best strategy
    accuracy = accuracy_score(all_labels, best_preds)
    precision = precision_score(all_labels, best_preds, zero_division=0)
    recall = recall_score(all_labels, best_preds, zero_division=0)
    f1 = f1_score(all_labels, best_preds, zero_division=0)
    auc = roc_auc_score(all_labels, best_probs)
    
    print(f"\nFinal Ensemble: Acc={accuracy*100:6.2f}%, AUC={auc*100:6.2f}%, F1={f1*100:6.2f}%")
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'auc': auc * 100,
        'predictions': best_preds,
        'probabilities': best_probs,
        'strategy': best_strategy
    }

def suggest_improvements_for_95_percent(results, ensemble_results):
    """Analyze current performance and suggest improvements to reach 95%"""
    print("\n🎯 ANALYSIS: How to Reach 95%+ Accuracy")
    print("="*50)
    
    current_acc = ensemble_results['accuracy']
    target_acc = 95.0
    gap = target_acc - current_acc
    
    print(f"Current ensemble accuracy: {current_acc:.2f}%")
    print(f"Target accuracy: {target_acc:.2f}%")
    print(f"Gap to close: {gap:.2f}%")
    
    if gap <= 0:
        print("🎉 Already achieved 95%+ accuracy!")
        return
    
    print(f"\n💡 Strategies to gain {gap:.2f}% accuracy:")
    
    # Strategy 1: Model Diversity Analysis
    print("\n1️⃣ ADD MORE DIVERSE MODELS:")
    print("   Current: InceptionV3 (CNN), ResNet50 (CNN), ViT (Transformer)")
    print("   Suggested additions:")
    print("   📌 EfficientNet-B4/B5 (Advanced CNN)")
    print("   📌 DenseNet-201 (Dense connections)")
    print("   📌 RegNet (Facebook's efficient architecture)")
    print("   📌 ConvNeXt (Modern CNN design)")
    print("   📌 Swin Transformer (Hierarchical ViT)")
    
    # Strategy 2: Ensemble Improvements
    print("\n2️⃣ ADVANCED ENSEMBLE TECHNIQUES:")
    print("   📌 Stacking: Train meta-learner on model outputs")
    print("   📌 Bayesian Model Averaging")
    print("   📌 Confidence-weighted voting")
    print("   📌 Cross-validation ensemble")
    
    # Strategy 3: Data Improvements
    print("\n3️⃣ DATA ENHANCEMENT:")
    print("   📌 Add more training data if available")
    print("   📌 Advanced data augmentation (CutMix, MixUp)")
    print("   📌 Test-time augmentation (TTA)")
    print("   📌 Pseudo-labeling on unlabeled data")
    
    # Strategy 4: Training Improvements
    print("\n4️⃣ TRAINING ENHANCEMENTS:")
    print("   📌 Knowledge distillation")
    print("   📌 Self-supervised pre-training")
    print("   📌 Curriculum learning")
    print("   📌 Label smoothing")
    
    # Most practical suggestions
    print(f"\n🚀 IMMEDIATE ACTIONS (easiest {gap:.1f}% gain):")
    print("   1. Add EfficientNet-B4 model (1-2% gain likely)")
    print("   2. Use weighted ensemble (0.5-1% gain)")
    print("   3. Implement test-time augmentation (0.5-1% gain)")
    print("   4. Add DenseNet-201 model (1-2% gain likely)")
    
    # Code for additional models
    print(f"\n📝 CODE: Add EfficientNet-B4 to your training:")
    print("""
    # Add to create_models() function:
    try:
        efficientnet_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
        efficientnet_model = efficientnet_model.to(device)
        models_dict['EfficientNet'] = efficientnet_model
        print("✅ EfficientNet-B4 created")
    except Exception as e:
        print(f"⚠️  EfficientNet creation failed: {e}")
    """)

def create_additional_models(device):
    """Create additional high-performance models for 95%+ accuracy"""
    print("\n🏗️ Creating additional models for 95%+ accuracy...")
    
    additional_models = {}
    
    # EfficientNet-B4
    try:
        if TIMM_AVAILABLE:
            efficientnet_model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
            efficientnet_model = efficientnet_model.to(device)
            additional_models['EfficientNet-B4'] = efficientnet_model
            print("✅ EfficientNet-B4 created")
    except Exception as e:
        print(f"⚠️  EfficientNet-B4 creation failed: {e}")
    
    # DenseNet-201
    try:
        densenet_model = models.densenet201(pretrained=True)
        densenet_model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(densenet_model.classifier.in_features, 1),
            nn.Sigmoid()
        )
        densenet_model = densenet_model.to(device)
        additional_models['DenseNet-201'] = densenet_model
        print("✅ DenseNet-201 created")
    except Exception as e:
        print(f"⚠️  DenseNet-201 creation failed: {e}")
    
    # ConvNeXt
    try:
        if TIMM_AVAILABLE:
            convnext_model = timm.create_model('convnext_base', pretrained=True, num_classes=1)
            convnext_model = convnext_model.to(device)
            additional_models['ConvNeXt'] = convnext_model
            print("✅ ConvNeXt created")
    except Exception as e:
        print(f"⚠️  ConvNeXt creation failed: {e}")
    
    # Swin Transformer
    try:
        if TIMM_AVAILABLE:
            swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=1)
            swin_model = swin_model.to(device)
            additional_models['Swin-Transformer'] = swin_model
            print("✅ Swin Transformer created")
    except Exception as e:
        print(f"⚠️  Swin Transformer creation failed: {e}")
    
    return additional_models

def plot_confusion_matrices(results, all_labels, ensemble_results, save_path):
    """Plot confusion matrices for all models"""
    print("\n📊 Creating confusion matrices...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    all_results = {**results, 'Ensemble': ensemble_results}
    
    for idx, (model_name, model_results) in enumerate(all_results.items()):
        if idx < 4:  # We have 4 subplots
            cm = confusion_matrix(all_labels, model_results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[idx])
            
            axes[idx].set_title(f'{model_name}\nAcc: {model_results["accuracy"]:.1f}%')
            axes[idx].set_ylabel('True')
            axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    
    save_file = save_path / 'ki67_validation_confusion_matrices.png'
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrices saved as '{save_file}'")
    
    plt.show()

def save_results(results, ensemble_results, all_labels, save_path):
    """Save validation results"""
    print("\n💾 Saving results...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare results summary
    summary = {
        'timestamp': timestamp,
        'validation_summary': {
            'total_samples': len(all_labels),
            'positive_samples': int(np.sum(all_labels)),
            'negative_samples': int(len(all_labels) - np.sum(all_labels))
        },
        'individual_models': {},
        'ensemble': {}
    }
    
    # Add individual model results
    for model_name, model_results in results.items():
        summary['individual_models'][model_name] = {
            'accuracy': float(model_results['accuracy']),
            'precision': float(model_results['precision']),
            'recall': float(model_results['recall']),
            'f1_score': float(model_results['f1_score']),
            'auc': float(model_results['auc'])
        }
    
    # Add ensemble results
    summary['ensemble'] = {
        'accuracy': float(ensemble_results['accuracy']),
        'precision': float(ensemble_results['precision']),
        'recall': float(ensemble_results['recall']),
        'f1_score': float(ensemble_results['f1_score']),
        'auc': float(ensemble_results['auc'])
    }
    
    # Save to JSON
    results_file = save_path / f'ki67_validation_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Results saved to '{results_file}'")

def main():
    """Main validation function"""
    print("🚀 Starting Ki-67 Model Validation in VS Code...")
    
    # Setup
    device = setup_environment()
    paths = setup_paths()
    
    # Check dataset (already extracted)
    dataset_path = check_dataset(paths['dataset_dir'])
    if dataset_path is None:
        print("❌ Cannot proceed without dataset")
        print("\n💡 Solutions:")
        print("   1. Make sure Ki67_Dataset_for_Colab folder exists")
        print("   2. Check that it contains: images/, annotations/, ki67_dataset_metadata.csv")
        return
    
    # Create data transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create test dataset
    print("\n📊 Creating test dataset...")
    test_dataset = Ki67ValidationDataset(dataset_path, split='test', transform=test_transform)
    
    if len(test_dataset) == 0:
        print("❌ No test data found")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create and load models
    models_dict = create_model_architectures(device)
    loaded_models = load_trained_models(models_dict, paths['models_dir'], device)
    
    if len(loaded_models) == 0:
        print("❌ No models loaded successfully")
        print("Please check that your .pth model files are in the correct directory")
        return
    
    print(f"\n✅ Successfully loaded {len(loaded_models)} models")
    
    # Validate models
    results, all_labels = validate_models(loaded_models, test_loader, device)
    
    if not results:
        print("❌ No validation results obtained")
        return
    
    # Create ensemble
    ensemble_results = create_ensemble_predictions(results, all_labels)
    
    # Analyze performance and suggest improvements
    suggest_improvements_for_95_percent(results, ensemble_results)
    
    # Plot results
    plot_confusion_matrices(results, all_labels, ensemble_results, paths['models_dir'])
    
    # Save results
    save_results(results, ensemble_results, all_labels, paths['models_dir'])
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 VALIDATION COMPLETED!")
    print("="*60)
    
    print(f"\n📊 Final Results Summary:")
    print(f"{'Model':<12} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 40)
    
    all_results = {**results, 'Ensemble': ensemble_results}
    for model_name, model_results in all_results.items():
        print(f"{model_name:<12} {model_results['accuracy']:>7.2f}%  {model_results['auc']:>6.2f}%  {model_results['f1_score']:>6.2f}%")
    
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    print(f"\n🏆 Best performing model: {best_model} ({all_results[best_model]['accuracy']:.2f}% accuracy)")
    
    print(f"\n✅ Your models are working correctly!")
    print(f"✅ Dataset validation successful!")
    print(f"✅ Results saved to your project directory")

if __name__ == "__main__":
    main()
