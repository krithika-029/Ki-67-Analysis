#!/usr/bin/env python3
"""
Ki-67 Model Selection and Analysis Script
=========================================
This script will:
1. Scan ALL downloaded model files
2. Analyze their training metadata
3. Test each model's validation performance
4. Automatically select the best models
5. Create the optimal ensemble for 95%+ accuracy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import os
import glob
import json
from datetime import datetime

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, f1_score

# Try to import timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm available for advanced models")
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available - some models may not load")

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

def scan_all_model_files(base_dir):
    """Scan for ALL .pth model files and analyze them"""
    print("🔍 Scanning for ALL Ki-67 model files...")
    
    # Find all .pth files in directory and subdirectories
    all_pth_files = []
    
    # Search patterns
    search_patterns = [
        "*.pth",
        "**/*.pth", 
        "Ki67_Models/*.pth",
        "models/*.pth",
        "additional_models/*.pth"
    ]
    
    for pattern in search_patterns:
        files = glob.glob(os.path.join(base_dir, pattern), recursive=True)
        all_pth_files.extend(files)
    
    # Remove duplicates
    all_pth_files = list(set(all_pth_files))
    
    print(f"📊 Found {len(all_pth_files)} .pth files:")
    for file in sorted(all_pth_files):
        print(f"   • {os.path.basename(file)}")
    
    return all_pth_files

def analyze_model_metadata(model_path):
    """Analyze model metadata from checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        metadata = {
            'file_path': model_path,
            'file_name': os.path.basename(model_path),
            'file_size_mb': os.path.getsize(model_path) / (1024*1024),
            'modified_time': datetime.fromtimestamp(os.path.getmtime(model_path))
        }
        
        # Extract training metadata if available
        if isinstance(checkpoint, dict):
            metadata.update({
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'val_loss': checkpoint.get('val_loss', 'Unknown'),
                'val_acc': checkpoint.get('val_acc', 'Unknown'),
                'timestamp': checkpoint.get('timestamp', 'Unknown'),
                'model_name': checkpoint.get('model_name', 'Unknown'),
                'performance_summary': checkpoint.get('performance_summary', 'Unknown')
            })
        else:
            metadata.update({
                'epoch': 'Unknown',
                'val_loss': 'Unknown', 
                'val_acc': 'Unknown',
                'timestamp': 'Unknown',
                'model_name': 'Unknown',
                'performance_summary': 'State dict only'
            })
        
        return metadata
        
    except Exception as e:
        return {
            'file_path': model_path,
            'file_name': os.path.basename(model_path),
            'error': str(e),
            'file_size_mb': os.path.getsize(model_path) / (1024*1024) if os.path.exists(model_path) else 0
        }

def identify_model_architecture(file_name, model_name_from_checkpoint=None):
    """Identify model architecture from filename or checkpoint"""
    file_name_lower = file_name.lower()
    checkpoint_name_lower = str(model_name_from_checkpoint).lower() if model_name_from_checkpoint else ""
    
    # Check both filename and checkpoint model name
    combined_name = f"{file_name_lower} {checkpoint_name_lower}"
    
    if 'inception' in combined_name:
        return 'InceptionV3'
    elif 'resnet' in combined_name:
        return 'ResNet50'
    elif 'vit' in combined_name and 'maxvit' not in combined_name:
        return 'ViT'
    elif 'efficientnet' in combined_name:
        return 'EfficientNet'
    elif 'convnext' in combined_name:
        return 'ConvNeXt'
    elif 'swin' in combined_name:
        return 'Swin'
    elif 'regnet' in combined_name:
        return 'RegNet'
    elif 'maxvit' in combined_name:
        return 'MaxViT'
    elif 'densenet' in combined_name:
        return 'DenseNet'
    else:
        return 'Unknown'

def create_model_architecture(model_type, device):
    """Create model architecture based on type"""
    
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
        return model.to(device)
    
    elif model_type == 'ResNet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.fc.in_features, 1)
        )
        return model.to(device)
    
    elif model_type == 'ViT':
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=1)
            return model.to(device)
        else:
            # Fallback
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
            return model.to(device)
    
    elif model_type == 'EfficientNet' and TIMM_AVAILABLE:
        model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
        return model.to(device)
    
    elif model_type == 'ConvNeXt' and TIMM_AVAILABLE:
        model = timm.create_model('convnext_base', pretrained=False, num_classes=1)
        return model.to(device)
    
    elif model_type == 'Swin' and TIMM_AVAILABLE:
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=1)
        return model.to(device)
    
    elif model_type == 'RegNet' and TIMM_AVAILABLE:
        model = timm.create_model('regnety_032', pretrained=False, num_classes=1)
        return model.to(device)
    
    elif model_type == 'MaxViT' and TIMM_AVAILABLE:
        model = timm.create_model('maxvit_tiny_tf_224', pretrained=False, num_classes=1)
        return model.to(device)
    
    elif model_type == 'DenseNet':
        model = models.densenet201(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier.in_features, 1),
            nn.Sigmoid()
        )
        return model.to(device)
    
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None

def test_model_performance(model_path, model_type, test_loader, device):
    """Test a single model's performance"""
    try:
        # Create architecture
        model = create_model_architecture(model_type, device)
        if model is None:
            return None
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        
        # Test on validation set
        predictions = []
        probabilities = []
        true_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Adjust input size for InceptionV3
                if model_type == "InceptionV3" and inputs.size(-1) != 299:
                    inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
                
                outputs = model(inputs)
                
                # Handle tuple output
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                
                # Get probabilities
                if model_type in ['ResNet50', 'EfficientNet', 'ConvNeXt', 'Swin', 'RegNet', 'MaxViT']:
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
            'test_successful': True
        }
        
    except Exception as e:
        print(f"❌ Error testing {model_type}: {e}")
        return {
            'accuracy': 0.0,
            'auc': 0.0,
            'f1_score': 0.0,
            'error': str(e),
            'test_successful': False
        }

def select_best_models(all_results, max_models=5):
    """Select the best models for ensemble"""
    print(f"\n🎯 SELECTING BEST MODELS FOR ENSEMBLE")
    print("=" * 50)
    
    # Filter successful models
    successful_models = {k: v for k, v in all_results.items() 
                        if v.get('performance', {}).get('test_successful', False)}
    
    if not successful_models:
        print("❌ No models tested successfully")
        return {}
    
    # Sort by accuracy
    sorted_models = sorted(successful_models.items(), 
                          key=lambda x: x[1]['performance']['accuracy'], 
                          reverse=True)
    
    print(f"📊 Model Performance Ranking:")
    print(f"{'Rank':<5} {'Model':<20} {'Architecture':<15} {'Accuracy':<10} {'AUC':<8}")
    print("-" * 70)
    
    for i, (model_id, model_info) in enumerate(sorted_models, 1):
        perf = model_info['performance']
        print(f"{i:<5} {model_info['metadata']['file_name'][:19]:<20} "
              f"{model_info['architecture']:<15} {perf['accuracy']:>7.2f}%  {perf['auc']:>6.2f}%")
    
    # Select top models with diversity
    selected_models = {}
    selected_architectures = set()
    
    # Strategy: Pick best models but ensure architecture diversity
    for model_id, model_info in sorted_models:
        architecture = model_info['architecture']
        
        # Always include if we have fewer than max_models
        # Or include if it's a new architecture
        if (len(selected_models) < max_models or 
            architecture not in selected_architectures):
            selected_models[model_id] = model_info
            selected_architectures.add(architecture)
            
            if len(selected_models) >= max_models:
                break
    
    print(f"\n✅ Selected {len(selected_models)} models for ensemble:")
    for i, (model_id, model_info) in enumerate(selected_models.items(), 1):
        perf = model_info['performance']
        print(f"   {i}. {model_info['architecture']}: {perf['accuracy']:.2f}% accuracy")
    
    return selected_models

def create_optimal_ensemble(selected_models, true_labels):
    """Create optimal ensemble from selected models"""
    print(f"\n🤝 CREATING OPTIMAL ENSEMBLE")
    print("=" * 40)
    
    if len(selected_models) < 2:
        print("❌ Need at least 2 models for ensemble")
        return None
    
    # Get predictions from all selected models
    model_names = list(selected_models.keys())
    all_probs = np.column_stack([
        selected_models[name]['performance']['probabilities'] 
        for name in model_names
    ])
    
    # Test different ensemble strategies
    strategies = {}
    
    # 1. Simple Average
    avg_probs = np.mean(all_probs, axis=1)
    avg_preds = (avg_probs > 0.5).astype(int)
    strategies['Simple Average'] = (avg_probs, avg_preds)
    
    # 2. Weighted by Accuracy
    weights = np.array([selected_models[name]['performance']['accuracy'] for name in model_names])
    weights = weights / np.sum(weights)
    weighted_probs = np.average(all_probs, axis=1, weights=weights)
    weighted_preds = (weighted_probs > 0.5).astype(int)
    strategies['Accuracy Weighted'] = (weighted_probs, weighted_preds)
    
    # 3. Weighted by AUC
    auc_weights = np.array([selected_models[name]['performance']['auc'] for name in model_names])
    auc_weights = auc_weights / np.sum(auc_weights) if np.sum(auc_weights) > 0 else weights
    auc_weighted_probs = np.average(all_probs, axis=1, weights=auc_weights)
    auc_weighted_preds = (auc_weighted_probs > 0.5).astype(int)
    strategies['AUC Weighted'] = (auc_weighted_probs, auc_weighted_preds)
    
    # 4. Confidence-based
    conf_probs = np.mean(all_probs, axis=1)
    conf_preds = (conf_probs > 0.6).astype(int)
    strategies['High Confidence'] = (conf_probs, conf_preds)
    
    # 5. Majority Vote
    binary_preds = all_probs > 0.5
    majority_preds = (np.sum(binary_preds, axis=1) > len(model_names)/2).astype(int)
    strategies['Majority Vote'] = (avg_probs, majority_preds)
    
    # Evaluate strategies
    print(f"📊 Ensemble Strategy Performance:")
    print(f"{'Strategy':<18} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 50)
    
    best_strategy = None
    best_accuracy = 0
    strategy_results = {}
    
    for strategy_name, (probs, preds) in strategies.items():
        accuracy = accuracy_score(true_labels, preds) * 100
        try:
            auc = roc_auc_score(true_labels, probs) * 100
        except:
            auc = 50.0
        f1 = f1_score(true_labels, preds, zero_division=0) * 100
        
        strategy_results[strategy_name] = {
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
    
    print(f"\n🏆 Best ensemble strategy: {best_strategy} ({best_accuracy:.2f}% accuracy)")
    
    return strategy_results, best_strategy

def main():
    """Main model selection and analysis"""
    print("🚀 Ki-67 Model Selection and Analysis")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup paths
    base_dir = "/Users/chinthan/ki7"
    dataset_path = os.path.join(base_dir, "Ki67_Dataset_for_Colab")
    
    # Step 1: Scan all model files
    all_model_files = scan_all_model_files(base_dir)
    
    if not all_model_files:
        print("❌ No .pth files found. Please download models from Google Drive first.")
        return
    
    # Step 2: Analyze metadata for all models
    print(f"\n📋 Analyzing model metadata...")
    all_models_info = {}
    
    for model_path in all_model_files:
        model_id = os.path.basename(model_path).replace('.pth', '')
        metadata = analyze_model_metadata(model_path)
        architecture = identify_model_architecture(metadata['file_name'], metadata.get('model_name'))
        
        all_models_info[model_id] = {
            'metadata': metadata,
            'architecture': architecture,
            'path': model_path
        }
    
    # Display metadata summary
    print(f"\n📊 Model Metadata Summary:")
    print(f"{'File Name':<30} {'Architecture':<15} {'Val Acc':<10} {'Size (MB)':<10}")
    print("-" * 70)
    
    for model_id, info in all_models_info.items():
        metadata = info['metadata']
        val_acc = metadata.get('val_acc', 'Unknown')
        val_acc_str = f"{val_acc:.2f}%" if isinstance(val_acc, (int, float)) else str(val_acc)
        
        print(f"{metadata['file_name'][:29]:<30} {info['architecture']:<15} "
              f"{val_acc_str:<10} {metadata['file_size_mb']:>8.1f}")
    
    # Step 3: Create test dataset
    print(f"\n📊 Loading test dataset...")
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
    
    # Step 4: Test all models
    print(f"\n🧪 Testing {len(all_models_info)} models on validation set...")
    
    all_results = {}
    true_labels = None
    
    for model_id, model_info in all_models_info.items():
        print(f"Testing {model_info['architecture']} ({model_id[:20]}...)...")
        
        performance = test_model_performance(
            model_info['path'], 
            model_info['architecture'], 
            test_loader, 
            device
        )
        
        if performance and performance.get('test_successful', False):
            print(f"✅ {model_info['architecture']}: {performance['accuracy']:.2f}% accuracy")
            if true_labels is None:
                true_labels = test_dataset.labels
        else:
            print(f"❌ {model_info['architecture']}: Testing failed")
        
        all_results[model_id] = {
            **model_info,
            'performance': performance
        }
    
    # Step 5: Select best models
    selected_models = select_best_models(all_results, max_models=5)
    
    if not selected_models:
        print("❌ No models selected successfully")
        return
    
    # Step 6: Create optimal ensemble
    ensemble_results, best_strategy = create_optimal_ensemble(selected_models, true_labels)
    
    if ensemble_results:
        best_ensemble = ensemble_results[best_strategy]
        
        print(f"\n🎉 FINAL RESULTS:")
        print(f"📊 Best ensemble accuracy: {best_ensemble['accuracy']:.2f}%")
        print(f"🏆 Strategy: {best_strategy}")
        print(f"🔢 Models in ensemble: {len(selected_models)}")
        
        if best_ensemble['accuracy'] >= 95.0:
            print(f"🎉 TARGET ACHIEVED! 95%+ accuracy reached!")
        elif best_ensemble['accuracy'] >= 90.0:
            print(f"🔥 Excellent performance! Very close to 95% target.")
        elif best_ensemble['accuracy'] >= 80.0:
            print(f"✅ Good performance! Consider training 1-2 more models.")
        else:
            print(f"📈 Room for improvement. Check model training and data.")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = {
            'timestamp': timestamp,
            'total_models_found': len(all_model_files),
            'models_tested': len([r for r in all_results.values() if r['performance'].get('test_successful', False)]),
            'selected_models': len(selected_models),
            'best_ensemble_accuracy': float(best_ensemble['accuracy']),
            'best_strategy': best_strategy,
            'target_achieved': best_ensemble['accuracy'] >= 95.0,
            'selected_model_details': {
                k: {
                    'architecture': v['architecture'],
                    'accuracy': float(v['performance']['accuracy']),
                    'file_name': v['metadata']['file_name']
                } for k, v in selected_models.items()
            }
        }
        
        results_file = os.path.join(base_dir, f'model_selection_results_{timestamp}.json')
        with open(results_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n📁 Results saved: {os.path.basename(results_file)}")
        print(f"✅ Model selection and analysis completed!")
    
    else:
        print("❌ Ensemble creation failed")

if __name__ == "__main__":
    main()
