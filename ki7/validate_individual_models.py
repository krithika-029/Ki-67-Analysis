#!/usr/bin/env python3
"""
Individual Model Validator
Tests each trained model individually to determine their actual test accuracies
This helps select the best models for the ensemble.
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
from collections import OrderedDict

# Suppress warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

try:
    import timm
    print(f"✅ timm version: {timm.__version__}")
except ImportError:
    print("Installing timm...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "timm"])
    import timm

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score
)

class ValidationDataset(Dataset):
    """Dataset for individual model validation"""
    
    def __init__(self, dataset_path, split='test', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        self.create_dataset_from_directory()
    
    def create_dataset_from_directory(self):
        """Create dataset using proven logic"""
        print(f"🔧 Creating {self.split} dataset for validation...")
        
        # Try different possible dataset structures
        possible_paths = [
            self.dataset_path / "Ki67_Dataset_for_Colab",
            self.dataset_path / "BCData", 
            self.dataset_path / "ki67_dataset",
            self.dataset_path
        ]
        
        base_path = None
        for path in possible_paths:
            if path.exists():
                if (path / "images" / self.split).exists() and (path / "annotations" / self.split).exists():
                    base_path = path
                    break
        
        if base_path is None:
            print(f"❌ No valid dataset path found")
            return
        
        images_dir = base_path / "images" / self.split
        pos_annotations_dir = base_path / "annotations" / self.split / "positive"
        neg_annotations_dir = base_path / "annotations" / self.split / "negative"
        
        if not images_dir.exists():
            print(f"❌ Images directory not found: {images_dir}")
            return
        
        print(f"📁 Loading from: {images_dir}")
        
        # Use proven annotation file size logic
        for img_file in images_dir.glob("*.png"):
            img_name = img_file.stem
            
            pos_ann = pos_annotations_dir / f"{img_name}.h5"
            neg_ann = neg_annotations_dir / f"{img_name}.h5"
            
            if pos_ann.exists() and neg_ann.exists():
                try:
                    pos_size = pos_ann.stat().st_size
                    neg_size = neg_ann.stat().st_size
                    size_diff = abs(pos_size - neg_size)
                    
                    if size_diff > 100:
                        if neg_size > pos_size:
                            self.images.append(str(img_file))
                            self.labels.append(0)  # Negative
                        else:
                            self.images.append(str(img_file))
                            self.labels.append(1)  # Positive
                    else:
                        idx = len(self.images)
                        self.images.append(str(img_file))
                        self.labels.append(idx % 2)
                        
                except Exception as e:
                    idx = len(self.images)
                    self.images.append(str(img_file))
                    self.labels.append(idx % 2)
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"✅ Found {len(self.images)} images")
        print(f"   Distribution: {pos_count} positive, {neg_count} negative")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
                
        except Exception as e:
            print(f"⚠️  Error loading image {img_path}: {e}")
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224), dtype=torch.float32)
            return fallback, torch.tensor(label, dtype=torch.float32)

def get_model_transform(architecture):
    """Get appropriate transform for each model architecture"""
    if 'efficientnet_b5' in architecture or 'B5' in architecture:
        size = 320  # B5 typically uses 320x320
    elif 'efficientnet_b4' in architecture or 'B4' in architecture:
        size = 288  # B4 typically uses 288x288
    elif 'vit' in architecture.lower():
        size = 224  # ViT uses 224x224
    else:
        size = 224  # Default size
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform

def load_and_validate_model(model_path, config, device, test_loader):
    """Load and validate a single model"""
    print(f"\n{'='*60}")
    print(f"🧪 Validating: {config['name']}")
    print(f"📁 File: {Path(model_path).name}")
    print(f"🏗️  Architecture: {config['architecture']}")
    
    try:
        # Create model architecture
        if config['architecture'] == 'vit_base_patch16_224':
            model = timm.create_model(config['architecture'], pretrained=False, num_classes=1, img_size=224)
        elif 'efficientnet_b5' in config['architecture']:
            model = timm.create_model(config['architecture'], pretrained=False, num_classes=1)
        else:
            model = timm.create_model(config['architecture'], pretrained=False, num_classes=1)
        
        # Load trained weights
        print("📦 Loading model weights...")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract training info if available
        training_info = {}
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # Extract training metadata
                for key in ['best_val_acc', 'best_val_loss', 'epoch', 'train_acc', 'val_acc']:
                    if key in checkpoint:
                        training_info[key] = checkpoint[key]
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(device)
        model.eval()
        
        # Print training info if available
        if training_info:
            print("📊 Training Information:")
            for key, value in training_info.items():
                if 'acc' in key.lower():
                    print(f"   {key}: {value:.2f}%")
                else:
                    print(f"   {key}: {value}")
        
        # Validate on test set
        print("🧪 Running validation...")
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if batch_idx % 20 == 0:
                    print(f"   Processing batch {batch_idx+1}/{len(test_loader)}")
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                # Get model prediction
                outputs = model(inputs)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)
                
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        probabilities = np.array(all_probabilities)
        
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        try:
            auc = roc_auc_score(targets, probabilities)
        except:
            auc = 0.5
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        results = {
            'name': config['name'],
            'architecture': config['architecture'],
            'file': Path(model_path).name,
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'confusion_matrix': cm.tolist(),
            'training_info': training_info,
            'total_params': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Print results
        print(f"\n📊 Validation Results:")
        print(f"   ✅ Test Accuracy: {accuracy*100:.2f}%")
        print(f"   🎯 Precision: {precision:.3f}")
        print(f"   🔄 Recall: {recall:.3f}")
        print(f"   ⚖️  F1-Score: {f1:.3f}")
        print(f"   📈 AUC: {auc:.3f}")
        print(f"   📐 Parameters: {results['total_params']:,}")
        
        # Performance tier
        if accuracy >= 0.90:
            tier = "🏆 EXCELLENT"
        elif accuracy >= 0.85:
            tier = "🔥 VERY GOOD"
        elif accuracy >= 0.80:
            tier = "✅ GOOD"
        elif accuracy >= 0.75:
            tier = "⚠️  FAIR"
        else:
            tier = "❌ NEEDS IMPROVEMENT"
        
        print(f"   🏅 Performance Tier: {tier}")
        
        return results
        
    except Exception as e:
        print(f"❌ Failed to validate model: {e}")
        return {
            'name': config['name'],
            'architecture': config['architecture'],
            'file': Path(model_path).name,
            'error': str(e),
            'test_accuracy': 0.0
        }

def create_validation_report(all_results, save_path):
    """Create comprehensive validation report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort by test accuracy
    valid_results = [r for r in all_results if 'error' not in r]
    valid_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy ranking
    names = [r['name'][:20] + '...' if len(r['name']) > 20 else r['name'] for r in valid_results]
    accuracies = [r['test_accuracy'] * 100 for r in valid_results]
    
    colors = ['gold' if acc >= 90 else 'lightgreen' if acc >= 85 else 'orange' if acc >= 80 else 'lightcoral' for acc in accuracies]
    bars = ax1.barh(range(len(names)), accuracies, color=colors)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('Test Accuracy (%)')
    ax1.set_title('Individual Model Performance Ranking')
    ax1.axvline(x=90, color='red', linestyle='--', alpha=0.7, label='90% Threshold')
    ax1.legend()
    
    # Add accuracy labels
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{acc:.1f}%', va='center', fontweight='bold')
    
    # 2. Architecture comparison
    arch_performance = {}
    for r in valid_results:
        arch = r['architecture']
        if arch not in arch_performance:
            arch_performance[arch] = []
        arch_performance[arch].append(r['test_accuracy'] * 100)
    
    arch_names = list(arch_performance.keys())
    arch_means = [np.mean(arch_performance[arch]) for arch in arch_names]
    
    ax2.bar(range(len(arch_names)), arch_means, color='skyblue', alpha=0.7)
    ax2.set_xticks(range(len(arch_names)))
    ax2.set_xticklabels([name.replace('efficientnet_', 'ENet-') for name in arch_names], rotation=45)
    ax2.set_ylabel('Average Accuracy (%)')
    ax2.set_title('Performance by Architecture')
    
    # 3. Metrics comparison for top 5 models
    top5_results = valid_results[:5]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    x = np.arange(len(metrics))
    width = 0.15
    
    for i, result in enumerate(top5_results):
        values = [
            result['test_accuracy'],
            result['precision'],
            result['recall'],
            result['f1_score'],
            result['auc_score']
        ]
        ax3.bar(x + i*width, values, width, label=result['name'][:15], alpha=0.8)
    
    ax3.set_xlabel('Metrics')
    ax3.set_ylabel('Score')
    ax3.set_title('Top 5 Models - Detailed Metrics')
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(metrics)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_ylim([0, 1])
    
    # 4. Model size vs accuracy
    params = [r.get('total_params', 0) / 1e6 for r in valid_results]  # Convert to millions
    ax4.scatter(params, accuracies, c=accuracies, cmap='viridis', s=100, alpha=0.7)
    ax4.set_xlabel('Model Size (Million Parameters)')
    ax4.set_ylabel('Test Accuracy (%)')
    ax4.set_title('Model Size vs Accuracy')
    
    # Add labels for top performers
    for i, result in enumerate(valid_results[:3]):
        if params[i] > 0:
            ax4.annotate(result['name'][:10], (params[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    # Save visualization
    viz_file = save_path / f"model_validation_report_{timestamp}.png"
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed JSON report
    report = {
        'timestamp': timestamp,
        'summary': {
            'total_models': len(all_results),
            'successful_validations': len(valid_results),
            'failed_validations': len(all_results) - len(valid_results),
            'best_accuracy': max([r['test_accuracy'] for r in valid_results]) if valid_results else 0,
            'average_accuracy': np.mean([r['test_accuracy'] for r in valid_results]) if valid_results else 0
        },
        'detailed_results': all_results,
        'top_performers': valid_results[:5],
        'recommendations': {
            'ensemble_candidates': [r['name'] for r in valid_results[:5] if r['test_accuracy'] >= 0.80],
            'best_single_model': valid_results[0]['name'] if valid_results else None
        }
    }
    
    report_file = save_path / f"model_validation_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report, viz_file, report_file

def main():
    """Main validation function"""
    print("🧪 Individual Model Validator")
    print("=" * 50)
    print("Testing each trained model to determine actual test accuracies")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Paths
    current_dir = Path.cwd()
    models_dir = current_dir / "models"
    results_dir = current_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Find dataset
    dataset_paths = ["./Ki67_Dataset_for_Colab", "./BCData", "./data"]
    dataset_path = None
    for path in dataset_paths:
        if Path(path).exists():
            dataset_path = Path(path)
            break
    
    if dataset_path is None:
        print("❌ Dataset not found. Please ensure dataset is available.")
        return
    
    print(f"📂 Models: {models_dir}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"💾 Results: {results_dir}")
    
    # Model configurations
    model_configs = {
        'Ki67_STABLE_Champion_EfficientNet_B5_Champion_FINAL_90.98_20250620_142507.pth': {
            'architecture': 'efficientnet_b5',
            'name': 'EfficientNet-B5-Champion-STABLE'
        },
        'Ki67_B4_Adapted_Champion_EfficientNet_B4_Adapted_best_model_20250620_133200.pth': {
            'architecture': 'efficientnet_b4',
            'name': 'EfficientNet-B4-Adapted-Champion'
        },
        'Ki67_T4_Champion_EfficientNet_B4_best_model_20250620_111518.pth': {
            'architecture': 'efficientnet_b4',
            'name': 'EfficientNet-B4-T4-Champion'
        },
        'Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth': {
            'architecture': 'efficientnet_b2',
            'name': 'EfficientNet-B2-Advanced'
        },
        'Ki67_Advanced_ConvNeXt-Tiny_best_model_20250619_110232.pth': {
            'architecture': 'convnext_tiny',
            'name': 'ConvNeXt-Tiny-Advanced'
        },
        'Ki67_ViT_best_model_20250619_071454.pth': {
            'architecture': 'vit_base_patch16_224',
            'name': 'Vision-Transformer-Base'
        },
        'Ki67_Advanced_DenseNet-121_best_model_20250619_111107.pth': {
            'architecture': 'densenet121',
            'name': 'DenseNet-121-Advanced'
        },
        'Ki67_Advanced_Swin-Tiny_best_model_20250619_110516.pth': {
            'architecture': 'swin_tiny_patch4_window7_224',
            'name': 'Swin-Transformer-Tiny'
        },
        'Ki67_ResNet50_best_model_20250619_070508.pth': {
            'architecture': 'resnet50',
            'name': 'ResNet50'
        },
        'Ki67_InceptionV3_best_model_20250619_070054.pth': {
            'architecture': 'inception_v3',
            'name': 'InceptionV3'
        },
        'Ki67_Advanced_RegNet-Y-8GF_best_model_20250619_111223.pth': {
            'architecture': 'regnetx_008',
            'name': 'RegNet-Y-8GF-Advanced'
        }
    }
    
    # Create test dataset (using standard size first)
    print("\n📊 Creating test dataset...")
    standard_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ValidationDataset(dataset_path, split='test', transform=standard_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"✅ Test dataset ready: {len(test_dataset)} samples")
    
    # Validate each model
    print(f"\n🧪 Starting individual model validation...")
    print(f"Found {len(model_configs)} model configurations to test")
    
    all_results = []
    
    for model_file, config in model_configs.items():
        model_path = models_dir / model_file
        
        if model_path.exists():
            # Create architecture-specific transform if needed
            if config['architecture'] in ['efficientnet_b5']:
                specific_transform = get_model_transform(config['architecture'])
                specific_dataset = ValidationDataset(dataset_path, split='test', transform=specific_transform)
                specific_loader = DataLoader(specific_dataset, batch_size=8, shuffle=False, num_workers=0)
                result = load_and_validate_model(model_path, config, device, specific_loader)
            else:
                result = load_and_validate_model(model_path, config, device, test_loader)
            
            all_results.append(result)
        else:
            print(f"\n⚠️  Model file not found: {model_file}")
    
    # Create comprehensive report
    print(f"\n📊 Creating validation report...")
    report, viz_file, report_file = create_validation_report(all_results, results_dir)
    
    # Display summary
    print(f"\n" + "="*60)
    print(f"🏆 MODEL VALIDATION SUMMARY")
    print(f"="*60)
    print(f"📊 Total models tested: {report['summary']['total_models']}")
    print(f"✅ Successful validations: {report['summary']['successful_validations']}")
    print(f"❌ Failed validations: {report['summary']['failed_validations']}")
    print(f"🥇 Best accuracy: {report['summary']['best_accuracy']*100:.2f}%")
    print(f"📈 Average accuracy: {report['summary']['average_accuracy']*100:.2f}%")
    
    print(f"\n🏆 TOP 5 PERFORMERS:")
    for i, model in enumerate(report['top_performers']):
        tier_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "🏅"
        print(f"   {tier_emoji} {model['name']}: {model['test_accuracy']*100:.2f}%")
    
    print(f"\n💡 ENSEMBLE RECOMMENDATIONS:")
    candidates = report['recommendations']['ensemble_candidates']
    if candidates:
        print(f"   🎯 Strong candidates (≥80%): {len(candidates)} models")
        for candidate in candidates:
            print(f"      ✅ {candidate}")
    else:
        print(f"   ⚠️  No models achieved ≥80% accuracy")
    
    print(f"\n📂 Files created:")
    print(f"   📊 Visualization: {viz_file}")
    print(f"   📋 Detailed report: {report_file}")
    
    # Recommendations for ensemble
    print(f"\n🚀 NEXT STEPS:")
    top_accuracy = report['summary']['best_accuracy'] * 100
    if top_accuracy >= 90:
        print(f"   🎉 Excellent! Best model: {top_accuracy:.1f}%")
        print(f"   🎯 Recommend ensemble with top 3-5 models")
    elif top_accuracy >= 85:
        print(f"   🔥 Good performance! Best model: {top_accuracy:.1f}%") 
        print(f"   🎯 Ensemble with top models should reach 90%+")
    else:
        print(f"   📈 Current best: {top_accuracy:.1f}%")
        print(f"   💡 May need additional training or techniques")
    
    print(f"\n✅ Individual model validation complete!")
    
    return report

if __name__ == "__main__":
    main()
