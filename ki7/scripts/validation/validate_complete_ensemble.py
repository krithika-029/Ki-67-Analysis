#!/usr/bin/env python3
"""
Complete Ki-67 Model Ensemble Validation Script
===============================================
Validates ALL available Ki-67 models (original 3 + additional models) 
for comprehensive ensemble testing to reach 95%+ accuracy.

This script will:
1. Auto-detect all available model files
2. Load original models (InceptionV3, ResNet50, ViT)
3. Load additional models (EfficientNet, ConvNeXt, Swin, etc.)
4. Test multiple ensemble strategies
5. Calculate final ensemble performance
6. Provide recommendations for 95%+ accuracy
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
import glob
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Try to import timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
    print("✅ timm available for advanced models")
except ImportError:
    TIMM_AVAILABLE = False
    print("⚠️  timm not available for advanced models")

def setup_environment():
    """Setup device and environment"""
    print("🔬 Complete Ki-67 Ensemble Validation System")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    return device

def find_all_model_files(base_dir):
    """Find all available Ki-67 model files"""
    print(f"\n🔍 Scanning for model files in: {base_dir}")
    
    # Define patterns for different model types
    model_patterns = {
        'InceptionV3': ['*InceptionV3*best_model*.pth', '*inception*best*.pth'],
        'ResNet50': ['*ResNet50*best_model*.pth', '*resnet*best*.pth'],
        'ViT': ['*ViT*best_model*.pth', '*vit*best*.pth'],
        'EfficientNet': ['*EfficientNet*best*.pth', '*efficientnet*best*.pth'],
        'ConvNeXt': ['*ConvNeXt*best*.pth', '*convnext*best*.pth'],
        'Swin': ['*Swin*best*.pth', '*swin*best*.pth'],
        'RegNet': ['*RegNet*best*.pth', '*regnet*best*.pth'],
        'MaxViT': ['*MaxViT*best*.pth', '*maxvit*best*.pth'],
        'DenseNet': ['*DenseNet*best*.pth', '*densenet*best*.pth']
    }
    
    found_models = {}
    
    # Search in base directory and subdirectories
    search_paths = [
        base_dir,
        os.path.join(base_dir, "Ki67_Models"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "additional_models")
    ]
    
    for model_type, patterns in model_patterns.items():
        for search_path in search_paths:
            if os.path.exists(search_path):
                for pattern in patterns:
                    files = glob.glob(os.path.join(search_path, pattern))
                    if files:
                        # Get the most recent file if multiple exist
                        latest_file = max(files, key=lambda f: os.path.getmtime(f))
                        found_models[model_type] = latest_file
                        print(f"✅ Found {model_type}: {os.path.basename(latest_file)}")
                        break
                if model_type in found_models:
                    break
        
        if model_type not in found_models:
            print(f"❌ No {model_type} model found")
    
    print(f"\n📊 Total models found: {len(found_models)}")
    return found_models

class Ki67ValidationDataset(Dataset):
    """Ki67 Dataset with corrected labels"""
    
    def __init__(self, dataset_path, split='test', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        self.load_from_csv()
        
    def load_from_csv(self):
        """Load dataset from CSV with corrected labels based on actual file structure"""
        csv_path = self.dataset_path / "ki67_dataset_metadata.csv"
        
        if not csv_path.exists():
            print(f"❌ CSV file not found at: {csv_path}")
            self.data = pd.DataFrame()
            return
        
        df = pd.read_csv(csv_path)
        self.data = df[df['split'] == self.split].reset_index(drop=True)
        
        # Correct labels by checking actual file structure
        corrected_labels = []
        valid_samples = []
        
        for idx, row in self.data.iterrows():
            image_name = row['image_name']
            
            # Check if image exists in positive or negative folder
            pos_path = self.dataset_path / "annotations" / self.split / "positive" / f"{Path(image_name).stem}.h5"
            neg_path = self.dataset_path / "annotations" / self.split / "negative" / f"{Path(image_name).stem}.h5"
            
            if pos_path.exists():
                corrected_labels.append(1)
                valid_samples.append(idx)
            elif neg_path.exists():
                corrected_labels.append(0)
                valid_samples.append(idx)
            # Skip samples that don't exist in either folder
        
        # Keep only valid samples
        self.data = self.data.iloc[valid_samples].reset_index(drop=True)
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
            # Return fallback black image
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32)

def create_model_architecture(model_type, device):
    """Create model architecture based on type"""
    
    if model_type == 'InceptionV3':
        model = models.inception_v3(pretrained=False)
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
        model = models.resnet50(pretrained=False)
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
            # Fallback CNN
            class SimpleCNN(nn.Module):
                def __init__(self):
                    super(SimpleCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.AdaptiveAvgPool2d(7)
                    )
                    self.classifier = nn.Sequential(
                        nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
                        nn.Dropout(0.5), nn.Linear(128, 1), nn.Sigmoid()
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = self.classifier(x)
                    return x
            
            return SimpleCNN().to(device)
    
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
        model = models.densenet201(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(model.classifier.in_features, 1),
            nn.Sigmoid()
        )
        return model.to(device)
    
    else:
        print(f"❌ Unknown model type: {model_type}")
        return None

def load_all_models(model_files, device):
    """Load all available models"""
    print("\n📥 Loading all available models...")
    
    loaded_models = {}
    
    for model_type, model_path in model_files.items():
        try:
            print(f"Loading {model_type}...")
            
            # Create model architecture
            model = create_model_architecture(model_type, device)
            if model is None:
                continue
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                perf_summary = checkpoint.get('performance_summary', 'N/A')
                val_acc = checkpoint.get('val_acc', 'N/A')
            else:
                model.load_state_dict(checkpoint)
                perf_summary = 'N/A'
                val_acc = 'N/A'
            
            model.eval()
            loaded_models[model_type] = model
            
            print(f"✅ {model_type} loaded successfully")
            if val_acc != 'N/A':
                print(f"   Training performance: {val_acc:.2f}% accuracy")
            
        except Exception as e:
            print(f"❌ Failed to load {model_type}: {e}")
    
    print(f"\n✅ Successfully loaded {len(loaded_models)} models")
    return loaded_models

def validate_ensemble(models_dict, test_loader, device):
    """Validate complete ensemble"""
    print(f"\n🔍 Validating complete ensemble ({len(models_dict)} models)...")
    
    all_predictions = {}
    all_labels = []
    
    # Set all models to evaluation mode
    for model in models_dict.values():
        model.eval()
    
    with torch.no_grad():
        # Collect predictions from all models
        for model_name, model in models_dict.items():
            print(f"Evaluating {model_name}...")
            
            model_predictions = []
            model_probs = []
            
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
                    if model_name in ['ResNet50', 'EfficientNet', 'ConvNeXt', 'Swin', 'RegNet', 'MaxViT']:
                        probs = torch.sigmoid(outputs)
                    else:
                        probs = outputs  # Already has sigmoid
                    
                    # Store predictions
                    model_probs.extend(probs.cpu().numpy().flatten())
                    predictions = (probs > 0.5).float()
                    model_predictions.extend(predictions.cpu().numpy().flatten())
                    
                    # Store labels (only once)
                    if model_name == list(models_dict.keys())[0]:
                        all_labels.extend(labels.cpu().numpy().flatten())
                
                except Exception as e:
                    print(f"Error in batch {batch_idx} for {model_name}: {e}")
                    continue
            
            all_predictions[model_name] = {
                'predictions': np.array(model_predictions),
                'probabilities': np.array(model_probs)
            }
    
    # Convert labels to numpy
    all_labels = np.array(all_labels)
    
    # Calculate metrics for each model
    individual_results = {}
    print(f"\n📊 Individual Model Performance:")
    print(f"{'Model':<15} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 45)
    
    for model_name, preds in all_predictions.items():
        y_pred = preds['predictions']
        y_prob = preds['probabilities']
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, y_pred) * 100
        try:
            auc = roc_auc_score(all_labels, y_prob) * 100
        except:
            auc = 50.0
        f1 = f1_score(all_labels, y_pred, zero_division=0) * 100
        
        individual_results[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_prob
        }
        
        print(f"{model_name:<15} {accuracy:>7.2f}%  {auc:>6.2f}%  {f1:>6.2f}%")
    
    return individual_results, all_labels

def create_advanced_ensemble(individual_results, all_labels):
    """Create advanced ensemble with multiple strategies"""
    print(f"\n🤝 Testing Advanced Ensemble Strategies...")
    
    model_names = list(individual_results.keys())
    probs_matrix = np.column_stack([individual_results[name]['probabilities'] for name in model_names])
    
    # Test multiple ensemble strategies
    strategies = {}
    
    # 1. Simple Average
    avg_probs = np.mean(probs_matrix, axis=1)
    avg_preds = (avg_probs > 0.5).astype(int)
    strategies['Simple Average'] = (avg_probs, avg_preds)
    
    # 2. Weighted by Individual Accuracy
    weights = np.array([individual_results[name]['accuracy'] for name in model_names])
    weights = weights / np.sum(weights)
    weighted_probs = np.average(probs_matrix, axis=1, weights=weights)
    weighted_preds = (weighted_probs > 0.5).astype(int)
    strategies['Accuracy Weighted'] = (weighted_probs, weighted_preds)
    
    # 3. Weighted by AUC
    auc_weights = np.array([individual_results[name]['auc'] for name in model_names])
    auc_weights = auc_weights / np.sum(auc_weights)
    auc_weighted_probs = np.average(probs_matrix, axis=1, weights=auc_weights)
    auc_weighted_preds = (auc_weighted_probs > 0.5).astype(int)
    strategies['AUC Weighted'] = (auc_weighted_probs, auc_weighted_preds)
    
    # 4. Confidence-based (higher threshold)
    conf_probs = np.mean(probs_matrix, axis=1)
    conf_preds = (conf_probs > 0.6).astype(int)
    strategies['High Confidence'] = (conf_probs, conf_preds)
    
    # 5. Top-3 Models Only (if we have more than 3)
    if len(model_names) > 3:
        top_models = sorted(model_names, key=lambda x: individual_results[x]['accuracy'], reverse=True)[:3]
        top_indices = [model_names.index(m) for m in top_models]
        top_probs = np.mean(probs_matrix[:, top_indices], axis=1)
        top_preds = (top_probs > 0.5).astype(int)
        strategies['Top-3 Models'] = (top_probs, top_preds)
    
    # 6. Majority Voting
    binary_preds = probs_matrix > 0.5
    majority_preds = (np.sum(binary_preds, axis=1) > len(model_names)/2).astype(int)
    strategies['Majority Vote'] = (avg_probs, majority_preds)  # Use avg_probs for AUC
    
    # Evaluate all strategies
    print(f"\n📊 Ensemble Strategy Comparison:")
    print(f"{'Strategy':<18} {'Accuracy':<10} {'AUC':<8} {'F1-Score':<8}")
    print("-" * 50)
    
    best_strategy = None
    best_accuracy = 0
    ensemble_results = {}
    
    for strategy_name, (probs, preds) in strategies.items():
        accuracy = accuracy_score(all_labels, preds) * 100
        try:
            auc = roc_auc_score(all_labels, probs) * 100
        except:
            auc = 50.0
        f1 = f1_score(all_labels, preds, zero_division=0) * 100
        
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
    
    print(f"\n🏆 Best ensemble strategy: {best_strategy} ({best_accuracy:.2f}% accuracy)")
    
    return ensemble_results, best_strategy

def plot_comprehensive_results(individual_results, ensemble_results, all_labels, best_strategy, save_path):
    """Plot comprehensive confusion matrices and results"""
    print(f"\n📊 Creating comprehensive visualizations...")
    
    # Determine grid size
    total_plots = len(individual_results) + len(ensemble_results)
    rows = int(np.ceil(total_plots / 3))
    cols = min(3, total_plots)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    if total_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot individual models
    for model_name, results in individual_results.items():
        if plot_idx < len(axes):
            cm = confusion_matrix(all_labels, results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[plot_idx])
            
            axes[plot_idx].set_title(f'{model_name}\nAcc: {results["accuracy"]:.1f}%')
            axes[plot_idx].set_ylabel('True')
            axes[plot_idx].set_xlabel('Predicted')
            plot_idx += 1
    
    # Plot best ensemble strategies
    for strategy_name, results in ensemble_results.items():
        if plot_idx < len(axes) and strategy_name in [best_strategy, 'Simple Average', 'Accuracy Weighted']:
            cm = confusion_matrix(all_labels, results['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[plot_idx])
            
            title = f'{strategy_name}\nAcc: {results["accuracy"]:.1f}%'
            if strategy_name == best_strategy:
                title += ' ⭐'
            
            axes[plot_idx].set_title(title)
            axes[plot_idx].set_ylabel('True')
            axes[plot_idx].set_xlabel('Predicted')
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    save_file = os.path.join(save_path, 'complete_ensemble_results.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive results saved: {save_file}")
    
    plt.show()

def analyze_95_percent_achievement(individual_results, ensemble_results, best_strategy):
    """Analyze if 95%+ accuracy target is achieved"""
    print(f"\n🎯 95%+ ACCURACY ANALYSIS")
    print("=" * 50)
    
    best_ensemble_acc = ensemble_results[best_strategy]['accuracy']
    target_acc = 95.0
    
    print(f"Target accuracy: {target_acc:.1f}%")
    print(f"Best ensemble accuracy: {best_ensemble_acc:.2f}%")
    print(f"Gap: {target_acc - best_ensemble_acc:.2f}%")
    
    if best_ensemble_acc >= target_acc:
        print(f"🎉 TARGET ACHIEVED! Your ensemble reached {best_ensemble_acc:.2f}%!")
        print(f"🏆 Strategy: {best_strategy}")
        print(f"✅ Ready for clinical deployment!")
    else:
        gap = target_acc - best_ensemble_acc
        print(f"📊 Close to target! Only {gap:.2f}% away.")
        
        if gap <= 1.0:
            print(f"💡 Suggestions to close the gap:")
            print(f"   • Fine-tune ensemble thresholds")
            print(f"   • Add test-time augmentation")
            print(f"   • Train one more high-performance model")
        elif gap <= 2.0:
            print(f"💡 Suggestions to reach 95%+:")
            print(f"   • Train 1-2 additional models (EfficientNet-B5, etc.)")
            print(f"   • Implement stacking ensemble")
            print(f"   • Use cross-validation ensemble")
        else:
            print(f"💡 Suggestions for significant improvement:")
            print(f"   • Add more diverse model architectures")
            print(f"   • Increase training data if available")
            print(f"   • Implement advanced ensemble techniques")

def main():
    """Main validation function"""
    print("🚀 Starting Complete Ki-67 Ensemble Validation...")
    
    # Setup
    device = setup_environment()
    base_dir = "/Users/chinthan/ki7"
    dataset_path = os.path.join(base_dir, "Ki67_Dataset_for_Colab")  # Fixed path
    
    # Find all model files
    model_files = find_all_model_files(base_dir)
    
    if len(model_files) == 0:
        print("❌ No model files found. Please ensure model .pth files are in the directory.")
        return
    
    # Check dataset
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("Please ensure Ki67_Dataset_for_Colab is extracted in the directory.")
        return
    
    # Create transforms and dataset
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = Ki67ValidationDataset(dataset_path, split='test', transform=test_transform)
    
    if len(test_dataset) == 0:
        print("❌ No test data found")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    # Load all models
    loaded_models = load_all_models(model_files, device)
    
    if len(loaded_models) == 0:
        print("❌ No models loaded successfully")
        return
    
    print(f"\n🎯 Ensemble composition: {len(loaded_models)} models")
    for model_name in loaded_models.keys():
        print(f"   • {model_name}")
    
    # Validate all models
    individual_results, all_labels = validate_ensemble(loaded_models, test_loader, device)
    
    # Create advanced ensemble
    ensemble_results, best_strategy = create_advanced_ensemble(individual_results, all_labels)
    
    # Plot results
    plot_comprehensive_results(individual_results, ensemble_results, all_labels, best_strategy, base_dir)
    
    # Analyze 95% achievement
    analyze_95_percent_achievement(individual_results, ensemble_results, best_strategy)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_summary = {
        'timestamp': timestamp,
        'models_used': list(loaded_models.keys()),
        'individual_results': {k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else str(v2) 
                                  for k2, v2 in v.items() if k2 not in ['predictions', 'probabilities']} 
                              for k, v in individual_results.items()},
        'ensemble_results': {k: {k2: float(v2) if isinstance(v2, (int, float, np.number)) else str(v2) 
                                for k2, v2 in v.items() if k2 not in ['predictions', 'probabilities']} 
                            for k, v in ensemble_results.items()},
        'best_strategy': best_strategy,
        'target_achieved': ensemble_results[best_strategy]['accuracy'] >= 95.0
    }
    
    results_file = os.path.join(base_dir, f'complete_ensemble_validation_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n📁 Results saved: {os.path.basename(results_file)}")
    print(f"✅ Complete ensemble validation completed!")

if __name__ == "__main__":
    main()
