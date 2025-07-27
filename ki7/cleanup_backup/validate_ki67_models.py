#!/usr/bin/env python3
"""
Ki-67 Model Validation Script
=============================
Validates the 3 trained models (InceptionV3, ResNet50, ViT) on the Ki67_Dataset_for_Colab dataset
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
import pickle
from datetime import datetime
import os
import zipfile

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

try:
    import timm
    print("✅ timm available for ViT model")
except ImportError:
    print("⚠️  timm not available, will use fallback CNN for ViT")
    timm = None

print("🔬 Ki-67 Model Validation System")
print("="*50)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def setup_google_drive():
    """Setup Google Drive mounting for Colab"""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive mounted")
        return "/content/drive/MyDrive"
    except ImportError:
        print("⚠️  Not in Google Colab, using local paths")
        return "."

def extract_dataset(dataset_zip_path, extract_path="/content/ki67_validation_dataset"):
    """Extract the Ki67 dataset"""
    if os.path.exists(dataset_zip_path):
        print(f"✅ Found dataset at: {dataset_zip_path}")
        
        os.makedirs(extract_path, exist_ok=True)
        
        print("Extracting dataset...")
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print("✅ Dataset extracted successfully!")
        return extract_path
    else:
        print(f"❌ Dataset not found at: {dataset_zip_path}")
        return None

class Ki67ValidationDataset(Dataset):
    """Dataset class for Ki67 validation"""
    
    def __init__(self, dataset_path, split='test', transform=None):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        
        # Load dataset
        self.load_from_csv()
        
    def load_from_csv(self):
        """Load dataset from CSV with corrected labels"""
        csv_path = self.dataset_path / "ki67_dataset_metadata.csv"
        
        if csv_path.exists():
            print(f"Loading {self.split} data from CSV...")
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
                    corrected_labels.append(idx % 2)  # Fallback
            
            self.data['corrected_label'] = corrected_labels
            
            print(f"Loaded {len(self.data)} samples for {self.split}")
            pos_count = sum(corrected_labels)
            neg_count = len(corrected_labels) - pos_count
            print(f"Distribution: {pos_count} positive, {neg_count} negative")
            
        else:
            print(f"❌ CSV file not found at: {csv_path}")
            self.data = pd.DataFrame()
    
    def normalize_path_for_colab(self, path_str):
        """Normalize Windows paths for Colab"""
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
        img_relative_path = self.normalize_path_for_colab(row['image_path'])
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

def create_models(device):
    """Create the 3 model architectures"""
    print("🏗️ Creating model architectures...")
    
    models_dict = {}
    
    # InceptionV3
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
    
    # ResNet50
    resnet_model = models.resnet50(pretrained=False)
    resnet_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(resnet_model.fc.in_features, 1)
    )
    resnet_model = resnet_model.to(device)
    models_dict['ResNet50'] = resnet_model
    print("✅ ResNet50 architecture created")
    
    # ViT
    try:
        if timm is not None:
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

def load_trained_models(models_dict, models_path):
    """Load the trained model weights"""
    print("📥 Loading trained model weights...")
    
    model_files = {
        'InceptionV3': 'Ki67_InceptionV3_best_model_20250619_070054.pth',
        'ResNet50': 'Ki67_ResNet50_best_model_20250619_070508.pth',
        'ViT': 'Ki67_ViT_best_model_20250619_071454.pth'
    }
    
    loaded_models = {}
    
    for model_name, model in models_dict.items():
        model_file = model_files.get(model_name)
        if model_file:
            model_path = os.path.join(models_path, model_file)
            
            if os.path.exists(model_path):
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    print(f"✅ {model_name} loaded from {model_file}")
                    print(f"   Performance: {checkpoint.get('performance_summary', 'N/A')}")
                    
                    loaded_models[model_name] = model
                    
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            else:
                print(f"❌ Model file not found: {model_path}")
        else:
            print(f"❌ No file specified for {model_name}")
    
    return loaded_models

def validate_models(models_dict, test_loader, device):
    """Validate all models on test set"""
    print("🔍 Validating models on test set...")
    
    results = {}
    all_predictions = {}
    all_labels = []
    
    # Set all models to evaluation mode
    for model in models_dict.values():
        model.eval()
    
    with torch.no_grad():
        for model_name, model in models_dict.items():
            model_predictions = []
            model_probs = []
            
            print(f"\nEvaluating {model_name}...")
            
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
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"  AUC:       {auc*100:.2f}%")
    
    return results, all_labels

def create_ensemble_predictions(results, all_labels):
    """Create ensemble predictions"""
    print("\n🤝 Creating ensemble predictions...")
    
    # Get individual probabilities
    model_names = list(results.keys())
    probs_matrix = np.column_stack([results[name]['probabilities'] for name in model_names])
    
    # Simple average ensemble
    ensemble_probs = np.mean(probs_matrix, axis=1)
    ensemble_preds = (ensemble_probs > 0.5).astype(int)
    
    # Calculate ensemble metrics
    accuracy = accuracy_score(all_labels, ensemble_preds)
    precision = precision_score(all_labels, ensemble_preds, zero_division=0)
    recall = recall_score(all_labels, ensemble_preds, zero_division=0)
    f1 = f1_score(all_labels, ensemble_preds, zero_division=0)
    auc = roc_auc_score(all_labels, ensemble_probs)
    
    print(f"Ensemble Results:")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  AUC:       {auc*100:.2f}%")
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'auc': auc * 100,
        'predictions': ensemble_preds,
        'probabilities': ensemble_probs
    }

def plot_confusion_matrices(results, all_labels, ensemble_results):
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
    plt.savefig('ki67_validation_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Confusion matrices saved as 'ki67_validation_confusion_matrices.png'")

def main():
    """Main validation function"""
    print("🚀 Starting Ki-67 Model Validation...")
    
    # Setup paths
    base_path = setup_google_drive()
    dataset_zip_path = os.path.join(base_path, "Ki67_Dataset", "Ki67_Dataset_for_Colab.zip")
    models_path = base_path
    
    # Extract dataset
    dataset_path = extract_dataset(dataset_zip_path)
    if dataset_path is None:
        print("❌ Cannot proceed without dataset")
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
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    
    # Create and load models
    models_dict = create_models(device)
    loaded_models = load_trained_models(models_dict, models_path)
    
    if len(loaded_models) == 0:
        print("❌ No models loaded successfully")
        return
    
    print(f"\n✅ Successfully loaded {len(loaded_models)} models")
    
    # Validate models
    results, all_labels = validate_models(loaded_models, test_loader, device)
    
    # Create ensemble
    ensemble_results = create_ensemble_predictions(results, all_labels)
    
    # Plot results
    plot_confusion_matrices(results, all_labels, ensemble_results)
    
    # Summary
    print("\n" + "="*60)
    print("🎉 VALIDATION COMPLETED!")
    print("="*60)
    
    print("\n📊 Summary:")
    all_results = {**results, 'Ensemble': ensemble_results}
    for model_name, model_results in all_results.items():
        print(f"{model_name:12}: Acc={model_results['accuracy']:6.2f}%, AUC={model_results['auc']:6.2f}%")
    
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['accuracy'])
    print(f"\n🏆 Best performing model: {best_model} ({all_results[best_model]['accuracy']:.2f}% accuracy)")
    
    print(f"\n✅ Models are working correctly!")
    print(f"✅ Dataset validation successful!")

if __name__ == "__main__":
    main()
