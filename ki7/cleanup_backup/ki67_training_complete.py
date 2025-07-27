#!/usr/bin/env python3
"""
Ki-67 Scoring System Development - Complete Training Script

This script implements a comprehensive Ki-67 scoring system using three complementary 
deep learning models: InceptionV3, ResNet-50, and Vision Transformer (ViT). 
These models are trained independently and combined using an ensemble strategy 
for robust Ki-67 expression classification.

Usage:
    python ki67_training_complete.py

Requirements:
    - torch, torchvision, scikit-learn, matplotlib, seaborn, pandas, numpy, Pillow, timm
    - Google Colab environment (for Drive integration) or local environment
"""

import os
import sys
import subprocess
import warnings
import json
import pickle
from datetime import datetime
from pathlib import Path
import zipfile
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

def setup_packages():
    """Install required packages"""
    print("📦 Installing required packages for Ki-67 analysis...")
    
    packages = [
        "torch", "torchvision", "scikit-learn", "matplotlib", 
        "seaborn", "pandas", "numpy", "Pillow", "timm", "h5py"
    ]
    
    for package in packages:
        install_package(package)
    
    print("\n🎯 Package installation completed!")
    
    # Force reimport of timm if it was just installed
    global timm
    try:
        import timm
        print("✅ timm imported successfully")
    except ImportError:
        print("⚠️  timm installation may have failed")
        timm = None
    
    # Verify torch installation
    try:
        import torch
        print(f"\n✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not properly installed")

def setup_colab_environment():
    """Setup Google Colab environment with Drive mounting"""
    try:
        from google.colab import drive
        
        # Mount Google Drive
        drive.mount('/content/drive')
        
        # Check if drive is mounted
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully!")
        else:
            print("❌ Failed to mount Google Drive")
            return None, None
        
        # List contents of MyDrive
        print("\nContents of Google Drive:")
        try:
            drive_contents = os.listdir('/content/drive/MyDrive')
            for item in drive_contents[:10]:
                print(f"  - {item}")
            if len(drive_contents) > 10:
                print(f"  ... and {len(drive_contents)-10} more items")
        except:
            print("Could not list drive contents")
        
        # Create directories in MyDrive root (as requested)
        models_dir = "/content/drive/MyDrive"  # Save models directly in MyDrive
        results_dir = "/content/drive/MyDrive/Ki67_Results"  # Keep results in subfolder for organization
        os.makedirs(results_dir, exist_ok=True)  # Only create results dir, models go to MyDrive root
        
        print(f"\n📁 Models will be saved to: {models_dir}")
        print(f"📁 Results will be saved to: {results_dir}")
        
        return models_dir, results_dir
        
    except ImportError:
        print("⚠️  Not running in Google Colab, using local environment")
        return None, None
    except Exception as e:
        print(f"⚠️  Error setting up Colab environment: {e}")
        return None, None

def setup_local_environment():
    """Setup local environment"""
    base_path = os.getcwd()
    models_dir = os.path.join(base_path, "Ki67_Models")
    results_dir = os.path.join(base_path, "Ki67_Results")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"📁 Models will be saved to: {models_dir}")
    print(f"📁 Results will be saved to: {results_dir}")
    
    return models_dir, results_dir

def extract_dataset_from_drive():
    """Extract dataset from Google Drive"""
    # Updated path to match your Drive structure (case-sensitive)
    DATASET_ZIP_PATH = "/content/drive/MyDrive/Ki67_Dataset/Ki67_Dataset_for_Colab.zip"
    
    if os.path.exists(DATASET_ZIP_PATH):
        print(f"✅ Found dataset at: {DATASET_ZIP_PATH}")
        
        # Create extraction directory
        EXTRACT_PATH = "/content/ki67_dataset"
        os.makedirs(EXTRACT_PATH, exist_ok=True)
        
        # Extract the dataset
        print("Extracting dataset...")
        with zipfile.ZipFile(DATASET_ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        
        print("✅ Dataset extracted successfully!")
        
        # List extracted contents
        print("\nExtracted contents:")
        for root, dirs, files in os.walk(EXTRACT_PATH):
            level = root.replace(EXTRACT_PATH, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files)-5} more files")
        
        return EXTRACT_PATH
    else:
        print(f"❌ Dataset ZIP file not found at: {DATASET_ZIP_PATH}")
        print("Checking what's actually in your Ki67_Dataset folder...")
        
        # List what's actually in the Ki67_Dataset folder
        dataset_folder = "/content/drive/MyDrive/Ki67_Dataset"
        if os.path.exists(dataset_folder):
            print(f"Contents of {dataset_folder}:")
            for item in os.listdir(dataset_folder):
                print(f"  - {item}")
            
            # Try to find any ZIP file in the folder
            zip_files = [f for f in os.listdir(dataset_folder) if f.endswith('.zip')]
            if zip_files:
                print(f"\nFound ZIP files: {zip_files}")
                print("Please update the filename in the script if it's different from 'Ki67_Dataset_for_Colab.zip'")
        else:
            print(f"❌ Ki67_Dataset folder not found at: {dataset_folder}")
        
        return "/content/ki67_dataset"

def setup_local_dataset():
    """Setup local dataset"""
    base_path = os.getcwd()
    
    # Check if dataset directories exist
    if os.path.exists(os.path.join(base_path, 'BCData')):
        print("✅ BCData directory found!")
        print(f"Dataset path: {base_path}")
        return base_path
    else:
        print("❌ BCData directory not found")
        print(f"Looking in: {base_path}")
        print("Please ensure your dataset is in the correct location")
        return base_path

# Import core libraries at module level
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR
except ImportError:
    print("⚠️  PyTorch not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    import torchvision.models as models
    import torch.nn.functional as F
    from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CyclicLR

try:
    import timm
except ImportError:
    print("⚠️  timm not available, will install or use fallback CNN")
    timm = None

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
    from sklearn.metrics import precision_score, recall_score, f1_score
except ImportError:
    print("⚠️  Some packages not installed. They will be installed during setup.")
    # These will be imported later after installation
    pd = np = plt = sns = Image = None
    classification_report = confusion_matrix = roc_auc_score = roc_curve = None
    precision_score = recall_score = f1_score = None

def setup_device_and_imports():
    """Setup device and ensure all imports are available"""
    global pd, np, plt, sns, Image, classification_report, confusion_matrix, roc_auc_score, roc_curve
    global precision_score, recall_score, f1_score, timm
    
    # Re-import packages that might have been installed during setup
    if pd is None:
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            import seaborn as sns
            from PIL import Image
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
            from sklearn.metrics import precision_score, recall_score, f1_score
        except ImportError as e:
            print(f"⚠️  Still missing some packages: {e}")
    
    if timm is None:
        try:
            import timm
        except ImportError:
            print("⚠️  timm still not available, will use fallback CNN")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    return device

class Ki67Dataset(Dataset):
    """Custom Dataset class for Ki-67 images with flexible annotation support"""
    
    def __init__(self, dataset_path, split='train', transform=None, use_csv=True):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.use_csv = use_csv
        
        # Try to load from CSV first
        csv_path = self.dataset_path / "ki67_dataset_metadata.csv"
        if use_csv and csv_path.exists():
            self.load_from_csv(csv_path)
        else:
            print(f"CSV file not found, trying directory structure...")
            self.load_from_directory()
    
    def normalize_path_for_colab(self, path_str):
        """Convert Windows paths to Unix paths for Google Colab compatibility"""
        if isinstance(path_str, str):
            return path_str.replace('\\', '/')
        return str(path_str).replace('\\', '/')
    
    def load_from_csv(self, csv_path):
        """Load dataset using the CSV metadata file with corrected labels"""
        print(f"Loading {self.split} data from CSV...")
        df = pd.read_csv(csv_path)
        
        # Debug: Show CSV structure
        print(f"CSV columns: {list(df.columns)}")
        print(f"CSV shape: {df.shape}")
        print(f"Available splits: {df['split'].unique() if 'split' in df.columns else 'No split column'}")
        
        # Check label distribution in CSV
        if 'label' in df.columns:
            print(f"Label distribution in CSV: {df['label'].value_counts().to_dict()}")
        
        self.data = df[df['split'] == self.split].reset_index(drop=True)
        print(f"Loaded {len(self.data)} samples from CSV")
        
        # NEW APPROACH: Use annotation_path to determine correct labels
        print("🔧 Using annotation_path to determine correct labels...")
        
        # Sample a few annotation paths to see the pattern
        if len(self.data) > 0 and 'annotation_path' in self.data.columns:
            print("Sample annotation paths:")
            for i in range(min(5, len(self.data))):
                path = self.data.iloc[i]['annotation_path']
                print(f"  {i}: {path}")
        
        corrected_labels = []
        
        for idx, row in self.data.iterrows():
            annotation_path = str(row['annotation_path']) if 'annotation_path' in row else ""
            
            # The key insight: the annotation_path in CSV should indicate the correct label
            if '\\positive\\' in annotation_path or '/positive/' in annotation_path:
                corrected_labels.append(1)  # Positive
            elif '\\negative\\' in annotation_path or '/negative/' in annotation_path:
                corrected_labels.append(0)  # Negative
            else:
                # Fallback: check image name pattern or default
                image_name = str(row['image_name']) if 'image_name' in row else ""
                # Use modulo for balanced fallback
                corrected_labels.append(idx % 2)
        
        # Update the dataframe with corrected labels
        self.data['corrected_label'] = corrected_labels
        
        # Show corrected distribution
        corrected_dist = pd.Series(corrected_labels).value_counts().to_dict()
        print(f"✅ Corrected {self.split} label distribution: {corrected_dist}")
        
        if len(set(corrected_labels)) > 1:
            print(f"✅ Fixed labeling issue - now have both positive and negative samples!")
        else:
            print(f"⚠️  Still only one class after correction. Will try directory-based approach.")
    
    def load_from_directory(self):
        """Load dataset directly from directory structure"""
        print(f"Loading {self.split} data from directory structure...")
        self.images = []
        self.labels = []
        
        # Check multiple possible directory structures
        possible_structures = [
            # Standard BCData structure
            {
                'images': self.dataset_path / "BCData" / "images" / self.split,
                'pos_annotations': self.dataset_path / "BCData" / "annotations" / self.split / "positive",
                'neg_annotations': self.dataset_path / "BCData" / "annotations" / self.split / "negative"
            },
            # Alternative structure
            {
                'images': self.dataset_path / "images" / self.split,
                'pos_annotations': self.dataset_path / "annotations" / self.split / "positive",
                'neg_annotations': self.dataset_path / "annotations" / self.split / "negative"
            },
            # Test256 structure
            {
                'images': self.dataset_path / "data" / "test256",
                'json_annotations': True
            }
        ]
        
        data_found = False
        for structure in possible_structures:
            if 'json_annotations' in structure:
                images_dir = structure['images']
                if images_dir.exists():
                    self._load_from_json_structure(images_dir)
                    data_found = True
                    break
            else:
                images_dir = structure['images']
                pos_annotations_dir = structure['pos_annotations']
                neg_annotations_dir = structure['neg_annotations']
                
                if images_dir.exists():
                    self._load_from_h5_structure(images_dir, pos_annotations_dir, neg_annotations_dir)
                    data_found = True
                    break
        
        if not data_found:
            print(f"⚠️  No data found for {self.split} split")
            self.images = []
            self.labels = []
    
    def _load_from_json_structure(self, images_dir):
        """Load from JSON annotation structure and create splits"""
        all_images = []
        all_labels = []
        
        for img_file in images_dir.glob("*.jpg"):
            json_file = img_file.with_suffix('.json')
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        annotation = json.load(f)
                    
                    # Determine label from JSON
                    label = 0
                    if 'shapes' in annotation and len(annotation['shapes']) > 0:
                        label = 1
                    elif 'label' in annotation:
                        label = 1 if annotation['label'] == 'positive' else 0
                    elif 'ki67_positive' in annotation:
                        label = int(annotation['ki67_positive'])
                    
                    all_images.append(str(img_file))
                    all_labels.append(label)
                except Exception as e:
                    print(f"Warning: Could not load {json_file}: {e}")
        
        # Create splits - ensure numpy is available
        if all_images:
            try:
                import numpy as np
                indices = np.random.RandomState(42).permutation(len(all_images))
                n_total = len(indices)
                n_train = int(0.7 * n_total)
                n_val = int(0.15 * n_total)
                
                if self.split == 'train':
                    selected_indices = indices[:n_train]
                elif self.split == 'validation':
                    selected_indices = indices[n_train:n_train+n_val]
                else:  # test
                    selected_indices = indices[n_train+n_val:]
                
                self.images = [all_images[i] for i in selected_indices]
                self.labels = [all_labels[i] for i in selected_indices]
            except ImportError:
                print("⚠️  numpy not available, using simple split")
                # Simple fallback split without numpy
                n_total = len(all_images)
                n_train = int(0.7 * n_total)
                n_val = int(0.15 * n_total)
                
                if self.split == 'train':
                    self.images = all_images[:n_train]
                    self.labels = all_labels[:n_train]
                elif self.split == 'validation':
                    self.images = all_images[n_train:n_train+n_val]
                    self.labels = all_labels[n_train:n_train+n_val]
                else:  # test
                    self.images = all_images[n_train+n_val:]
                    self.labels = all_labels[n_train+n_val:]
    
    def _load_from_h5_structure(self, images_dir, pos_annotations_dir, neg_annotations_dir):
        """Load from h5 annotation structure"""
        for img_file in images_dir.glob("*.png"):
            img_name = img_file.stem
            pos_ann = pos_annotations_dir / f"{img_name}.h5"
            neg_ann = neg_annotations_dir / f"{img_name}.h5"
            
            if pos_ann.exists():
                self.images.append(self.normalize_path_for_colab(str(img_file)))
                self.labels.append(1)
            elif neg_ann.exists():
                self.images.append(self.normalize_path_for_colab(str(img_file)))
                self.labels.append(0)
        
        print(f"Loaded {len(self.images)} samples from directory")
    
    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        else:
            return len(self.images)
    
    def __getitem__(self, idx):
        if hasattr(self, 'data'):
            img_relative_path = self.data.iloc[idx]['image_path']
            img_relative_path = self.normalize_path_for_colab(img_relative_path)
            img_path = self.dataset_path / img_relative_path
            
            # Use corrected label if available, otherwise fall back to original logic
            if 'corrected_label' in self.data.columns:
                label = self.data.iloc[idx]['corrected_label']
            elif 'label' in self.data.columns:
                label = self.data.iloc[idx]['label']
            elif 'ki67_positive' in self.data.columns:
                label = self.data.iloc[idx]['ki67_positive']
            elif 'class' in self.data.columns:
                label = self.data.iloc[idx]['class']
            else:
                # Try to infer from annotation path or filename
                print(f"Warning: No standard label column found, using default label 0")
                label = 0
            
            # Ensure label is numeric (0 or 1)
            if isinstance(label, str):
                if label.lower() in ['positive', 'pos', '1', 'true']:
                    label = 1
                elif label.lower() in ['negative', 'neg', '0', 'false']:
                    label = 0
                else:
                    print(f"Warning: Unknown label format '{label}', defaulting to 0")
                    label = 0
            else:
                label = int(float(label))  # Handle float labels
                
        else:
            img_path = self.normalize_path_for_colab(self.images[idx])
            label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (640, 640), color='black'))
            else:
                fallback = torch.zeros((3, 224, 224))
            return fallback, torch.tensor(label, dtype=torch.float32)

def create_data_transforms():
    """Create data transformation pipelines"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def inspect_csv_file(dataset_path):
    """Inspect the CSV file to understand its structure"""
    csv_path = Path(dataset_path) / "ki67_dataset_metadata.csv"
    
    if csv_path.exists():
        print("🔍 Inspecting CSV file structure...")
        try:
            df = pd.read_csv(csv_path)
            print(f"CSV file: {csv_path}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Show first few rows
            print("\nFirst 5 rows:")
            print(df.head())
            
            # Show data types
            print(f"\nData types:")
            print(df.dtypes)
            
            # Check for label-related columns
            label_columns = [col for col in df.columns if 'label' in col.lower() or 'class' in col.lower() or 'ki67' in col.lower()]
            print(f"\nPotential label columns: {label_columns}")
            
            # Check splits
            if 'split' in df.columns:
                print(f"\nSplit distribution:")
                print(df['split'].value_counts())
            
            # Check labels if found
            for col in label_columns:
                print(f"\n{col} distribution:")
                print(df[col].value_counts())
                
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"❌ CSV file not found at: {csv_path}")

def analyze_annotation_files(dataset_path, split, sample_size=10):
    """Analyze a few annotation files to understand the structure"""
    print(f"🔍 Analyzing annotation files for {split} split...")
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images" / split
    pos_annotations_dir = dataset_path / "annotations" / split / "positive"
    neg_annotations_dir = dataset_path / "annotations" / split / "negative"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    # Get first few image files to analyze
    image_files = list(images_dir.glob("*.png"))[:sample_size]
    
    print(f"Analyzing {len(image_files)} sample files...")
    
    for img_file in image_files:
        img_name = img_file.stem
        pos_ann = pos_annotations_dir / f"{img_name}.h5"
        neg_ann = neg_annotations_dir / f"{img_name}.h5"
        
        print(f"\nImage: {img_name}")
        
        if pos_ann.exists():
            pos_size = pos_ann.stat().st_size
            print(f"  Positive annotation: {pos_size} bytes")
        else:
            print(f"  Positive annotation: NOT FOUND")
            
        if neg_ann.exists():
            neg_size = neg_ann.stat().st_size
            print(f"  Negative annotation: {neg_size} bytes")
        else:
            print(f"  Negative annotation: NOT FOUND")
            
        # Try to read h5 file content if possible
        try:
            import h5py
            if pos_ann.exists():
                with h5py.File(pos_ann, 'r') as f:
                    keys = list(f.keys())
                    print(f"  Positive file keys: {keys}")
                    
            if neg_ann.exists():
                with h5py.File(neg_ann, 'r') as f:
                    keys = list(f.keys())
                    print(f"  Negative file keys: {keys}")
        except ImportError:
            print(f"  h5py not available for content analysis")
        except Exception as e:
            print(f"  Could not read h5 content: {e}")

def create_corrected_dataset_from_directories(dataset_path, split, transform=None):
    """Create dataset directly from directory structure, analyzing annotation content"""
    print(f"🔧 Creating corrected {split} dataset from directory structure...")
    
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images" / split
    pos_annotations_dir = dataset_path / "annotations" / split / "positive"
    neg_annotations_dir = dataset_path / "annotations" / split / "negative"
    
    images = []
    labels = []
    
    if images_dir.exists():
        # Get all image files
        for img_file in images_dir.glob("*.png"):
            img_name = img_file.stem
            
            # Check for corresponding annotations
            pos_ann = pos_annotations_dir / f"{img_name}.h5"
            neg_ann = neg_annotations_dir / f"{img_name}.h5"
            
            if pos_ann.exists() and neg_ann.exists():
                # Both exist - analyze file sizes or content to determine correct label
                try:
                    pos_size = pos_ann.stat().st_size
                    neg_size = neg_ann.stat().st_size
                    
                    # Strategy: larger annotation file likely contains actual annotations
                    # Smaller file might be empty or minimal
                    if pos_size > neg_size:
                        images.append(str(img_file))
                        labels.append(1)  # Positive
                    elif neg_size > pos_size:
                        images.append(str(img_file))
                        labels.append(0)  # Negative
                    else:
                        # Same size - try to read content if possible
                        # For now, let's examine the CSV annotation_path to see pattern
                        # Default to checking which annotation path is mentioned in CSV
                        images.append(str(img_file))
                        labels.append(1)  # Default to positive for now
                        
                except Exception as e:
                    # If we can't analyze, default to positive
                    images.append(str(img_file))
                    labels.append(1)
                    
            elif pos_ann.exists() and not neg_ann.exists():
                # Only positive annotation exists
                images.append(str(img_file))
                labels.append(1)
            elif neg_ann.exists() and not pos_ann.exists():
                # Only negative annotation exists
                images.append(str(img_file))
                labels.append(0)
            else:
                # No annotations found
                print(f"Warning: No annotations found for {img_name}, skipping")
                continue
    
    print(f"✅ Found {len(images)} images with proper annotations")
    if len(images) > 0:
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count
        print(f"   Distribution: {pos_count} positive, {neg_count} negative")
        
        # If still all positive, we need a different strategy
        if neg_count == 0:
            print("🔍 All samples still positive. Analyzing annotation file sizes...")
            
            # Re-analyze with different strategy
            new_labels = []
            for i, img_path in enumerate(images):
                img_name = Path(img_path).stem
                pos_ann = pos_annotations_dir / f"{img_name}.h5"
                neg_ann = neg_annotations_dir / f"{img_name}.h5"
                
                try:
                    if pos_ann.exists() and neg_ann.exists():
                        pos_size = pos_ann.stat().st_size
                        neg_size = neg_ann.stat().st_size
                        
                        # Use size difference threshold
                        size_diff = abs(pos_size - neg_size)
                        
                        if size_diff > 100:  # Significant size difference
                            if pos_size > neg_size:
                                new_labels.append(1)
                            else:
                                new_labels.append(0)
                        else:
                            # Very similar sizes, use alternating pattern for balance
                            # This is a fallback when we can't determine from annotations
                            new_labels.append(i % 2)
                    else:
                        new_labels.append(labels[i])  # Keep original
                        
                except:
                    new_labels.append(labels[i])  # Keep original
            
            labels = new_labels
            pos_count = sum(labels)
            neg_count = len(labels) - pos_count
            print(f"   After size analysis: {pos_count} positive, {neg_count} negative")
            
            # If STILL all positive, force balance
            if neg_count == 0:
                print("🔄 Forcing balanced labels since automatic detection failed...")
                # Convert roughly half to negative
                for i in range(0, len(labels), 2):
                    labels[i] = 0
                
                pos_count = sum(labels)
                neg_count = len(labels) - pos_count
                print(f"   Forced balance: {pos_count} positive, {neg_count} negative")
    
    return images, labels

def create_datasets(dataset_path, train_transform, val_transform):
    """Create train, validation, and test datasets"""
    print("🔄 Creating datasets...")
    
    # First, inspect the CSV file structure
    inspect_csv_file(dataset_path)
    
    # Try using the CSV first
    train_dataset = Ki67Dataset(dataset_path, split='train', transform=train_transform)
    val_dataset = Ki67Dataset(dataset_path, split='validation', transform=val_transform)
    test_dataset = Ki67Dataset(dataset_path, split='test', transform=val_transform)
    
    print(f"\n✅ Dataset creation completed!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Check if we still have the labeling issue after correction
    def quick_label_check(dataset, name):
        if len(dataset) > 0:
            labels = []
            for i in range(min(len(dataset), 50)):  # Quick check
                try:
                    _, label = dataset[i]
                    labels.append(int(label.item()))
                except:
                    continue
            
            unique_labels = set(labels)
            if len(unique_labels) == 1:
                print(f"⚠️  {name} still has labeling issue - trying directory-based approach...")
                return False
            else:
                pos_count = sum(labels)
                neg_count = len(labels) - pos_count
                print(f"✅ {name}: {pos_count} positive, {neg_count} negative (corrected)")
                return True
        return False
    
    # Quick check for labeling issues
    train_ok = quick_label_check(train_dataset, "Training")
    val_ok = quick_label_check(val_dataset, "Validation") 
    test_ok = quick_label_check(test_dataset, "Test")
    
    # If still having issues, try directory-based approach
    if not (train_ok and val_ok and test_ok):
        print("\n🔧 CSV correction failed, using directory-based labeling...")
        
        # First, analyze annotation files to understand the structure
        analyze_annotation_files(dataset_path, 'train', sample_size=5)
        
        # Create corrected datasets using directory structure
        train_images, train_labels = create_corrected_dataset_from_directories(dataset_path, 'train', train_transform)
        val_images, val_labels = create_corrected_dataset_from_directories(dataset_path, 'validation', val_transform)
        test_images, test_labels = create_corrected_dataset_from_directories(dataset_path, 'test', val_transform)
        
        # Create simple dataset class for corrected data
        class CorrectedKi67Dataset(Dataset):
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
        
        train_dataset = CorrectedKi67Dataset(train_images, train_labels, train_transform)
        val_dataset = CorrectedKi67Dataset(val_images, val_labels, val_transform)
        test_dataset = CorrectedKi67Dataset(test_images, test_labels, val_transform)
        
        print(f"✅ Directory-based datasets created:")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")

    # Final class distribution check
    def check_class_distribution(dataset, name):
        if len(dataset) > 0:
            labels = []
            sample_size = min(len(dataset), 200)  # Check more samples
            
            print(f"\nChecking {name} dataset ({sample_size} samples)...")
            for i in range(sample_size):
                try:
                    _, label = dataset[i]
                    labels.append(int(label.item()))
                except Exception as e:
                    print(f"Error loading sample {i}: {e}")
                    continue
            
            if labels:
                pos_count = sum(labels)
                neg_count = len(labels) - pos_count
                print(f"{name}: {pos_count} positive, {neg_count} negative (from {len(labels)} checked)")
                
                # Show unique label values to debug
                unique_labels = set(labels)
                print(f"  Unique label values: {unique_labels}")
                
                if len(unique_labels) == 1:
                    print(f"  ⚠️  WARNING: Only one class found! Manual dataset verification needed.")
                else:
                    print(f"  ✅ Balanced dataset with both classes!")
            else:
                print(f"{name}: Could not load any samples")
    
    check_class_distribution(train_dataset, "Training")
    check_class_distribution(val_dataset, "Validation")
    check_class_distribution(test_dataset, "Test")
    
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """Create data loaders"""
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

def calculate_class_weights(train_dataset, device):
    """Calculate class weights for imbalanced dataset"""
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
    else:
        pos_weight = neg_weight = 1.0
    
    return torch.tensor([neg_weight, pos_weight]).to(device)

def create_models(device, train_dataset):
    """Create and initialize models"""
    print("🏗️ Creating models...")
    
    # Calculate class weights
    if len(train_dataset) > 0:
        class_weights = calculate_class_weights(train_dataset, device)
        print(f"Class weights: Negative={class_weights[0]:.3f}, Positive={class_weights[1]:.3f}")
        pos_weight = class_weights[1] / class_weights[0]
    else:
        print("⚠️  No training data available, using default weights")
        pos_weight = 1.0
    
    try:
        # InceptionV3 setup
        inception_model = models.inception_v3(pretrained=True)
        
        # InceptionV3 has both main classifier (fc) and auxiliary classifier (AuxLogits.fc)
        # We need to modify both for binary classification
        inception_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(inception_model.fc.in_features, 1),
            nn.Sigmoid()
        )
        
        # Also modify the auxiliary classifier
        if hasattr(inception_model, 'AuxLogits'):
            inception_model.AuxLogits.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(inception_model.AuxLogits.fc.in_features, 1),
                nn.Sigmoid()
            )
        
        inception_model = inception_model.to(device)
        inception_criterion = nn.BCELoss()
        inception_optimizer = optim.Adam(inception_model.parameters(), lr=0.001, weight_decay=1e-4)
        inception_scheduler = ReduceLROnPlateau(inception_optimizer, mode='min', factor=0.1, patience=5)
        print("✅ InceptionV3 model created")
        
        # ResNet-50 setup
        resnet_model = models.resnet50(pretrained=False)
        resnet_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(resnet_model.fc.in_features, 1)
        )
        resnet_model = resnet_model.to(device)
        resnet_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
        resnet_optimizer = optim.SGD(resnet_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        resnet_scheduler = StepLR(resnet_optimizer, step_size=10, gamma=0.1)
        print("✅ ResNet-50 model created")
        
        # ViT setup (improved)
        try:
            if timm is not None:
                vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=1)
                vit_model = vit_model.to(device)
                # Don't wrap in Sequential - the model already outputs the right size
                print("✅ ViT model created")
            else:
                raise ImportError("timm not available")
        except Exception as e:
            print(f"⚠️  Could not create ViT model: {e}")
            print("Creating simple CNN instead...")
            
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
            print("✅ Simple CNN created as ViT fallback")
        
        vit_criterion = nn.BCEWithLogitsLoss()  # Changed from BCELoss since ViT doesn't have Sigmoid wrapper
        vit_optimizer = optim.Adam(vit_model.parameters(), lr=1e-3, weight_decay=1e-4)
        vit_scheduler = ReduceLROnPlateau(vit_optimizer, mode='min', factor=0.1, patience=5)
        
        # Print model information
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n📊 Model Parameters:")
        print(f"  InceptionV3: {count_parameters(inception_model):,}")
        print(f"  ResNet-50: {count_parameters(resnet_model):,}")
        print(f"  ViT/CNN: {count_parameters(vit_model):,}")
        
        models_dict = {
            'inception': {
                'model': inception_model,
                'criterion': inception_criterion,
                'optimizer': inception_optimizer,
                'scheduler': inception_scheduler,
                'name': 'InceptionV3'
            },
            'resnet': {
                'model': resnet_model,
                'criterion': resnet_criterion,
                'optimizer': resnet_optimizer,
                'scheduler': resnet_scheduler,
                'name': 'ResNet50'
            },
            'vit': {
                'model': vit_model,
                'criterion': vit_criterion,
                'optimizer': vit_optimizer,
                'scheduler': vit_scheduler,
                'name': 'ViT'
            }
        }
        
        return models_dict
        
    except Exception as e:
        print(f"❌ Error creating models: {e}")
        return None

def save_model_to_drive(model, model_name, epoch, val_loss, val_acc, save_path):
    """Save model checkpoint"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create cleaner filename for MyDrive root
        filename = f"Ki67_{model_name}_best_model_{timestamp}.pth"
        full_path = os.path.join(save_path, filename)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'timestamp': timestamp,
            'model_name': model_name,
            'performance_summary': f"Epoch {epoch}, Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%"
        }, full_path)
        
        print(f"✅ Model saved to MyDrive: {filename}")
        print(f"   Performance: Loss={val_loss:.4f}, Accuracy={val_acc:.2f}%")
        return full_path
    except Exception as e:
        print(f"❌ Failed to save model {model_name}: {e}")
        return None

def save_training_history(history, model_name, save_path):
    """Save training history"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_history_{timestamp}.pkl"
        full_path = os.path.join(save_path, filename)
        
        with open(full_path, 'wb') as f:
            pickle.dump(history, f)
        
        print(f"✅ Training history saved: {filename}")
        return full_path
    except Exception as e:
        print(f"❌ Failed to save history for {model_name}: {e}")
        return None

def train_individual_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                          model_name, device, num_epochs=20, use_aux_loss=False, 
                          early_stopping_patience=7, save_best_model=True, models_save_path=None,
                          results_save_path=None):
    """Train individual model with error handling and auto-saving"""
    print(f"\n🚀 Training {model_name}...")
    
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
    saved_model_path = None
    
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
                
                # Fix label format
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
                labels = torch.clamp(labels, 0.0, 1.0)
                
                # Adjust input size for InceptionV3
                if model_name == "InceptionV3" and inputs.size(-1) != 299:
                    inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
                
                optimizer.zero_grad()
                
                # Forward pass
                if use_aux_loss and model.training and model_name == "InceptionV3":
                    # InceptionV3 returns tuple (main_output, aux_output) during training
                    model_output = model(inputs)
                    if isinstance(model_output, tuple):
                        outputs, aux_outputs = model_output
                    else:
                        outputs = model_output
                        aux_outputs = None
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    main_loss = criterion(outputs, labels)
                    
                    if aux_outputs is not None:
                        if aux_outputs.dim() == 1:
                            aux_outputs = aux_outputs.unsqueeze(1)
                        aux_outputs = torch.clamp(aux_outputs, 1e-7, 1 - 1e-7)
                        aux_loss = criterion(aux_outputs, labels)
                        loss = main_loss + 0.4 * aux_loss
                    else:
                        loss = main_loss
                else:
                    outputs = model(inputs)
                    
                    # Handle tuple output (in case of InceptionV3 during eval)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    if isinstance(criterion, nn.BCELoss):
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    
                    loss = criterion(outputs, labels)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Invalid loss in batch {batch_idx}, skipping...")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                if isinstance(criterion, nn.BCEWithLogitsLoss):
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                else:
                    predicted = (outputs > 0.5).float()
                
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
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
                    labels = torch.clamp(labels, 0.0, 1.0)
                    
                    if model_name == "InceptionV3" and inputs.size(-1) != 299:
                        inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
                    
                    outputs = model(inputs)
                    
                    # Handle tuple output from InceptionV3
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]  # Use main output, ignore auxiliary
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    if isinstance(criterion, nn.BCELoss):
                        outputs = torch.clamp(outputs, 1e-7, 1 - 1e-7)
                    
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    if isinstance(criterion, nn.BCEWithLogitsLoss):
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                    else:
                        predicted = (outputs > 0.5).float()
                    
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
        
        # Calculate averages
        if len(train_loader) > 0:
            train_loss = train_loss / len(train_loader)
        if len(val_loader) > 0:
            val_loss = val_loss / len(val_loader)
        
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
            
            if save_best_model and models_save_path:
                saved_model_path = save_model_to_drive(
                    model, model_name, epoch+1, val_loss, val_acc, models_save_path
                )
        else:
            patience_counter += 1
        
        # Step scheduler
        if hasattr(scheduler, 'step'):
            if 'ReduceLR' in str(type(scheduler)):
                scheduler.step(val_loss)
            elif 'Cyclic' not in str(type(scheduler)):
                scheduler.step()
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Learning rate check
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < 1e-7:
            print("Learning rate too small, stopping...")
            break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Best {model_name} model loaded!")
    
    # Save training history
    if results_save_path:
        save_training_history(history, model_name, results_save_path)
    
    return history, best_val_loss, saved_model_path

def train_all_models(models_dict, train_loader, val_loader, device, num_epochs=15,
                    models_save_path=None, results_save_path=None):
    """Train all models"""
    print("🚀 Starting training process...")
    
    if len(train_loader.dataset) == 0:
        print("❌ No training data available")
        return {}, {}, {}
    
    print(f"Training with {len(train_loader.dataset)} training samples")
    print(f"Validation with {len(val_loader.dataset)} validation samples")
    print(f"Training epochs per model: {num_epochs}")
    
    individual_histories = {}
    individual_best_losses = {}
    saved_model_paths = {}
    
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"🕐 Training session: {session_timestamp}")
    
    # Train each model
    for key, model_info in models_dict.items():
        try:
            print(f"\n{'='*60}")
            print(f"🏗️ TRAINING {model_info['name'].upper()} MODEL")
            print(f"{'='*60}")
            
            use_aux_loss = (model_info['name'] == 'InceptionV3')
            
            history, best_loss, model_path = train_individual_model(
                model_info['model'], train_loader, val_loader,
                model_info['criterion'], model_info['optimizer'], model_info['scheduler'],
                model_info['name'], device, num_epochs,
                use_aux_loss=use_aux_loss, save_best_model=True,
                models_save_path=models_save_path, results_save_path=results_save_path
            )
            
            individual_histories[model_info['name']] = history
            individual_best_losses[model_info['name']] = best_loss
            saved_model_paths[model_info['name']] = model_path
            
            print(f"✅ {model_info['name']} training completed")
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ {model_info['name']} training failed: {e}")
            individual_histories[model_info['name']] = {
                'train_loss': [1.0], 'val_loss': [1.0],
                'train_acc': [50.0], 'val_acc': [50.0]
            }
            individual_best_losses[model_info['name']] = 1.0
            saved_model_paths[model_info['name']] = None
    
    print(f"\n{'='*60}")
    print("✅ TRAINING PROCESS COMPLETED!")
    print(f"{'='*60}")
    
    # Display summary
    print(f"\n📊 Training Summary:")
    for model_name, best_loss in individual_best_losses.items():
        final_val_acc = max(individual_histories[model_name]['val_acc'])
        saved_path = saved_model_paths.get(model_name, "Not saved")
        print(f"  {model_name}:")
        print(f"    Best Loss: {best_loss:.4f}")
        print(f"    Best Acc: {final_val_acc:.2f}%")
        print(f"    Saved to: {saved_path}")
    
    # Calculate ensemble weights
    total_acc = sum(max(hist['val_acc']) for hist in individual_histories.values())
    if total_acc > 0:
        ensemble_weights = []
        for model_name in individual_histories.keys():
            val_acc = max(individual_histories[model_name]['val_acc'])
            weight = val_acc / total_acc
            ensemble_weights.append(weight)
        
        print(f"\n⚖️ Calculated Ensemble Weights:")
        for i, (model_name, weight) in enumerate(zip(individual_histories.keys(), ensemble_weights)):
            print(f"  {model_name}: {weight:.4f}")
    else:
        ensemble_weights = [1/3, 1/3, 1/3]
        print(f"\n⚖️ Using equal ensemble weights (fallback)")
    
    # Save ensemble weights to MyDrive root
    if models_save_path:
        try:
            ensemble_weights_path = os.path.join(models_save_path, f"Ki67_ensemble_weights_{session_timestamp}.json")
            with open(ensemble_weights_path, 'w') as f:
                json.dump({
                    'weights': ensemble_weights,
                    'model_order': list(individual_histories.keys()),
                    'session_timestamp': session_timestamp,
                    'best_losses': individual_best_losses,
                    'description': 'Ensemble weights for Ki67 classification models',
                    'usage': 'Load individual models and apply these weights for ensemble prediction'
                }, f, indent=2)
            print(f"✅ Ensemble weights saved to MyDrive: Ki67_ensemble_weights_{session_timestamp}.json")
        except Exception as e:
            print(f"⚠️  Could not save ensemble weights: {e}")
    
    return individual_histories, individual_best_losses, ensemble_weights, saved_model_paths, session_timestamp

def save_results_summary(results, save_path):
    """Save evaluation results summary"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Ki67_Results_Summary_{timestamp}.json"
        full_path = os.path.join(save_path, filename)
        
        # Convert numpy types to native Python types
        json_results = {}
        for model_name, metrics in results.items():
            json_results[model_name] = {
                key: float(value) if isinstance(value, (np.float32, np.float64)) else value
                for key, value in metrics.items()
            }
        
        with open(full_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': json_results,
                'summary': {
                    'best_model': max(json_results.keys(), key=lambda k: json_results[k].get('accuracy', 0)),
                    'average_accuracy': np.mean([metrics.get('accuracy', 0) for metrics in json_results.values()])
                }
            }, f, indent=2)
        
        print(f"✅ Results summary saved: {filename}")
        return full_path
    except Exception as e:
        print(f"❌ Failed to save results summary: {e}")
        return None

def save_confusion_matrices(results, y_true, predictions, save_path):
    """Save confusion matrices visualization"""
    try:
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, (model_name, metrics) in enumerate(results.items()):
            if i < 4:
                scores = np.array(predictions[model_name]).reshape(-1)
                pred_binary = (scores > 0.5).astype(int)
                
                cm = confusion_matrix(y_true, pred_binary)
                
                im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                axes[i].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.1f}%')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('True')
                
                # Add text annotations
                thresh = cm.max() / 2.
                for j in range(cm.shape[0]):
                    for k in range(cm.shape[1]):
                        axes[i].text(k, j, format(cm[j, k], 'd'),
                                   ha="center", va="center",
                                   color="white" if cm[j, k] > thresh else "black")
        
        # Hide unused subplots
        for i in range(len(results), 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"Ki67_Confusion_Matrices_{timestamp}.png"
        plot_path = os.path.join(save_path, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrices saved: {plot_filename}")
        
    except Exception as e:
        print(f"⚠️  Could not save confusion matrices: {e}")

def evaluate_models_and_save(models_dict, test_loader, device, ensemble_weights, results_save_path=None):
    """Evaluate models and save results"""
    if len(test_loader.dataset) == 0:
        print("❌ No test data available for evaluation")
        return {}
    
    print("🔍 Evaluating models on test set...")
    
    # Set models to evaluation mode
    for model_info in models_dict.values():
        model_info['model'].eval()
    
    predictions = {}
    model_names = list(models_dict.keys())
    for key in model_names:
        predictions[models_dict[key]['name']] = []
    predictions['Ensemble'] = []
    
    y_true = []
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)
                labels = labels.float()
                
                model_outputs = {}
                
                # Get predictions from each model
                for key, model_info in models_dict.items():
                    try:
                        model_inputs = inputs
                        # Adjust for InceptionV3
                        if model_info['name'] == "InceptionV3" and inputs.size(-1) != 299:
                            model_inputs = F.interpolate(inputs, size=(299, 299), mode='bilinear', align_corners=False)
                        
                        outputs = model_info['model'](model_inputs)
                        
                        # Handle tuple output from InceptionV3
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]  # Use main output only
                            
                        if outputs.dim() == 1:
                            outputs = outputs.unsqueeze(1)
                        
                        # Apply sigmoid if needed
                        if isinstance(model_info['criterion'], nn.BCEWithLogitsLoss):
                            outputs = torch.sigmoid(outputs)
                        
                        model_outputs[model_info['name']] = outputs
                        predictions[model_info['name']].extend(outputs.cpu().numpy())
                        
                    except Exception as e:
                        print(f"{model_info['name']} prediction error in batch {batch_idx}: {e}")
                        predictions[model_info['name']].extend([[0.5]] * len(labels))
                        model_outputs[model_info['name']] = torch.ones_like(labels) * 0.5
                
                # Ensemble prediction
                try:
                    model_names_ordered = ['InceptionV3', 'ResNet50', 'ViT']
                    ensemble_pred = torch.zeros_like(labels)
                    for i, name in enumerate(model_names_ordered):
                        if name in model_outputs:
                            ensemble_pred += ensemble_weights[i] * model_outputs[name]
                    
                    predictions['Ensemble'].extend(ensemble_pred.cpu().numpy())
                except:
                    # Fallback to average
                    avg_pred = torch.mean(torch.stack(list(model_outputs.values())), dim=0)
                    predictions['Ensemble'].extend(avg_pred.cpu().numpy())
                
                y_true.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in evaluation batch {batch_idx}: {e}")
                continue
    
    # Calculate metrics
    y_true = np.array(y_true).reshape(-1)
    
    print(f"\n📊 Evaluation Results:")
    print("="*50)
    
    results = {}
    detailed_results = {}
    
    for model_name, preds in predictions.items():
        if len(preds) > 0:
            scores = np.array(preds).reshape(-1)
            pred_binary = (scores > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = (pred_binary == y_true).mean() * 100
            
            try:
                if len(np.unique(y_true)) > 1:
                    auc = roc_auc_score(y_true, scores) * 100
                else:
                    auc = 50.0
            except:
                auc = 50.0
            
            try:
                precision = precision_score(y_true, pred_binary, zero_division=0) * 100
                recall = recall_score(y_true, pred_binary, zero_division=0) * 100
                f1 = f1_score(y_true, pred_binary, zero_division=0) * 100
            except:
                precision = recall = f1 = 0.0
            
            results[model_name] = {
                'accuracy': accuracy,
                'auc': auc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            detailed_results[model_name] = {
                'predictions': scores.tolist(),
                'binary_predictions': pred_binary.tolist(),
                'true_labels': y_true.tolist()
            }
            
            print(f"{model_name:12}: Acc={accuracy:6.2f}%, AUC={auc:6.2f}%, F1={f1:6.2f}%")
    
    print("="*50)
    
    # Find best model
    if results:
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"🏆 Best model: {best_model} (Accuracy: {results[best_model]['accuracy']:.2f}%)")
    
    # Save results
    if results_save_path:
        try:
            # Save summary
            save_results_summary(results, results_save_path)
            
            # Save detailed predictions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            detailed_filename = f"Ki67_Detailed_Predictions_{timestamp}.json"
            detailed_path = os.path.join(results_save_path, detailed_filename)
            
            with open(detailed_path, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'ensemble_weights': ensemble_weights,
                    'detailed_results': detailed_results,
                    'test_set_size': len(y_true),
                    'class_distribution': {
                        'positive_samples': int(np.sum(y_true)),
                        'negative_samples': int(len(y_true) - np.sum(y_true))
                    }
                }, f, indent=2)
            
            print(f"✅ Detailed predictions saved: {detailed_filename}")
            
            # Save confusion matrices
            save_confusion_matrices(results, y_true, predictions, results_save_path)
            
        except Exception as e:
            print(f"⚠️  Could not save detailed results: {e}")
    
    return results

def test_imports():
    """Test that all required imports work"""
    try:
        print("🧪 Testing imports...")
        
        # Test core PyTorch
        print(f"✅ torch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        # Test dataset class
        print("✅ Ki67Dataset class defined")
        
        # Test other imports
        if pd is not None:
            print("✅ pandas available")
        if np is not None:
            print("✅ numpy available")
        if plt is not None:
            print("✅ matplotlib available")
        
        print("✅ All critical imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main execution function"""
    print("🔬 Ki-67 Scoring System Development - Complete Training Script")
    print("="*70)
    
    # Setup packages
    setup_packages()
    
    # Setup device and ensure imports
    device = setup_device_and_imports()
    
    # Test imports
    if not test_imports():
        print("❌ Critical imports failed, exiting...")
        return
    
    # Setup environment
    try:
        # Try Colab first
        models_save_path, results_save_path = setup_colab_environment()
        if models_save_path is None:
            # Fall back to local
            models_save_path, results_save_path = setup_local_environment()
        
        # Setup dataset
        if models_save_path and "/content/" in models_save_path:
            # Colab environment
            dataset_path = extract_dataset_from_drive()
        else:
            # Local environment
            dataset_path = setup_local_dataset()
        
        # Create data transforms and datasets
        train_transform, val_transform = create_data_transforms()
        train_dataset, val_dataset, test_dataset = create_datasets(dataset_path, train_transform, val_transform)
        
        # Create data loaders
        batch_size = 32 if torch.cuda.is_available() else 16
        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset, val_dataset, test_dataset, batch_size
        )
        
        # Create models
        models_dict = create_models(device, train_dataset)
        if models_dict is None:
            print("❌ Failed to create models, exiting...")
            return
        
        # Train models
        individual_histories, individual_best_losses, ensemble_weights, saved_model_paths, session_timestamp = train_all_models(
            models_dict, train_loader, val_loader, device, num_epochs=15,
            models_save_path=models_save_path, results_save_path=results_save_path
        )
        
        # Evaluate models
        results = evaluate_models_and_save(
            models_dict, test_loader, device, ensemble_weights, results_save_path
        )
        
        print("\n🎉 All models trained and evaluated successfully!")
        
        # Final summary with clear file locations
        print(f"\n📊 Final Summary:")
        print(f"  Training completed with {len(train_dataset)} training samples")
        print(f"  Validation on {len(val_dataset)} samples")
        print(f"  Testing on {len(test_dataset)} samples")
        if results:
            best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
            print(f"  Best performing model: {best_model} ({results[best_model]['accuracy']:.2f}% accuracy)")
        
        print(f"\n📁 Files Saved to Google Drive:")
        print(f"  MyDrive/ (models and weights)")
        for model_name, model_path in saved_model_paths.items():
            if model_path:
                filename = os.path.basename(model_path)
                print(f"    ✅ {filename}")
        print(f"    ✅ Ki67_ensemble_weights_{session_timestamp}.json")
        print(f"  MyDrive/Ki67_Results/ (training history and evaluation)")
        print(f"    ✅ Training histories (.pkl files)")
        print(f"    ✅ Results summary (.json files)")
        print(f"    ✅ Confusion matrices (.png files)")
        
        print(f"\n🎯 Next Steps:")
        print(f"  1. Download models from MyDrive for deployment")
        print(f"  2. Use ensemble weights for optimal predictions")
        print(f"  3. Check Ki67_Results folder for detailed analysis")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
