#!/usr/bin/env python3
"""
TUPAC16 Dataset Download and Setup Guide

TUPAC16 (Tumor Proliferation Assessment Challenge) is the gold standard 
dataset for Ki-67 proliferation assessment with expert pathologist annotations.
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import json
from urllib.parse import urljoin
import time

class TUPAC16Downloader:
    """
    TUPAC16 Dataset Downloader and Setup
    
    Note: TUPAC16 requires registration at http://tupac.tue-image.nl/
    This script helps with the setup process after registration.
    """
    
    def __init__(self, download_dir="TUPAC16_Dataset"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        self.dataset_info = {
            "name": "TUPAC16 - Tumor Proliferation Assessment Challenge",
            "description": "Gold standard Ki-67 dataset with expert annotations",
            "registration_url": "http://tupac.tue-image.nl/",
            "challenge_url": "https://tupac.grand-challenge.org/",
            "paper": "Veta et al., 2019 - Assessment of algorithms for mitosis detection",
            "size": "500+ breast cancer images",
            "annotation_types": [
                "Ki-67 proliferation index ground truth",
                "Mitotic figure annotations", 
                "Region of interest boundaries",
                "Expert pathologist scores"
            ],
            "image_types": [
                "H&E stained histology images",
                "Ki-67 IHC stained images"
            ]
        }
    
    def print_registration_guide(self):
        """Print step-by-step registration guide"""
        print("🏆 TUPAC16 DATASET - REGISTRATION GUIDE")
        print("=" * 60)
        print("TUPAC16 is the GOLD STANDARD for Ki-67 research!")
        print()
        
        print("📋 STEP-BY-STEP REGISTRATION:")
        print("=" * 40)
        print("1. 🌐 Visit: http://tupac.tue-image.nl/")
        print("2. 📝 Click 'Register' or 'Download'")
        print("3. 🔐 Create account with:")
        print("   • Your email address")
        print("   • Institution name")
        print("   • Research purpose")
        print("4. ✅ Confirm email verification")
        print("5. 📥 Download dataset files")
        print()
        
        print("📦 WHAT YOU'LL GET:")
        print("=" * 30)
        for item in self.dataset_info["annotation_types"]:
            print(f"   • {item}")
        print()
        for item in self.dataset_info["image_types"]:
            print(f"   • {item}")
        print()
        
        print("🎯 WHY TUPAC16 IS SUPERIOR:")
        print("=" * 35)
        print("✅ Expert pathologist annotations")
        print("✅ Standardized Ki-67 scoring protocol")
        print("✅ Benchmark dataset for competitions")
        print("✅ Multiple annotation levels")
        print("✅ Clinical ground truth")
        print("✅ Published evaluation metrics")
        print()
        
        print("🚫 PROBLEMS WITH YOUR CURRENT APPROACH:")
        print("=" * 45)
        print("❌ File size classification is unreliable")
        print("❌ No biological basis for ground truth")
        print("❌ Arbitrary 100-byte threshold")
        print("❌ Random assignment for similar sizes")
        print("❌ Not scientifically publishable")
    
    def create_download_urls(self):
        """Create list of known TUPAC16 download URLs"""
        # Note: These are example URLs - actual URLs require authentication
        urls = {
            "training_data": "Training data with annotations",
            "test_data": "Test data for evaluation",
            "mitosis_annotations": "Mitotic figure ground truth",
            "ki67_scores": "Ki-67 proliferation indices",
            "evaluation_kit": "Evaluation scripts and metrics"
        }
        
        print("📥 TUPAC16 DOWNLOAD COMPONENTS:")
        print("=" * 40)
        for component, description in urls.items():
            print(f"• {component}: {description}")
        
        return urls
    
    def check_download_status(self):
        """Check if TUPAC16 files are already downloaded"""
        print(f"\n🔍 Checking download directory: {self.download_dir}")
        
        if not self.download_dir.exists():
            print("❌ Download directory doesn't exist")
            return False
        
        # Look for typical TUPAC16 files
        expected_files = [
            "*.tif", "*.tiff",  # Image files
            "*.csv", "*.txt",   # Annotation files
            "*.xml",            # Metadata
            "*.zip"             # Archives
        ]
        
        found_files = []
        for pattern in expected_files:
            found_files.extend(list(self.download_dir.glob(pattern)))
        
        if found_files:
            print(f"✅ Found {len(found_files)} TUPAC16 files:")
            for f in found_files[:5]:  # Show first 5
                print(f"   • {f.name}")
            if len(found_files) > 5:
                print(f"   ... and {len(found_files) - 5} more")
            return True
        else:
            print("❌ No TUPAC16 files found")
            return False
    
    def setup_directory_structure(self):
        """Create proper directory structure for TUPAC16"""
        dirs = [
            "images/training",
            "images/test", 
            "annotations/mitosis",
            "annotations/ki67",
            "metadata",
            "evaluation"
        ]
        
        print(f"\n📁 Setting up directory structure...")
        for dir_path in dirs:
            full_path = self.download_dir / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {dir_path}")
        
        print("✅ Directory structure ready")
    
    def create_tupac16_dataset_class(self):
        """Create a dataset class for TUPAC16"""
        
        dataset_code = '''#!/usr/bin/env python3
"""
TUPAC16 Dataset Class - Gold Standard Ki-67 Dataset

Proper dataset class for TUPAC16 with expert Ki-67 annotations.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET

class TUPAC16Dataset(Dataset):
    """
    TUPAC16 Dataset Class
    
    Gold standard dataset for Ki-67 proliferation assessment with
    expert pathologist annotations and standardized scoring protocol.
    """
    
    def __init__(self, root_dir, split='train', transform=None, task='ki67'):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.task = task  # 'ki67' or 'mitosis'
        
        self.image_paths = []
        self.labels = []
        self.ki67_scores = []
        self.annotations = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load TUPAC16 dataset with proper annotations"""
        print(f"🏆 Loading TUPAC16 {self.split} dataset...")
        
        # Load images
        images_dir = self.root_dir / "images" / self.split
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        
        # Load Ki-67 scores if available
        ki67_file = self.root_dir / "annotations" / "ki67_scores.csv"
        ki67_data = {}
        if ki67_file.exists():
            df = pd.read_csv(ki67_file)
            ki67_data = dict(zip(df['image_id'], df['ki67_score']))
        
        # Process images
        for img_file in images_dir.glob("*.tif*"):
            img_id = img_file.stem
            
            # Get Ki-67 score
            ki67_score = ki67_data.get(img_id, 0.0)
            
            # Classification based on Ki-67 proliferation index
            # Standard threshold: >20% = high proliferation
            label = 1 if ki67_score >= 20.0 else 0
            
            self.image_paths.append(str(img_file))
            self.labels.append(label)
            self.ki67_scores.append(ki67_score)
        
        print(f"✅ Loaded {len(self.image_paths)} TUPAC16 images")
        if self.ki67_scores:
            avg_score = np.mean(self.ki67_scores)
            high_prolif = sum(1 for s in self.ki67_scores if s >= 20.0)
            print(f"   Average Ki-67 score: {avg_score:.1f}%")
            print(f"   High proliferation cases: {high_prolif}/{len(self.ki67_scores)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        ki67_score = self.ki67_scores[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'ki67_score': torch.tensor(ki67_score, dtype=torch.float32),
            'image_id': Path(image_path).stem
        }
    
    def get_statistics(self):
        """Get dataset statistics"""
        return {
            'total_images': len(self.image_paths),
            'high_proliferation': sum(self.labels),
            'low_proliferation': len(self.labels) - sum(self.labels),
            'avg_ki67_score': np.mean(self.ki67_scores) if self.ki67_scores else 0,
            'ki67_score_std': np.std(self.ki67_scores) if self.ki67_scores else 0,
            'score_range': [min(self.ki67_scores), max(self.ki67_scores)] if self.ki67_scores else [0, 0]
        }

# Example usage
if __name__ == "__main__":
    # Create transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    try:
        dataset = TUPAC16Dataset(
            root_dir="TUPAC16_Dataset",
            split="train",
            transform=transform,
            task="ki67"
        )
        
        print(f"\\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\\nSample data:")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label: {sample['label'].item()}")
            print(f"  Ki-67 score: {sample['ki67_score'].item():.1f}%")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please download TUPAC16 dataset first!")
'''
        
        with open("tupac16_dataset.py", "w") as f:
            f.write(dataset_code)
        
        print(f"📝 Created tupac16_dataset.py")
        print(f"   Professional dataset class for TUPAC16")
    
    def create_comparison_script(self):
        """Create script to compare TUPAC16 vs current file-size method"""
        
        comparison_code = '''#!/usr/bin/env python3
"""
TUPAC16 vs File-Size Method Comparison

Compare the gold standard TUPAC16 approach with the current file-size method.
"""

import numpy as np
import matplotlib.pyplot as plt
from tupac16_dataset import TUPAC16Dataset
from improved_ki67_dataset import ImprovedKi67Dataset
import torchvision.transforms as transforms

def compare_classification_approaches():
    """Compare TUPAC16 vs file-size classification"""
    print("🏆 TUPAC16 vs FILE-SIZE METHOD COMPARISON")
    print("=" * 60)
    
    # Standard transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\\n📊 CLASSIFICATION METHODS:")
    print("-" * 40)
    
    # Method 1: Current file-size approach
    print("\\n1. 🔧 CURRENT FILE-SIZE METHOD:")
    try:
        current_dataset = ImprovedKi67Dataset(
            dataset_path=".",
            classification_method='file_size_legacy',
            transform=transform
        )
        current_stats = current_dataset.get_annotation_statistics()
        print(f"   ✅ Loaded: {current_stats['total_images']} images")
        print(f"   📊 Positive: {current_stats['positive_images']}")
        print(f"   📊 Negative: {current_stats['negative_images']}")
        print(f"   ⚠️  Method: File size differences (unreliable)")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: TUPAC16 gold standard
    print("\\n2. 🏆 TUPAC16 GOLD STANDARD:")
    try:
        tupac_dataset = TUPAC16Dataset(
            root_dir="TUPAC16_Dataset",
            split="train",
            transform=transform
        )
        tupac_stats = tupac_dataset.get_statistics()
        print(f"   ✅ Loaded: {tupac_stats['total_images']} images")
        print(f"   📊 High proliferation: {tupac_stats['high_proliferation']}")
        print(f"   📊 Low proliferation: {tupac_stats['low_proliferation']}")
        print(f"   📊 Avg Ki-67 score: {tupac_stats['avg_ki67_score']:.1f}%")
        print(f"   ✅ Method: Expert pathologist annotations")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        print(f"   💡 Download TUPAC16 first!")
    
    print("\\n🎯 ADVANTAGES OF TUPAC16:")
    print("-" * 35)
    print("✅ Expert pathologist ground truth")
    print("✅ Standardized Ki-67 scoring protocol")
    print("✅ Clinical relevance and validity")
    print("✅ Established benchmark metrics")
    print("✅ Publishable research results")
    print("✅ International competition standard")
    
    print("\\n🚫 PROBLEMS WITH FILE-SIZE METHOD:")
    print("-" * 40)
    print("❌ No biological basis")
    print("❌ Arbitrary thresholds (100 bytes)")
    print("❌ Random assignment fallback")
    print("❌ Storage-dependent results")
    print("❌ Not scientifically valid")
    print("❌ Unreproducible across systems")

if __name__ == "__main__":
    compare_classification_approaches()
'''
        
        with open("tupac16_comparison.py", "w") as f:
            f.write(comparison_code)
        
        print(f"📝 Created tupac16_comparison.py")
        print(f"   Script to compare methods")
    
    def print_next_steps(self):
        """Print next steps after registration"""
        print(f"\n🚀 NEXT STEPS AFTER REGISTRATION:")
        print("=" * 40)
        print("1. 📝 Register at http://tupac.tue-image.nl/")
        print("2. 📥 Download TUPAC16 dataset files") 
        print("3. 📁 Extract to TUPAC16_Dataset/ directory")
        print("4. 🧪 Run: python tupac16_dataset.py")
        print("5. 📊 Run: python tupac16_comparison.py")
        print("6. 🔄 Retrain models with proper ground truth")
        print("7. 📖 Compare performance improvements")
        
        print(f"\n💡 MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 35)
        print("After registration, you'll typically get:")
        print("• Training images (.tif files)")
        print("• Test images (.tif files)")
        print("• Ki-67 scores (.csv files)")
        print("• Annotation files (.xml or .csv)")
        print("• Evaluation scripts")
        
        print(f"\n📁 ORGANIZE FILES LIKE THIS:")
        print("TUPAC16_Dataset/")
        print("├── images/")
        print("│   ├── training/")
        print("│   └── test/")
        print("├── annotations/")
        print("│   ├── ki67_scores.csv")
        print("│   └── mitosis/")
        print("└── metadata/")

def main():
    """Main function to guide TUPAC16 setup"""
    downloader = TUPAC16Downloader()
    
    # Print registration guide
    downloader.print_registration_guide()
    
    # Check if already downloaded
    if downloader.check_download_status():
        print("\n✅ TUPAC16 files detected!")
    else:
        print("\n📥 TUPAC16 files not found - registration needed")
    
    # Setup directory structure
    downloader.setup_directory_structure()
    
    # Create dataset classes
    downloader.create_tupac16_dataset_class()
    downloader.create_comparison_script()
    
    # Print next steps
    downloader.print_next_steps()
    
    print(f"\n🎉 SETUP COMPLETE!")
    print("=" * 30)
    print("✅ Registration guide provided")
    print("✅ Directory structure created")
    print("✅ Dataset classes generated")
    print("✅ Comparison scripts ready")
    
    print(f"\n🏆 TUPAC16 IS THE GOLD STANDARD!")
    print("Your 98% accuracy could be even higher with proper ground truth! 🚀")

if __name__ == "__main__":
    main()
