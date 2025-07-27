#!/usr/bin/env python3
"""
Quick Dataset Setup: BreakHis Download and Preparation

This script helps you quickly download and set up the BreakHis dataset
as a superior alternative to the file-size based classification approach.
"""

import os
import requests
import tarfile
import pandas as pd
from pathlib import Path
import shutil
from urllib.parse import urlparse

def download_breakhis():
    """Download the BreakHis dataset"""
    print("🔄 Downloading BreakHis Dataset...")
    print("=" * 50)
    
    url = "https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
    filename = "BreaKHis_v1.tar.gz"
    
    if Path(filename).exists():
        print(f"✅ {filename} already exists")
        return filename
    
    print(f"📥 Downloading from: {url}")
    print("⏳ This may take several minutes...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\r📥 Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Downloaded {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Try downloading manually from:")
        print("   https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/")
        return None

def extract_breakhis(filename):
    """Extract the BreakHis dataset"""
    print(f"\n📦 Extracting {filename}...")
    
    if Path("BreaKHis_v1").exists():
        print("✅ BreaKHis_v1 already extracted")
        return "BreaKHis_v1"
    
    try:
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall()
        print("✅ Extraction complete")
        return "BreaKHis_v1"
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None

def analyze_breakhis_structure(dataset_dir):
    """Analyze the BreakHis dataset structure"""
    print(f"\n🔬 Analyzing BreakHis Dataset Structure...")
    print("=" * 50)
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return
    
    # Find histology slides directory
    histology_dir = dataset_path / "histology_slides" / "breast"
    
    if not histology_dir.exists():
        print(f"❌ Histology directory not found: {histology_dir}")
        return
    
    # Count images by category
    stats = {}
    total_images = 0
    
    for category in ["benign", "malignant"]:
        category_dir = histology_dir / category
        if category_dir.exists():
            # Count by magnification
            magnifications = {}
            for mag_dir in category_dir.iterdir():
                if mag_dir.is_dir():
                    mag_name = mag_dir.name
                    count = 0
                    for class_dir in mag_dir.iterdir():
                        if class_dir.is_dir():
                            count += len(list(class_dir.glob("*.png")))
                    magnifications[mag_name] = count
                    total_images += count
            stats[category] = magnifications
    
    # Print analysis
    print(f"📊 Dataset Statistics:")
    print(f"   Total Images: {total_images:,}")
    
    for category, magnifications in stats.items():
        category_total = sum(magnifications.values())
        print(f"\n📂 {category.capitalize()}: {category_total:,} images")
        for mag, count in magnifications.items():
            print(f"   {mag}: {count:,} images")
    
    print(f"\n📝 Dataset Structure:")
    print(f"BreaKHis_v1/")
    print(f"├── histology_slides/")
    print(f"│   └── breast/")
    print(f"│       ├── benign/")
    print(f"│       │   ├── 40X/")
    print(f"│       │   ├── 100X/")
    print(f"│       │   ├── 200X/")
    print(f"│       │   └── 400X/")
    print(f"│       └── malignant/")
    print(f"│           ├── 40X/")
    print(f"│           ├── 100X/")
    print(f"│           ├── 200X/")
    print(f"│           └── 400X/")
    print(f"└── Folds.csv")
    
    return stats

def create_breakhis_dataset_class():
    """Create a proper dataset class for BreakHis"""
    
    dataset_code = '''#!/usr/bin/env python3
"""
BreakHis Dataset Class - Proper Alternative to File Size Classification
"""

import os
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BreakHisDataset(Dataset):
    """
    BreakHis Breast Cancer Dataset
    
    A proper, well-annotated alternative to file-size based classification.
    """
    
    def __init__(self, root_dir, magnification='40X', transform=None, split='train'):
        self.root_dir = Path(root_dir)
        self.magnification = magnification
        self.transform = transform
        self.split = split
        
        self.image_paths = []
        self.labels = []
        self.classes = ['benign', 'malignant']
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset with proper annotations"""
        histology_dir = self.root_dir / "histology_slides" / "breast"
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = histology_dir / class_name / self.magnification
            
            if not class_dir.exists():
                continue
            
            # Process each subclass directory
            for subclass_dir in class_dir.iterdir():
                if subclass_dir.is_dir():
                    for img_file in subclass_dir.glob("*.png"):
                        self.image_paths.append(str(img_file))
                        self.labels.append(class_idx)
        
        print(f"✅ Loaded {len(self.image_paths)} images at {self.magnification}")
        print(f"   Benign: {self.labels.count(0)}")
        print(f"   Malignant: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)

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
    dataset = BreakHisDataset(
        root_dir="BreaKHis_v1",
        magnification="200X",
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test loading a sample
    sample_image, sample_label = dataset[0]
    print(f"Sample shape: {sample_image.shape}")
    print(f"Sample label: {dataset.classes[sample_label]}")
'''
    
    with open("breakhis_dataset.py", "w") as f:
        f.write(dataset_code)
    
    print(f"📝 Created breakhis_dataset.py")
    print(f"   Proper dataset class with clear annotations")

def main():
    """Main setup function"""
    print("🚀 BREAKHIS DATASET SETUP")
    print("=" * 50)
    print("Setting up a superior alternative to file-size classification")
    print()
    
    # Download dataset
    filename = download_breakhis()
    if not filename:
        return
    
    # Extract dataset
    dataset_dir = extract_breakhis(filename)
    if not dataset_dir:
        return
    
    # Analyze structure
    stats = analyze_breakhis_structure(dataset_dir)
    
    # Create dataset class
    create_breakhis_dataset_class()
    
    print(f"\n🎉 SETUP COMPLETE!")
    print("=" * 50)
    print("✅ BreakHis dataset downloaded and extracted")
    print("✅ Dataset structure analyzed")
    print("✅ Python dataset class created")
    
    print(f"\n🔄 NEXT STEPS:")
    print("1. Test the dataset: python breakhis_dataset.py")
    print("2. Compare with your current approach")
    print("3. Retrain models using proper annotations")
    print("4. Evaluate performance improvement")
    
    print(f"\n📊 ADVANTAGES OVER FILE-SIZE METHOD:")
    print("• Clear benign vs malignant labels")
    print("• Expert pathologist annotations")
    print("• Multiple magnification levels")
    print("• Established benchmark dataset")
    print("• No arbitrary file size thresholds")
    print("• Scientifically sound classification")

if __name__ == "__main__":
    main()
'''
