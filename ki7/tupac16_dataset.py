#!/usr/bin/env python3
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
        
        print(f"\nDataset statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test loading a sample
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample data:")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Label: {sample['label'].item()}")
            print(f"  Ki-67 score: {sample['ki67_score'].item():.1f}%")
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please download TUPAC16 dataset first!")
