#!/usr/bin/env python3
"""
Improved Ki-67 Dataset with Proper Annotation-Based Classification

Uses actual annotation counts instead of file sizes for ground truth determination.
"""

import os
import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImprovedKi67Dataset(Dataset):
    """Dataset class using proper annotation-based classification"""
    
    def __init__(self, dataset_path, split='test', transform=None, classification_method='count_based'):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.transform = transform
        self.classification_method = classification_method
        
        self.image_paths = []
        self.labels = []
        self.annotation_counts = []  # Store actual counts for analysis
        
        self.create_dataset_from_annotations()
    
    def create_dataset_from_annotations(self):
        """Create dataset using proper annotation-based logic"""
        print(f"📁 Loading from dataset: {self.dataset_path}")
        print(f"🔬 Classification method: {self.classification_method}")
        
        # Find the correct dataset structure
        possible_paths = [
            self.dataset_path / "Ki67_Dataset_for_Colab",
            self.dataset_path / "BCData",
            self.dataset_path
        ]
        
        base_path = None
        for path in possible_paths:
            if path.exists() and (path / "images" / self.split).exists():
                base_path = path
                break
        
        if base_path is None:
            raise FileNotFoundError(f"Dataset not found in {self.dataset_path}")
        
        images_dir = base_path / "images" / self.split
        pos_annotations_dir = base_path / "annotations" / self.split / "positive"
        neg_annotations_dir = base_path / "annotations" / self.split / "negative"
        
        print(f"📁 Loading from: {images_dir}")
        
        # Process each image
        for img_file in images_dir.glob("*.png"):
            img_name = img_file.stem
            
            pos_ann = pos_annotations_dir / f"{img_name}.h5"
            neg_ann = neg_annotations_dir / f"{img_name}.h5"
            
            # Get label using proper method
            label, pos_count, neg_count = self._get_proper_label(pos_ann, neg_ann, img_name)
            
            if label is not None:
                self.image_paths.append(str(img_file))
                self.labels.append(label)
                self.annotation_counts.append({'positive': pos_count, 'negative': neg_count})
        
        pos_count = sum(self.labels)
        neg_count = len(self.labels) - pos_count
        print(f"✅ Found {len(self.image_paths)} images")
        print(f"   Distribution: {pos_count} positive, {neg_count} negative")
        
        # Analysis of annotation counts
        if self.annotation_counts:
            pos_annotations = [c['positive'] for c in self.annotation_counts]
            neg_annotations = [c['negative'] for c in self.annotation_counts]
            print(f"📊 Annotation statistics:")
            print(f"   Positive annotations: mean={np.mean(pos_annotations):.1f}, max={np.max(pos_annotations)}")
            print(f"   Negative annotations: mean={np.mean(neg_annotations):.1f}, max={np.max(neg_annotations)}")
    
    def _get_proper_label(self, pos_ann, neg_ann, img_name):
        """Get label using proper annotation-based methods"""
        pos_count = 0
        neg_count = 0
        
        # Count annotations in each file
        try:
            if pos_ann.exists():
                with h5py.File(pos_ann, 'r') as f:
                    if 'coordinates' in f:
                        pos_count = len(f['coordinates'])
        except:
            pass
        
        try:
            if neg_ann.exists():
                with h5py.File(neg_ann, 'r') as f:
                    if 'coordinates' in f:
                        neg_count = len(f['coordinates'])
        except:
            pass
        
        # Apply classification method
        if self.classification_method == 'count_based':
            # More positive annotations = positive class
            if pos_count > 0 or neg_count > 0:
                label = 1 if pos_count >= neg_count else 0
            else:
                label = None
                
        elif self.classification_method == 'threshold_based':
            # Ki-67 index threshold (e.g., >20 positive cells = high proliferation)
            threshold = 20
            label = 1 if pos_count >= threshold else 0
            
        elif self.classification_method == 'directory_based':
            # Use directory as intended (simplest)
            if pos_ann.exists() and not neg_ann.exists():
                label = 1
            elif neg_ann.exists() and not pos_ann.exists():
                label = 0
            elif pos_ann.exists() and neg_ann.exists():
                # Both exist - use count-based as fallback
                label = 1 if pos_count >= neg_count else 0
            else:
                label = None
                
        elif self.classification_method == 'file_size_legacy':
            # Original file size method (for comparison)
            if pos_ann.exists() and neg_ann.exists():
                try:
                    pos_size = pos_ann.stat().st_size
                    neg_size = neg_ann.stat().st_size
                    size_diff = abs(pos_size - neg_size)
                    
                    if size_diff > 100:
                        label = 1 if pos_size > neg_size else 0
                    else:
                        # Fallback to alternating (unreliable)
                        label = int(img_name) % 2
                except:
                    label = 1
            elif pos_ann.exists():
                label = 1
            elif neg_ann.exists():
                label = 0
            else:
                label = None
        else:
            raise ValueError(f"Unknown classification method: {self.classification_method}")
        
        return label, pos_count, neg_count
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)
    
    def get_annotation_statistics(self):
        """Get detailed statistics about annotations"""
        stats = {
            'total_images': len(self.image_paths),
            'positive_images': sum(self.labels),
            'negative_images': len(self.labels) - sum(self.labels),
            'annotation_counts': self.annotation_counts
        }
        
        if self.annotation_counts:
            pos_counts = [c['positive'] for c in self.annotation_counts]
            neg_counts = [c['negative'] for c in self.annotation_counts]
            
            stats['positive_annotations'] = {
                'mean': np.mean(pos_counts),
                'std': np.std(pos_counts),
                'min': np.min(pos_counts),
                'max': np.max(pos_counts),
                'total': np.sum(pos_counts)
            }
            
            stats['negative_annotations'] = {
                'mean': np.mean(neg_counts),
                'std': np.std(neg_counts),
                'min': np.min(neg_counts),
                'max': np.max(neg_counts),
                'total': np.sum(neg_counts)
            }
        
        return stats


def compare_classification_methods(dataset_path):
    """Compare different classification methods"""
    print("🔬 Comparing Classification Methods")
    print("=" * 50)
    
    methods = ['count_based', 'threshold_based', 'directory_based', 'file_size_legacy']
    results = {}
    
    for method in methods:
        print(f"\n📊 Testing method: {method}")
        try:
            dataset = ImprovedKi67Dataset(dataset_path, classification_method=method)
            results[method] = {
                'total_images': len(dataset),
                'positive_count': sum(dataset.labels),
                'negative_count': len(dataset.labels) - sum(dataset.labels),
                'success': True
            }
        except Exception as e:
            print(f"❌ Method {method} failed: {e}")
            results[method] = {'success': False, 'error': str(e)}
    
    # Summary comparison
    print(f"\n📊 CLASSIFICATION METHOD COMPARISON:")
    print("=" * 50)
    for method, result in results.items():
        if result['success']:
            pos_pct = result['positive_count'] / result['total_images'] * 100
            print(f"{method:20s}: {result['positive_count']:3d} pos, {result['negative_count']:3d} neg ({pos_pct:.1f}% positive)")
        else:
            print(f"{method:20s}: FAILED - {result['error']}")
    
    return results


if __name__ == "__main__":
    # Test the improved classification methods
    dataset_path = Path(".")
    
    # Compare all methods
    results = compare_classification_methods(dataset_path)
    
    # Detailed analysis of the best method
    print(f"\n🔬 DETAILED ANALYSIS: Count-Based Method")
    print("=" * 50)
    
    dataset = ImprovedKi67Dataset(dataset_path, classification_method='count_based')
    stats = dataset.get_annotation_statistics()
    
    print(f"📊 Dataset Statistics:")
    print(f"   Total images: {stats['total_images']}")
    print(f"   Positive images: {stats['positive_images']}")
    print(f"   Negative images: {stats['negative_images']}")
    
    if 'positive_annotations' in stats:
        print(f"\n📊 Positive Annotations:")
        print(f"   Mean per image: {stats['positive_annotations']['mean']:.1f}")
        print(f"   Std deviation: {stats['positive_annotations']['std']:.1f}")
        print(f"   Range: {stats['positive_annotations']['min']}-{stats['positive_annotations']['max']}")
        print(f"   Total: {stats['positive_annotations']['total']}")
        
        print(f"\n📊 Negative Annotations:")
        print(f"   Mean per image: {stats['negative_annotations']['mean']:.1f}")
        print(f"   Std deviation: {stats['negative_annotations']['std']:.1f}")
        print(f"   Range: {stats['negative_annotations']['min']}-{stats['negative_annotations']['max']}")
        print(f"   Total: {stats['negative_annotations']['total']}")
