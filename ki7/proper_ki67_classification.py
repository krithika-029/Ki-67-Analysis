#!/usr/bin/env python3
"""
Corrected Ki-67 Classification Logic

Based on analysis showing that the file size method was accidentally working
due to inverted annotation meanings. This provides the correct approach.
"""

import os
import h5py
import numpy as np
from pathlib import Path

def analyze_sample_annotations():
    """Analyze a few sample annotations to understand the data structure"""
    print("🔬 SAMPLE ANNOTATION ANALYSIS")
    print("=" * 50)
    
    base_path = Path("Ki67_Dataset_for_Colab/annotations/test")
    
    # Check a few samples
    for i in [1, 10, 50, 100]:
        pos_file = base_path / "positive" / f"{i}.h5"
        neg_file = base_path / "negative" / f"{i}.h5"
        
        print(f"\n📊 Image {i}:")
        
        pos_count = 0
        neg_count = 0
        
        if pos_file.exists():
            try:
                with h5py.File(pos_file, 'r') as f:
                    pos_count = len(f['coordinates']) if 'coordinates' in f else 0
                print(f"   Positive annotations: {pos_count}")
            except:
                print(f"   Positive annotations: ERROR")
        
        if neg_file.exists():
            try:
                with h5py.File(neg_file, 'r') as f:
                    neg_count = len(f['coordinates']) if 'coordinates' in f else 0
                print(f"   Negative annotations: {neg_count}")
            except:
                print(f"   Negative annotations: ERROR")
        
        # File sizes for comparison
        if pos_file.exists() and neg_file.exists():
            pos_size = pos_file.stat().st_size
            neg_size = neg_file.stat().st_size
            print(f"   File sizes: pos={pos_size}, neg={neg_size}")
            print(f"   Size-based label: {'Positive' if pos_size > neg_size else 'Negative'}")
            print(f"   Count-based label: {'Positive' if pos_count >= neg_count else 'Negative'}")

def get_proper_ki67_label(image_id, method='corrected_biological'):
    """
    Get the proper Ki-67 label using biological reasoning
    
    Methods:
    - 'corrected_biological': Use positive annotations as Ki-67+ cells, classify based on proliferation index
    - 'count_ratio': Use ratio of positive to total annotations
    - 'absolute_threshold': Use absolute count threshold
    """
    
    base_path = Path("Ki67_Dataset_for_Colab/annotations/test")
    pos_file = base_path / "positive" / f"{image_id}.h5"
    neg_file = base_path / "negative" / f"{image_id}.h5"
    
    pos_count = 0
    neg_count = 0
    
    # Get annotation counts
    if pos_file.exists():
        try:
            with h5py.File(pos_file, 'r') as f:
                pos_count = len(f['coordinates']) if 'coordinates' in f else 0
        except:
            pass
    
    if neg_file.exists():
        try:
            with h5py.File(neg_file, 'r') as f:
                neg_count = len(f['coordinates']) if 'coordinates' in f else 0
        except:
            pass
    
    if method == 'corrected_biological':
        # Ki-67 proliferation index = (Ki-67+ cells / total cells) * 100
        # Typically >20% is considered high proliferation
        total_cells = pos_count + neg_count
        if total_cells == 0:
            return None
        
        ki67_index = (pos_count / total_cells) * 100
        return 1 if ki67_index >= 20 else 0  # High vs Low proliferation
        
    elif method == 'count_ratio':
        # Simple ratio comparison
        total_count = pos_count + neg_count
        if total_count == 0:
            return None
        return 1 if pos_count / total_count >= 0.3 else 0
        
    elif method == 'absolute_threshold':
        # Absolute count threshold
        return 1 if pos_count >= 20 else 0
    
    else:
        raise ValueError(f"Unknown method: {method}")

def validate_classification_approaches():
    """Validate different classification approaches"""
    print("🧪 VALIDATION OF CLASSIFICATION APPROACHES")
    print("=" * 60)
    
    # Sample a few images to test different methods
    test_images = [1, 10, 50, 100, 200]
    
    print(f"{'Image':<6} {'File Size':<10} {'Count':<8} {'Bio':<6} {'Ratio':<8} {'Thresh':<8}")
    print("-" * 60)
    
    for img_id in test_images:
        # File size method (original)
        base_path = Path("Ki67_Dataset_for_Colab/annotations/test")
        pos_file = base_path / "positive" / f"{img_id}.h5"
        neg_file = base_path / "negative" / f"{img_id}.h5"
        
        file_size_label = "N/A"
        if pos_file.exists() and neg_file.exists():
            pos_size = pos_file.stat().st_size
            neg_size = neg_file.stat().st_size
            if abs(pos_size - neg_size) > 100:
                file_size_label = "Pos" if pos_size > neg_size else "Neg"
            else:
                file_size_label = "Alt"
        
        # Other methods
        count_label = "Pos" if get_proper_ki67_label(img_id, 'count_ratio') == 1 else "Neg"
        bio_label = "Pos" if get_proper_ki67_label(img_id, 'corrected_biological') == 1 else "Neg"
        ratio_label = "Pos" if get_proper_ki67_label(img_id, 'count_ratio') == 1 else "Neg"
        thresh_label = "Pos" if get_proper_ki67_label(img_id, 'absolute_threshold') == 1 else "Neg"
        
        print(f"{img_id:<6} {file_size_label:<10} {count_label:<8} {bio_label:<6} {ratio_label:<8} {thresh_label:<8}")

if __name__ == "__main__":
    # Run analysis
    analyze_sample_annotations()
    print()
    validate_classification_approaches()
    
    print(f"\n✅ RECOMMENDATIONS:")
    print("=" * 50)
    print("1. 🔬 Use BIOLOGICAL METHOD: Ki-67 proliferation index")
    print("   - Positive annotations = Ki-67+ cells")
    print("   - Ki-67 index = (Ki-67+ / total) * 100")
    print("   - Threshold: >20% = High proliferation")
    print()
    print("2. 🚫 AVOID file size method - it's unreliable")
    print("3. 📊 Validate with pathologist ground truth if available")
    print("4. 🔄 Consider using directory structure if it's correctly labeled")
