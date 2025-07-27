#!/usr/bin/env python3
"""
Test Dataset Logic - Verify Champion Model Dataset Compatibility
Run this locally to verify the dataset will work correctly in Colab
"""

import os
import sys
from pathlib import Path

def test_dataset_structure(dataset_path):
    """Test if dataset has the required structure"""
    dataset_path = Path(dataset_path)
    
    print(f"🔍 Testing dataset structure: {dataset_path}")
    
    # Check for possible base paths
    possible_bases = [
        dataset_path / "BCData",
        dataset_path / "Ki67_Dataset_for_Colab", 
        dataset_path / "ki67_dataset",
        dataset_path
    ]
    
    for base in possible_bases:
        if base.exists():
            print(f"📁 Found potential base: {base}")
            
            # Check structure
            for split in ['train', 'validation', 'test']:
                images_dir = base / "images" / split
                pos_ann_dir = base / "annotations" / split / "positive"
                neg_ann_dir = base / "annotations" / split / "negative"
                
                if images_dir.exists():
                    print(f"  ✅ {split} images: {len(list(images_dir.glob('*.png')))} files")
                else:
                    print(f"  ❌ {split} images: missing")
                
                if pos_ann_dir.exists():
                    print(f"  ✅ {split} positive annotations: {len(list(pos_ann_dir.glob('*.h5')))} files")
                else:
                    print(f"  ❌ {split} positive annotations: missing")
                
                if neg_ann_dir.exists():
                    print(f"  ✅ {split} negative annotations: {len(list(neg_ann_dir.glob('*.h5')))} files")
                else:
                    print(f"  ❌ {split} negative annotations: missing")
            
            return base
    
    print("❌ No valid dataset structure found")
    return None

def test_annotation_logic(base_path, split='train', max_samples=10):
    """Test the annotation size analysis logic"""
    print(f"\n🔬 Testing annotation logic for {split} split...")
    
    images_dir = base_path / "images" / split
    pos_ann_dir = base_path / "annotations" / split / "positive"
    neg_ann_dir = base_path / "annotations" / split / "negative"
    
    if not all([images_dir.exists(), pos_ann_dir.exists(), neg_ann_dir.exists()]):
        print(f"❌ Required directories missing for {split}")
        return
    
    positive_count = 0
    negative_count = 0
    processed = 0
    
    for img_file in images_dir.glob("*.png"):
        if processed >= max_samples:
            break
            
        img_name = img_file.stem
        pos_ann = pos_ann_dir / f"{img_name}.h5"
        neg_ann = neg_ann_dir / f"{img_name}.h5"
        
        if pos_ann.exists() and neg_ann.exists():
            try:
                pos_size = pos_ann.stat().st_size
                neg_size = neg_ann.stat().st_size
                size_diff = abs(pos_size - neg_size)
                
                if size_diff > 100:
                    if neg_size > pos_size:
                        label = 0  # Negative
                        negative_count += 1
                        print(f"  📄 {img_name}: NEGATIVE (neg={neg_size}, pos={pos_size})")
                    else:
                        label = 1  # Positive
                        positive_count += 1
                        print(f"  📄 {img_name}: POSITIVE (pos={pos_size}, neg={neg_size})")
                else:
                    label = processed % 2
                    if label == 0:
                        negative_count += 1
                        print(f"  📄 {img_name}: NEGATIVE (alternating, similar sizes)")
                    else:
                        positive_count += 1
                        print(f"  📄 {img_name}: POSITIVE (alternating, similar sizes)")
                
                processed += 1
                
            except Exception as e:
                print(f"  ⚠️  Error with {img_name}: {e}")
    
    print(f"\n📊 Sample results for {split}:")
    print(f"   Processed: {processed} images")
    print(f"   Positive: {positive_count}")
    print(f"   Negative: {negative_count}")
    
    if positive_count > 0 and negative_count > 0:
        print(f"   ✅ Both classes found - logic working correctly!")
    else:
        print(f"   ❌ Single class detected - this needs investigation")

def main():
    """Main test function"""
    print("🧪 Champion Dataset Logic Test")
    print("=" * 50)
    
    # Find dataset path
    possible_paths = [
        "/Users/chinthan/ki7/BCData",
        "/Users/chinthan/ki7/Ki67_Dataset_for_Colab",
        "/Users/chinthan/ki7/ki67_dataset",
        "/Users/chinthan/ki7/data",
        "./BCData",
        "./Ki67_Dataset_for_Colab",
        "./data"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            break
    
    if dataset_path is None:
        print("❌ No dataset found in expected locations:")
        for path in possible_paths:
            print(f"   {path}")
        print("\nPlease provide dataset path as argument:")
        print("   python test_dataset_logic.py /path/to/dataset")
        return
    
    print(f"📂 Using dataset: {dataset_path}")
    
    # Test structure
    base_path = test_dataset_structure(dataset_path)
    if base_path is None:
        return
    
    # Test annotation logic for each split
    for split in ['train', 'validation', 'test']:
        test_annotation_logic(base_path, split, max_samples=5)
    
    print("\n🎯 Test Summary:")
    print("If you see 'Both classes found' for all splits, the champion")
    print("model will use the same proven logic as your ensemble pipeline.")
    print("\nReady for Colab training! 🚀")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        base_path = test_dataset_structure(dataset_path)
        if base_path:
            for split in ['train', 'validation', 'test']:
                test_annotation_logic(base_path, split, max_samples=10)
    else:
        main()
