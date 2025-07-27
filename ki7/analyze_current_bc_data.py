#!/usr/bin/env python3
"""
BC Dataset Viability Checker

Analyzes whether your existing Ki67_Dataset_for_Colab can be used effectively
for Ki-67 classification without switching to TUPAC16.
"""

import os
import h5py
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def analyze_current_dataset():
    """Quick analysis of current BC dataset quality"""
    print("🔍 ANALYZING YOUR CURRENT BC DATASET")
    print("=" * 50)
    print("Checking if Ki67_Dataset_for_Colab can be used instead of TUPAC16...")
    print()
    
    dataset_path = Path("Ki67_Dataset_for_Colab")
    
    # Check structure
    print("📁 Dataset Structure Check:")
    required_paths = [
        dataset_path / "images" / "test",
        dataset_path / "annotations" / "test" / "positive",
        dataset_path / "annotations" / "test" / "negative"
    ]
    
    structure_ok = True
    for path in required_paths:
        if path.exists():
            file_count = len(list(path.glob("*")))
            print(f"   ✅ {path.name}: {file_count} files")
        else:
            print(f"   ❌ Missing: {path}")
            structure_ok = False
    
    if not structure_ok:
        print("\n❌ VERDICT: Dataset structure invalid - cannot use")
        return False
    
    # Quick annotation analysis
    print(f"\n🔬 Quick Annotation Analysis:")
    
    pos_dir = dataset_path / "annotations" / "test" / "positive"
    neg_dir = dataset_path / "annotations" / "test" / "negative"
    
    # Sample 10 files for quick analysis
    sample_data = []
    for i in [1, 10, 50, 100, 200]:
        pos_file = pos_dir / f"{i}.h5"
        neg_file = neg_dir / f"{i}.h5"
        
        if pos_file.exists() and neg_file.exists():
            try:
                with h5py.File(pos_file, 'r') as f:
                    pos_count = len(f['coordinates']) if 'coordinates' in f else 0
                
                with h5py.File(neg_file, 'r') as f:
                    neg_count = len(f['coordinates']) if 'coordinates' in f else 0
                
                ki67_index = (pos_count / (pos_count + neg_count) * 100) if (pos_count + neg_count) > 0 else 0
                
                sample_data.append({
                    'id': i,
                    'pos_count': pos_count,
                    'neg_count': neg_count,
                    'ki67_index': ki67_index
                })
                
                print(f"   Image {i}: {pos_count} pos, {neg_count} neg → {ki67_index:.1f}% Ki-67 index")
                
            except Exception as e:
                print(f"   ⚠️  Image {i}: Error reading - {e}")
    
    if not sample_data:
        print("   ❌ No valid annotations found")
        return False
    
    # Calculate statistics
    pos_counts = [d['pos_count'] for d in sample_data]
    neg_counts = [d['neg_count'] for d in sample_data]
    ki67_indices = [d['ki67_index'] for d in sample_data]
    
    print(f"\n📊 Sample Statistics:")
    print(f"   Positive cells: {np.mean(pos_counts):.1f} ± {np.std(pos_counts):.1f} (range: {min(pos_counts)}-{max(pos_counts)})")
    print(f"   Negative cells: {np.mean(neg_counts):.1f} ± {np.std(neg_counts):.1f} (range: {min(neg_counts)}-{max(neg_counts)})")
    print(f"   Ki-67 index: {np.mean(ki67_indices):.1f}% ± {np.std(ki67_indices):.1f}% (range: {min(ki67_indices):.1f}-{max(ki67_indices):.1f}%)")
    
    # Quality assessment
    quality_score = 10.0
    issues = []
    
    if np.mean(pos_counts) < 5:
        quality_score -= 3
        issues.append("Very low positive cell counts")
    
    if np.mean(neg_counts) < 10:
        quality_score -= 2
        issues.append("Low negative cell counts")
    
    ki67_range = max(ki67_indices) - min(ki67_indices)
    if ki67_range < 20:
        quality_score -= 2
        issues.append("Limited Ki-67 index range")
    
    if max(ki67_indices) > 80:
        quality_score -= 1
        issues.append("Unusually high Ki-67 indices")
    
    print(f"\n🎯 Quality Assessment: {quality_score:.1f}/10")
    
    if issues:
        print("   ⚠️  Issues detected:")
        for issue in issues:
            print(f"      • {issue}")
    else:
        print("   ✅ No major issues detected")
    
    # Final verdict
    print(f"\n🏆 FINAL VERDICT:")
    print("=" * 30)
    
    if quality_score >= 7:
        print("✅ EXCELLENT - Use your current dataset!")
        print("   Your BC dataset is high quality")
        print("   Switch from file-size to annotation-based classification")
        print("   Your 98% accuracy will likely improve")
        verdict = True
    elif quality_score >= 5:
        print("🟡 GOOD - Can use with improvements")
        print("   Your BC dataset is usable")
        print("   Consider validation with manual review")
        print("   May want to supplement with TUPAC16 later")
        verdict = True
    else:
        print("❌ POOR - Consider switching to TUPAC16/BreakHis")
        print("   Current dataset has quality issues")
        print("   Recommend downloading better dataset")
        verdict = False
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if verdict:
        print("1. 🔄 Switch to annotation-based classification:")
        print("   from improved_ki67_dataset import ImprovedKi67Dataset")
        print("   dataset = ImprovedKi67Dataset('.', classification_method='count_based')")
        print()
        print("2. 🧪 Test the improved method:")
        print("   python improved_ki67_dataset.py")
        print()
        print("3. 🔬 Update your ensemble script:")
        print("   Replace RefinedKi67Dataset with ImprovedKi67Dataset")
        print()
        print("4. 📊 Compare performance:")
        print("   Retrain and see if accuracy improves beyond 98%")
    else:
        print("1. 📥 Download BreakHis (immediate): python quick_dataset_setup.py")
        print("2. 📧 Register for TUPAC16: http://tupac.tue-image.nl/")
        print("3. 🔍 Manual validation of current annotations")
    
    return verdict

def compare_with_file_size_method():
    """Compare annotation-based vs file-size classification"""
    print(f"\n🔄 COMPARING CLASSIFICATION METHODS")
    print("=" * 50)
    
    try:
        from improved_ki67_dataset import ImprovedKi67Dataset
        
        print("Testing different classification methods on your data...")
        
        methods = {
            'count_based': 'Annotation count comparison',
            'threshold_based': 'Ki-67 proliferation index (>20%)',
            'file_size_legacy': 'Your current file size method'
        }
        
        for method_name, description in methods.items():
            try:
                dataset = ImprovedKi67Dataset(".", classification_method=method_name)
                pos_count = sum(dataset.labels)
                total = len(dataset.labels)
                pos_pct = pos_count / total * 100
                
                print(f"✅ {method_name}: {pos_count}/{total} ({pos_pct:.1f}% positive)")
                print(f"   └─ {description}")
                
            except Exception as e:
                print(f"❌ {method_name}: Failed - {e}")
        
        print(f"\n💡 INSIGHT:")
        print("If methods give similar results, your current file-size approach")
        print("accidentally works because file size correlates with annotation count!")
        print("But annotation-based is more reliable and scientifically sound.")
        
    except ImportError:
        print("❌ Cannot import ImprovedKi67Dataset")
        print("   Run this first: python improved_ki67_dataset.py")

def main():
    """Main analysis function"""
    print("🚀 BC DATASET VIABILITY CHECK")
    print("=" * 40)
    print("Checking if you can use your current dataset instead of TUPAC16")
    print()
    
    # Quick analysis
    can_use_current = analyze_current_dataset()
    
    # Method comparison
    compare_with_file_size_method()
    
    print(f"\n🎯 SUMMARY:")
    print("=" * 20)
    
    if can_use_current:
        print("✅ You can continue with your current BC dataset!")
        print("✅ Just switch to proper annotation-based classification")
        print("✅ No need to wait for TUPAC16 registration")
        print()
        print("🚀 Quick start:")
        print("   python improved_ki67_dataset.py")
    else:
        print("⚠️  Your current dataset has quality issues")
        print("📥 Recommended: Download BreakHis or register for TUPAC16")
        print()
        print("🚀 Quick alternative:")
        print("   python quick_dataset_setup.py  # BreakHis download")

if __name__ == "__main__":
    main()
