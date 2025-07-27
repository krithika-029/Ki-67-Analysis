#!/usr/bin/env python3
"""
Recommended Ki-67 and Histopathology Datasets

A comprehensive list of high-quality, properly annotated datasets for Ki-67 
proliferation analysis and breast cancer histopathology.
"""

import requests
import json
from pathlib import Path

class DatasetRecommendations:
    """Collection of recommended datasets for Ki-67 analysis"""
    
    def __init__(self):
        self.datasets = {
            "ki67_specific": [
                {
                    "name": "TUPAC16 - Tumor Proliferation Assessment Challenge",
                    "description": "Gold standard Ki-67 dataset with expert annotations",
                    "url": "http://tupac.tue-image.nl/",
                    "size": "500 breast cancer images",
                    "annotation_type": "Ki-67 proliferation index ground truth",
                    "advantages": [
                        "Expert pathologist annotations",
                        "Standardized Ki-67 scoring protocol", 
                        "Benchmark dataset used in competitions",
                        "Multiple annotation levels (mitotic figures, Ki-67 index)"
                    ],
                    "format": "H&E and Ki-67 IHC stained images",
                    "availability": "Free for research",
                    "paper": "Veta et al., 2019 - Assessment of algorithms for mitosis detection",
                    "recommended": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "MIDOG Challenge Dataset",
                    "description": "Mitosis detection in histopathology images",
                    "url": "https://midog.deepmicroscopy.org/",
                    "size": "405 HPF images from 200 cases",
                    "annotation_type": "Mitotic figure annotations",
                    "advantages": [
                        "Multiple scanners and domains",
                        "Expert pathologist validation",
                        "Cross-domain generalization focus"
                    ],
                    "format": "Digital whole slide images",
                    "availability": "Free registration required",
                    "recommended": "⭐⭐⭐⭐"
                }
            ],
            "breast_cancer_general": [
                {
                    "name": "BreakHis - Breast Cancer Histopathological Database",
                    "description": "7,909 microscopy images of breast tumor tissue",
                    "url": "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/",
                    "size": "7,909 images (2,480 benign, 5,429 malignant)",
                    "annotation_type": "Benign vs malignant classification",
                    "advantages": [
                        "Large dataset size",
                        "Multiple magnifications (40X, 100X, 200X, 400X)",
                        "Well-established benchmark",
                        "Clear classification labels"
                    ],
                    "format": "PNG images at different magnifications",
                    "availability": "Free for research",
                    "paper": "Spanhol et al., 2016",
                    "recommended": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "BACH - Grand Challenge on Breast Cancer Histology",
                    "description": "400 H&E stained breast histology images",
                    "url": "https://iciar2018-challenge.grand-challenge.org/",
                    "size": "400 images (100 per class)",
                    "annotation_type": "4 classes: Normal, Benign, In situ carcinoma, Invasive carcinoma",
                    "advantages": [
                        "Expert pathologist annotations",
                        "Challenge dataset with benchmarks",
                        "High-resolution images",
                        "Balanced classes"
                    ],
                    "format": "TIFF images 2048x1536 pixels",
                    "availability": "Free registration required",
                    "recommended": "⭐⭐⭐⭐"
                },
                {
                    "name": "PCam - PatchCamelyon",
                    "description": "327,680 color images extracted from histopathologic scans",
                    "url": "https://github.com/basveeling/pcam",
                    "size": "327,680 patches (96x96 pixels)",
                    "annotation_type": "Binary classification (tumor vs normal)",
                    "advantages": [
                        "Very large dataset",
                        "Derived from CAMELYON challenge",
                        "Easy to use format",
                        "Good for deep learning"
                    ],
                    "format": "HDF5 files",
                    "availability": "Public domain",
                    "recommended": "⭐⭐⭐⭐"
                }
            ],
            "cell_detection": [
                {
                    "name": "MoNuSeg - Multi-Organ Nucleus Segmentation",
                    "description": "30 tissue images with nucleus boundaries",
                    "url": "https://monuseg.grand-challenge.org/",
                    "size": "30 images with 21,623 annotated nuclei",
                    "annotation_type": "Pixel-level nucleus segmentation",
                    "advantages": [
                        "Precise nucleus boundaries",
                        "Multiple organ types",
                        "Challenge validation"
                    ],
                    "format": "TIFF images with XML annotations",
                    "availability": "Free registration required",
                    "recommended": "⭐⭐⭐"
                },
                {
                    "name": "CoNSeP - Colorectal Nuclear Segmentation and Phenotypes",
                    "description": "41 H&E stained colorectal adenocarcinoma images",
                    "url": "https://warwick.ac.uk/fac/sci/dcs/research/tia/data/hovernet/",
                    "size": "41 images with 24,319 annotated nuclei",
                    "annotation_type": "Nuclear segmentation + classification (6 types)",
                    "advantages": [
                        "Nuclear type classification",
                        "High-quality annotations",
                        "Benchmark for cell detection"
                    ],
                    "format": "TIFF + MAT files",
                    "availability": "Free for research",
                    "recommended": "⭐⭐⭐⭐"
                }
            ],
            "public_repositories": [
                {
                    "name": "The Cancer Imaging Archive (TCIA)",
                    "description": "Large collection of medical images including histopathology",
                    "url": "https://www.cancerimagingarchive.net/",
                    "advantages": [
                        "Massive collection",
                        "High-quality clinical data",
                        "Multiple cancer types",
                        "DICOM format support"
                    ],
                    "recommended": "⭐⭐⭐⭐⭐"
                },
                {
                    "name": "Grand Challenge",
                    "description": "Platform hosting medical imaging challenges",
                    "url": "https://grand-challenge.org/",
                    "advantages": [
                        "Multiple histopathology challenges",
                        "Standardized evaluation metrics",
                        "Recent datasets available"
                    ],
                    "recommended": "⭐⭐⭐⭐"
                },
                {
                    "name": "Kaggle Medical Datasets",
                    "description": "Various histopathology datasets",
                    "url": "https://www.kaggle.com/datasets?search=histopathology",
                    "advantages": [
                        "Easy access and download",
                        "Community discussions",
                        "Kernels with examples"
                    ],
                    "recommended": "⭐⭐⭐"
                }
            ]
        }
    
    def print_recommendations(self):
        """Print formatted dataset recommendations"""
        print("🔬 RECOMMENDED Ki-67 & HISTOPATHOLOGY DATASETS")
        print("=" * 60)
        
        for category, datasets in self.datasets.items():
            print(f"\n📊 {category.upper().replace('_', ' ')}")
            print("-" * 50)
            
            for dataset in datasets:
                print(f"\n{dataset['recommended']} {dataset['name']}")
                print(f"   📝 {dataset['description']}")
                print(f"   🔗 {dataset['url']}")
                
                if 'size' in dataset:
                    print(f"   📏 Size: {dataset['size']}")
                if 'annotation_type' in dataset:
                    print(f"   🏷️  Annotations: {dataset['annotation_type']}")
                if 'format' in dataset:
                    print(f"   💾 Format: {dataset['format']}")
                if 'availability' in dataset:
                    print(f"   🌐 Access: {dataset['availability']}")
                
                if 'advantages' in dataset:
                    print(f"   ✅ Advantages:")
                    for advantage in dataset['advantages']:
                        print(f"      • {advantage}")
    
    def get_top_recommendations(self):
        """Get top 3 recommended datasets"""
        top_picks = [
            {
                "rank": 1,
                "name": "TUPAC16",
                "reason": "Gold standard for Ki-67 with expert annotations",
                "best_for": "Ki-67 proliferation index prediction",
                "download_cmd": "# Register at http://tupac.tue-image.nl/ and download"
            },
            {
                "rank": 2, 
                "name": "BreakHis",
                "reason": "Large, well-annotated breast cancer dataset",
                "best_for": "General breast cancer classification",
                "download_cmd": "wget https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
            },
            {
                "rank": 3,
                "name": "BACH Challenge",
                "reason": "High-quality breast histology with expert validation",
                "best_for": "Multi-class breast tissue classification", 
                "download_cmd": "# Register at https://iciar2018-challenge.grand-challenge.org/"
            }
        ]
        
        print(f"\n🏆 TOP 3 RECOMMENDATIONS FOR Ki-67 RESEARCH:")
        print("=" * 60)
        
        for pick in top_picks:
            print(f"\n#{pick['rank']} - {pick['name']}")
            print(f"   🎯 Reason: {pick['reason']}")
            print(f"   📚 Best for: {pick['best_for']}")
            print(f"   💻 Download: {pick['download_cmd']}")
    
    def create_download_script(self):
        """Create a script to help download datasets"""
        script_content = '''#!/bin/bash
# Dataset Download Helper Script
# Run this script to get guidance on downloading recommended datasets

echo "🔬 Ki-67 Dataset Download Guide"
echo "================================"

echo ""
echo "🏆 TOP RECOMMENDATIONS:"
echo ""

echo "1. TUPAC16 (Best for Ki-67)"
echo "   • Register at: http://tupac.tue-image.nl/"
echo "   • Download training and test sets"
echo "   • Includes expert Ki-67 annotations"
echo ""

echo "2. BreakHis (Large breast cancer dataset)"
echo "   • Direct download available"
echo "   • Command: wget https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
echo "   • Extract: tar -xzf BreaKHis_v1.tar.gz"
echo ""

echo "3. BACH Challenge (High-quality annotations)"
echo "   • Register at: https://iciar2018-challenge.grand-challenge.org/"
echo "   • Download training and test data"
echo "   • 4-class breast tissue classification"
echo ""

echo "📝 For more datasets, visit:"
echo "   • Grand Challenge: https://grand-challenge.org/"
echo "   • TCIA: https://www.cancerimagingarchive.net/"
echo "   • Kaggle: https://www.kaggle.com/datasets?search=histopathology"
'''
        
        with open("download_datasets.sh", "w") as f:
            f.write(script_content)
        
        print(f"📝 Created download_datasets.sh script")
        print(f"   Run: chmod +x download_datasets.sh && ./download_datasets.sh")


def main():
    recommender = DatasetRecommendations()
    
    # Print all recommendations
    recommender.print_recommendations()
    
    # Show top picks
    recommender.get_top_recommendations()
    
    # Create download helper
    recommender.create_download_script()
    
    print(f"\n💡 MIGRATION STRATEGY FROM CURRENT DATASET:")
    print("=" * 50)
    print("1. 🎯 Download TUPAC16 for proper Ki-67 ground truth")
    print("2. 🔄 Retrain models using biological classification")
    print("3. 📊 Compare performance with current file-size method")
    print("4. 📖 Document methodology for research paper")
    print("5. 🧪 Validate on multiple datasets for robustness")
    
    print(f"\n🚀 QUICK START WITH BREAKHIS:")
    print("=" * 50)
    print("# Download BreakHis dataset")
    print("wget https://web.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz")
    print("tar -xzf BreaKHis_v1.tar.gz")
    print("")
    print("# The dataset structure will be:")
    print("BreaKHis_v1/")
    print("├── histology_slides/")
    print("│   ├── breast/")
    print("│   │   ├── benign/")
    print("│   │   └── malignant/")
    print("└── Folds.csv  # Train/test splits")


if __name__ == "__main__":
    main()
