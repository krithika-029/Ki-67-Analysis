import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json

def load_annotation_data(annotation_path):
    """Load annotation data from H5 file"""
    try:
        with h5py.File(annotation_path, 'r') as f:
            print("Keys in annotation file:", list(f.keys()))
            
            # Try to read common annotation formats
            if 'annotations' in f.keys():
                annotations = f['annotations'][:]
                print("Annotations shape:", annotations.shape)
                return annotations
            elif 'labels' in f.keys():
                labels = f['labels'][:]
                print("Labels shape:", labels.shape)
                return labels
            elif 'mask' in f.keys():
                mask = f['mask'][:]
                print("Mask shape:", mask.shape)
                return mask
            else:
                # Print all datasets and their shapes
                for key in f.keys():
                    try:
                        data = f[key][:]
                        print(f"{key}: shape {data.shape}, dtype {data.dtype}")
                        if len(data.shape) == 2:  # Could be a mask
                            return data
                    except:
                        print(f"{key}: Could not read as array")
                        
    except Exception as e:
        print(f"Error loading annotation: {e}")
        return None

def analyze_annotations_vs_prediction():
    """Compare ground truth annotations with our model prediction"""
    
    # Paths
    image_path = "/Users/chinthan/ki7/test_image_6.png"
    annotation_path = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/annotations/test/positive/6.h5"
    
    print("=== Ki-67 Detection Validation ===")
    print(f"Image: {image_path}")
    print(f"Annotation: {annotation_path}")
    print()
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is not None:
            print(f"Image loaded: {image.shape}")
        else:
            print("Could not load image")
            return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load annotations
    annotation_data = load_annotation_data(annotation_path)
    
    if annotation_data is not None:
        print(f"Annotation data loaded successfully")
        print(f"Annotation data type: {type(annotation_data)}")
        print(f"Annotation data shape: {annotation_data.shape}")
        print(f"Annotation data range: {annotation_data.min()} to {annotation_data.max()}")
        print(f"Unique values in annotation: {np.unique(annotation_data)}")
        
        # Calculate ground truth Ki-67 index
        if len(annotation_data.shape) == 2:
            # Assume it's a binary mask where 1 = Ki-67 positive, 0 = negative
            total_pixels = annotation_data.size
            positive_pixels = np.sum(annotation_data > 0)
            ground_truth_ki67_index = (positive_pixels / total_pixels) * 100
            
            print(f"\n=== Ground Truth Analysis ===")
            print(f"Total pixels: {total_pixels}")
            print(f"Positive pixels: {positive_pixels}")
            print(f"Ground truth Ki-67 index: {ground_truth_ki67_index:.2f}%")
            
            print(f"\n=== Model Prediction vs Ground Truth ===")
            print(f"Model predicted Ki-67 index: 40.11%")
            print(f"Ground truth Ki-67 index: {ground_truth_ki67_index:.2f}%")
            
            difference = abs(40.11 - ground_truth_ki67_index)
            print(f"Absolute difference: {difference:.2f}%")
            
            if difference < 10:
                print("✅ EXCELLENT: Model prediction within 10% of ground truth")
            elif difference < 20:
                print("✅ GOOD: Model prediction within 20% of ground truth")
            else:
                print("❌ NEEDS IMPROVEMENT: Model prediction differs significantly")
                
            # Save visualization
            try:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Original image
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                # Ground truth annotation
                axes[1].imshow(annotation_data, cmap='hot')
                axes[1].set_title(f'Ground Truth\nKi-67 Index: {ground_truth_ki67_index:.2f}%')
                axes[1].axis('off')
                
                # Comparison
                axes[2].bar(['Ground Truth', 'Model Prediction'], 
                           [ground_truth_ki67_index, 40.11],
                           color=['blue', 'green'])
                axes[2].set_ylabel('Ki-67 Index (%)')
                axes[2].set_title('Comparison')
                
                plt.tight_layout()
                plt.savefig('/Users/chinthan/ki7/annotation_validation.png', dpi=150, bbox_inches='tight')
                print(f"\n📊 Visualization saved to: /Users/chinthan/ki7/annotation_validation.png")
                
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        else:
            print("Annotation data format not recognized as binary mask")
            print("Available data for manual inspection:")
            print(annotation_data)
    
    else:
        print("Could not load annotation data")

if __name__ == "__main__":
    analyze_annotations_vs_prediction()
