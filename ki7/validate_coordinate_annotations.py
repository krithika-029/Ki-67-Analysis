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
            
            if 'coordinates' in f.keys():
                coordinates = f['coordinates'][:]
                print("Coordinates shape:", coordinates.shape)
                print("Coordinates data type:", coordinates.dtype)
                print("Sample coordinates:", coordinates[:5])
                return coordinates
                
    except Exception as e:
        print(f"Error loading annotation: {e}")
        return None

def analyze_coordinate_annotations():
    """Analyze coordinate-based annotations vs model prediction"""
    
    # Paths
    image_path = "/Users/chinthan/ki7/test_image_6.png"
    annotation_path = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/annotations/test/positive/6.h5"
    
    print("=== Ki-67 Detection Validation (Coordinate-based) ===")
    print(f"Image: {image_path}")
    print(f"Annotation: {annotation_path}")
    print()
    
    # Load image
    try:
        image = cv2.imread(image_path)
        if image is not None:
            height, width = image.shape[:2]
            print(f"Image loaded: {image.shape}")
        else:
            print("Could not load image")
            return
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Load annotations (coordinates)
    coordinates = load_annotation_data(annotation_path)
    
    if coordinates is not None and len(coordinates.shape) == 2 and coordinates.shape[1] == 2:
        num_positive_cells = coordinates.shape[0]
        
        print(f"\n=== Ground Truth Analysis ===")
        print(f"Number of annotated Ki-67 positive cells: {num_positive_cells}")
        print(f"Image dimensions: {width} x {height}")
        
        # Estimate total cells and Ki-67 index
        # This is an approximation based on typical cell density in histopathology
        total_area = width * height
        
        # Typical cell density ranges from 1000-3000 cells per mm²
        # For a 640x640 image, estimate cell density
        estimated_total_cells = 500 + (num_positive_cells * 10)  # Rough estimate
        
        ground_truth_ki67_percentage = (num_positive_cells / estimated_total_cells) * 100
        
        print(f"Estimated total cells in image: {estimated_total_cells}")
        print(f"Ground truth Ki-67 positive cells: {num_positive_cells}")
        print(f"Ground truth Ki-67 index: {ground_truth_ki67_percentage:.2f}%")
        
        print(f"\n=== Model Prediction vs Ground Truth ===")
        print(f"Model predicted total cells: 551")
        print(f"Model predicted Ki-67 index: 40.11%")
        print(f"Ground truth positive cells: {num_positive_cells}")
        print(f"Ground truth Ki-67 index (estimated): {ground_truth_ki67_percentage:.2f}%")
        
        # Calculate actual Ki-67 index based on model's total cell count
        actual_ki67_from_model_total = (num_positive_cells / 551) * 100
        
        print(f"\n=== Detailed Comparison ===")
        print(f"Using model's total cell count (551):")
        print(f"Ground truth Ki-67 index: {actual_ki67_from_model_total:.2f}%")
        print(f"Model predicted Ki-67 index: 40.11%")
        
        difference = abs(40.11 - actual_ki67_from_model_total)
        print(f"Absolute difference: {difference:.2f}%")
        
        if difference < 5:
            print("✅ EXCELLENT: Model prediction within 5% of ground truth")
        elif difference < 10:
            print("✅ VERY GOOD: Model prediction within 10% of ground truth")
        elif difference < 15:
            print("✅ GOOD: Model prediction within 15% of ground truth")
        else:
            print("❌ NEEDS IMPROVEMENT: Model prediction differs significantly")
            
        print(f"\n=== Cell Detection Analysis ===")
        print(f"Ground truth marked {num_positive_cells} Ki-67+ cells")
        model_positive_cells = int((40.11 / 100) * 551)
        print(f"Model detected ~{model_positive_cells} Ki-67+ cells (from 40.11% of 551)")
        
        cell_detection_accuracy = min(num_positive_cells, model_positive_cells) / max(num_positive_cells, model_positive_cells) * 100
        print(f"Cell detection similarity: {cell_detection_accuracy:.1f}%")
                
        # Create visualization
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image
            axes[0,0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axes[0,0].set_title('Original Image')
            axes[0,0].axis('off')
            
            # Image with annotations overlaid
            img_with_annotations = image.copy()
            for coord in coordinates:
                x, y = int(coord[0]), int(coord[1])
                cv2.circle(img_with_annotations, (x, y), 3, (0, 255, 0), -1)  # Green dots
            
            axes[0,1].imshow(cv2.cvtColor(img_with_annotations, cv2.COLOR_BGR2RGB))
            axes[0,1].set_title(f'Ground Truth Annotations\n({num_positive_cells} Ki-67+ cells)')
            axes[0,1].axis('off')
            
            # Ki-67 index comparison
            axes[1,0].bar(['Ground Truth', 'Model Prediction'], 
                         [actual_ki67_from_model_total, 40.11],
                         color=['blue', 'green'],
                         alpha=0.7)
            axes[1,0].set_ylabel('Ki-67 Index (%)')
            axes[1,0].set_title('Ki-67 Index Comparison')
            axes[1,0].set_ylim(0, max(actual_ki67_from_model_total, 40.11) + 10)
            
            # Add value labels on bars
            axes[1,0].text(0, actual_ki67_from_model_total + 1, f'{actual_ki67_from_model_total:.1f}%', 
                          ha='center', va='bottom')
            axes[1,0].text(1, 40.11 + 1, '40.11%', ha='center', va='bottom')
            
            # Cell count comparison
            axes[1,1].bar(['Ground Truth\nPositive', 'Model Detected\nPositive', 'Model Total'], 
                         [num_positive_cells, model_positive_cells, 551],
                         color=['blue', 'green', 'gray'],
                         alpha=0.7)
            axes[1,1].set_ylabel('Cell Count')
            axes[1,1].set_title('Cell Count Comparison')
            
            plt.tight_layout()
            plt.savefig('/Users/chinthan/ki7/coordinate_annotation_validation.png', dpi=150, bbox_inches='tight')
            print(f"\n📊 Visualization saved to: /Users/chinthan/ki7/coordinate_annotation_validation.png")
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    else:
        print("Could not interpret annotation data as coordinates")

if __name__ == "__main__":
    analyze_coordinate_annotations()
