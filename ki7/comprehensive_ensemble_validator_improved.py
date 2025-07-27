#!/usr/bin/env python3
"""
Comprehensive Validation with Improved Confidence
================================================

Re-run the comprehensive validation using the improved confidence calculation
to see how it affects the relationship between confidence and accuracy.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import our improved ensemble
from simple_improved_confidence import ImprovedKi67Ensemble

# Import the annotation parsing code
import h5py

def load_ground_truth_annotations(image_name, annotations_dir):
    """Load ground truth annotations for an image"""
    base_name = image_name.replace('.png', '')
    
    # Check both positive and negative annotation folders
    positive_path = annotations_dir / "positive" / f"{base_name}.h5"
    negative_path = annotations_dir / "negative" / f"{base_name}.h5"
    
    if positive_path.exists():
        coords = load_h5_coordinates(str(positive_path))
        return coords, "positive"
    elif negative_path.exists():
        coords = load_h5_coordinates(str(negative_path))
        return coords, "negative"
    else:
        return None, "unknown"

def load_h5_coordinates(annotation_path):
    """Load coordinates from H5 file"""
    try:
        with h5py.File(annotation_path, 'r') as f:
            if 'coordinates' in f.keys():
                coordinates = f['coordinates'][:]
                return coordinates
    except Exception as e:
        print(f"Error loading {annotation_path}: {e}")
        return None

def validate_with_improved_confidence():
    """Run comprehensive validation with improved confidence"""
    
    print("🎯 Comprehensive Validation with Improved Confidence")
    print("=" * 55)
    
    # Initialize improved ensemble
    ensemble = ImprovedKi67Ensemble()
    
    if len(ensemble.models) == 0:
        print("❌ No models loaded. Cannot validate.")
        return
    
    # Dataset paths
    test_images_dir = Path("/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test")
    annotations_dir = Path("/Users/chinthan/ki7/Ki67_Dataset_for_Colab/annotations/test")
    
    print(f"📁 Images: {test_images_dir}")
    print(f"📁 Annotations: {annotations_dir}")
    
    # Get test images (use first 20 for comparison with previous results)
    test_images = sorted([f for f in test_images_dir.glob("*.png")])[:20]
    print(f"🔍 Testing on {len(test_images)} images")
    
    validation_results = []
    
    for i, image_path in enumerate(test_images):
        image_name = image_path.name
        print(f"\\nProcessing {i+1}/{len(test_images)}: {image_name}")
        
        # Get improved prediction
        result = ensemble.predict_with_improved_confidence(str(image_path))
        
        if 'error' in result:
            print(f"   ❌ Prediction error: {result['error']}")
            continue
        
        # Parse ground truth annotations
        coordinates, true_class = load_ground_truth_annotations(image_name, annotations_dir)
        
        if coordinates is None:
            print(f"   ⚠️  No annotation data found for {image_name}")
            continue
        
        try:
            ground_truth_positive_cells = len(coordinates) if coordinates is not None else 0
            
            # Estimate total cells (using model's estimate as done in original validation)
            # This approach assumes that positive cells represent a reasonable fraction
            if true_class == "positive" and ground_truth_positive_cells > 0:
                # Estimate total cells assuming Ki-67 index of 10-50% for positive images
                estimated_total_cells = max(ground_truth_positive_cells * 4, 800)
            else:
                # For negative class or no positive cells, estimate a baseline
                estimated_total_cells = 1000
            
            ground_truth_ki67 = (ground_truth_positive_cells / estimated_total_cells) * 100
            
        except Exception as e:
            print(f"   ❌ Annotation processing error: {e}")
            continue
        
        # Extract results
        model_class = result['prediction_label'].lower()
        model_prob = result['probability']
        model_ki67 = model_prob * 100  # Convert to percentage
        
        # Calculate classification accuracy
        class_correct = (true_class == model_class)
        ki67_error = abs(ground_truth_ki67 - model_ki67)
        
        validation_result = {
            'image_name': image_name,
            'true_class': true_class,
            'model_class': model_class,
            'class_correct': class_correct,
            'ground_truth_positive_cells': ground_truth_positive_cells,
            'ground_truth_ki67': ground_truth_ki67,
            'model_ki67': model_ki67,
            'model_probability': model_prob,
            
            # Original confidence metrics
            'raw_confidence': result['raw_confidence'] * 100,
            
            # Improved confidence metrics
            'improved_confidence': result['improved_confidence'] * 100,
            'agreement_confidence': result['agreement_confidence'] * 100,
            'variance_confidence': result['variance_confidence'] * 100,
            'entropy_confidence': result['entropy_confidence'] * 100,
            'magnitude_confidence': result['magnitude_confidence'] * 100,
            
            # Additional info
            'confidence_level': result['confidence_level'],
            'clinical_recommendation': result['clinical_recommendation'],
            'model_agreement': result['model_agreement'],
            'probability_variance': result['probability_variance'],
            'ki67_error': ki67_error,
            'individual_probabilities': result['individual_probabilities']
        }
        
        validation_results.append(validation_result)
        
        print(f"   Ground Truth: {true_class} ({ground_truth_ki67:.1f}% Ki-67)")
        print(f"   Prediction: {model_class} ({model_ki67:.1f}% Ki-67)")
        print(f"   Raw Confidence: {validation_result['raw_confidence']:.1f}%")
        print(f"   Improved Confidence: {validation_result['improved_confidence']:.1f}% ({result['confidence_level']})")
        print(f"   Class Correct: {class_correct}, Ki-67 Error: {ki67_error:.1f}%")
    
    # Calculate summary statistics
    if validation_results:
        classification_accuracy = np.mean([r['class_correct'] for r in validation_results]) * 100
        mean_ki67_error = np.mean([r['ki67_error'] for r in validation_results])
        median_ki67_error = np.median([r['ki67_error'] for r in validation_results])
        mean_raw_confidence = np.mean([r['raw_confidence'] for r in validation_results])
        mean_improved_confidence = np.mean([r['improved_confidence'] for r in validation_results])
        
        # Analyze confidence vs accuracy relationship
        high_conf_results = [r for r in validation_results if r['improved_confidence'] >= 80]
        low_conf_results = [r for r in validation_results if r['improved_confidence'] < 80]
        
        high_conf_accuracy = np.mean([r['class_correct'] for r in high_conf_results]) * 100 if high_conf_results else 0
        low_conf_accuracy = np.mean([r['class_correct'] for r in low_conf_results]) * 100 if low_conf_results else 0
        
        summary = {
            'classification_accuracy': classification_accuracy,
            'mean_ki67_error': mean_ki67_error,
            'median_ki67_error': median_ki67_error,
            'mean_raw_confidence': mean_raw_confidence,
            'mean_improved_confidence': mean_improved_confidence,
            'total_images': len(validation_results),
            'high_confidence_cases': len(high_conf_results),
            'low_confidence_cases': len(low_conf_results),
            'high_confidence_accuracy': high_conf_accuracy,
            'low_confidence_accuracy': low_conf_accuracy,
            'confidence_accuracy_gap': high_conf_accuracy - low_conf_accuracy
        }
        
        print(f"\\n📊 VALIDATION SUMMARY")
        print("=" * 35)
        print(f"Classification Accuracy: {classification_accuracy:.1f}%")
        print(f"Mean Ki-67 Error: {mean_ki67_error:.1f}%")
        print(f"Median Ki-67 Error: {median_ki67_error:.1f}%")
        print(f"Mean Raw Confidence: {mean_raw_confidence:.1f}%")
        print(f"Mean Improved Confidence: {mean_improved_confidence:.1f}%")
        print(f"\\nConfidence Analysis:")
        print(f"High Confidence (≥80%): {len(high_conf_results)} cases, {high_conf_accuracy:.1f}% accuracy")
        print(f"Low Confidence (<80%): {len(low_conf_results)} cases, {low_conf_accuracy:.1f}% accuracy")
        print(f"Confidence-Accuracy Gap: {high_conf_accuracy - low_conf_accuracy:.1f}%")
        
        # Save results
        output_data = {
            'validation_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'method': 'improved_confidence_validation',
            'summary': summary,
            'detailed_results': validation_results
        }
        
        output_file = "improved_confidence_validation_results.json"
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\\n💾 Results saved to {output_file}")
        
        # Create confidence vs accuracy plot
        create_confidence_analysis_plot(validation_results)
        
        return validation_results
    
    else:
        print("❌ No validation results obtained")
        return []

def create_confidence_analysis_plot(results):
    """Create plots comparing old vs new confidence metrics"""
    
    # Extract data for plotting
    raw_confidences = [r['raw_confidence'] for r in results]
    improved_confidences = [r['improved_confidence'] for r in results]
    accuracies = [1 if r['class_correct'] else 0 for r in results]
    ki67_errors = [r['ki67_error'] for r in results]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Improved Confidence Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Raw Confidence vs Accuracy
    axes[0, 0].scatter(raw_confidences, accuracies, alpha=0.7, color='red', s=60)
    axes[0, 0].set_xlabel('Raw Confidence (%)')
    axes[0, 0].set_ylabel('Correct Classification')
    axes[0, 0].set_title('Raw Confidence vs Accuracy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Improved Confidence vs Accuracy
    axes[0, 1].scatter(improved_confidences, accuracies, alpha=0.7, color='green', s=60)
    axes[0, 1].set_xlabel('Improved Confidence (%)')
    axes[0, 1].set_ylabel('Correct Classification')
    axes[0, 1].set_title('Improved Confidence vs Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confidence comparison
    axes[1, 0].scatter(raw_confidences, improved_confidences, alpha=0.7, color='blue', s=60)
    axes[1, 0].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='y=x')
    axes[1, 0].set_xlabel('Raw Confidence (%)')
    axes[1, 0].set_ylabel('Improved Confidence (%)')
    axes[1, 0].set_title('Raw vs Improved Confidence')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confidence vs Ki-67 Error
    axes[1, 1].scatter(improved_confidences, ki67_errors, alpha=0.7, color='purple', s=60)
    axes[1, 1].set_xlabel('Improved Confidence (%)')
    axes[1, 1].set_ylabel('Ki-67 Error (%)')
    axes[1, 1].set_title('Improved Confidence vs Ki-67 Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('improved_confidence_analysis.png', dpi=300, bbox_inches='tight')
    print("📈 Confidence analysis plot saved: improved_confidence_analysis.png")
    plt.close()

def main():
    """Main validation function"""
    results = validate_with_improved_confidence()
    
    if results:
        print("\\n✅ Improved confidence validation complete!")
        print("🔍 Key improvements:")
        print("   • Multi-factor confidence calculation")
        print("   • Model agreement analysis")
        print("   • Variance-based uncertainty")
        print("   • Clinical interpretability")
    else:
        print("❌ Validation failed")

if __name__ == "__main__":
    main()
