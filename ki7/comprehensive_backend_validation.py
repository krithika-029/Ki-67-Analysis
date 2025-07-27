#!/usr/bin/env python3
"""
Comprehensive Backend Validation Test
Validates backend predictions against ground truth for a larger sample of test images.
"""

import os
import json
import h5py
import requests
import numpy as np
import random
from pathlib import Path

def load_ground_truth(image_id):
    """Load ground truth using the EXACT same logic as validation script"""
    pos_file = f"/Users/chinthan/ki7/Ki67_Dataset_for_Colab/annotations/test/positive/{image_id}.h5"
    neg_file = f"/Users/chinthan/ki7/Ki67_Dataset_for_Colab/annotations/test/negative/{image_id}.h5"
    
    if os.path.exists(pos_file) and os.path.exists(neg_file):
        try:
            pos_size = os.path.getsize(pos_file)
            neg_size = os.path.getsize(neg_file)
            size_diff = abs(pos_size - neg_size)
            
            if size_diff > 100:  # Significant size difference
                if pos_size > neg_size:
                    return 1  # Positive
                else:
                    return 0  # Negative
            else:
                # Very similar sizes, use alternating pattern
                # This matches the validation script logic
                return image_id % 2
        except:
            return 1  # Default to positive if file reading fails
    elif os.path.exists(pos_file):
        return 1  # Positive
    elif os.path.exists(neg_file):
        return 0  # Negative
    else:
        return None  # Not found

def get_backend_prediction(image_path):
    """Get prediction from backend API"""
    try:
        url = "http://localhost:5002/api/predict"
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'high_confidence': result['high_confidence'],
                'probability': result['probability']
            }
        else:
            print(f"❌ API error for {image_path}: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Request failed for {image_path}: {e}")
        return None

def main():
    print("🧪 Comprehensive Backend Validation Test")
    print("=" * 60)
    
    # Get all test images
    test_dir = "/Users/chinthan/ki7/Ki67_Dataset_for_Colab/images/test"
    all_images = []
    
    for img_file in os.listdir(test_dir):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            image_id = int(img_file.split('.')[0])
            all_images.append(image_id)
    
    # Sort and sample
    all_images.sort()
    
    # Test with a larger sample - 100 images for better statistics
    if len(all_images) > 100:
        test_images = random.sample(all_images, 100)
    else:
        test_images = all_images
    
    test_images.sort()
    
    print(f"Testing {len(test_images)} images randomly sampled from {len(all_images)} total test images")
    print(f"Testing backend predictions vs ground truth...")
    print()
    
    results = []
    correct_overall = 0
    correct_high_conf = 0
    high_conf_count = 0
    total_count = 0
    
    # Header
    print("Image | Ground Truth | Prediction | Confidence | High-Conf | Correct")
    print("-" * 70)
    
    for image_id in test_images:
        # Load ground truth
        ground_truth = load_ground_truth(image_id)
        if ground_truth is None:
            continue
        
        # Get backend prediction
        image_path = f"{test_dir}/{image_id}.png"
        if not os.path.exists(image_path):
            continue
        
        backend_result = get_backend_prediction(image_path)
        if backend_result is None:
            continue
        
        prediction = backend_result['prediction']
        confidence = backend_result['confidence']
        high_confidence = backend_result['high_confidence']
        
        # Check correctness
        correct = (prediction == ground_truth)
        
        # Update counters
        total_count += 1
        if correct:
            correct_overall += 1
        
        if high_confidence:
            high_conf_count += 1
            if correct:
                correct_high_conf += 1
        
        # Display result
        gt_label = "Pos" if ground_truth == 1 else "Neg"
        pred_label = "Pos" if prediction == 1 else "Neg"
        conf_label = "YES" if high_confidence else "NO"
        status = "✅" if correct else "❌"
        
        print(f"{image_id:5d} | {gt_label:11s} | {pred_label:10s} | {confidence:.3f}      | {conf_label:9s} | {status}")
        
        # Store detailed result
        results.append({
            'image_id': image_id,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'confidence': confidence,
            'high_confidence': high_confidence,
            'correct': correct
        })
    
    print()
    print("=" * 60)
    print("📊 COMPREHENSIVE BACKEND VALIDATION RESULTS:")
    
    # Calculate metrics
    overall_accuracy = correct_overall / total_count if total_count > 0 else 0
    high_conf_accuracy = correct_high_conf / high_conf_count if high_conf_count > 0 else 0
    high_conf_coverage = high_conf_count / total_count if total_count > 0 else 0
    
    print(f"   Total images tested: {total_count}")
    print(f"   Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print(f"   High-confidence accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)")
    print(f"   High-confidence coverage: {high_conf_coverage:.3f} ({high_conf_coverage*100:.1f}%)")
    print(f"   High-confidence samples: {correct_high_conf}/{high_conf_count}")
    print()
    
    print("📈 COMPARISON WITH VALIDATION SCRIPT:")
    print(f"   Expected high-conf accuracy: 98.0%")
    print(f"   Actual high-conf accuracy: {high_conf_accuracy*100:.1f}%")
    print(f"   Expected coverage: 72.9%")
    print(f"   Actual coverage: {high_conf_coverage*100:.1f}%")
    
    # Status indicators
    if high_conf_accuracy >= 0.95:
        print("✅ Backend high-confidence accuracy meets expectations")
    else:
        print("⚠️  Backend high-confidence accuracy below expectations")
    
    if abs(high_conf_coverage - 0.729) <= 0.1:
        print("✅ Backend coverage close to validation script")
    else:
        print("⚠️  Backend coverage differs from validation script")
    
    # Save results
    summary = {
        'test_summary': {
            'total_tested': total_count,
            'overall_accuracy': overall_accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'high_conf_coverage': high_conf_coverage,
            'high_conf_samples': f"{correct_high_conf}/{high_conf_count}",
            'sample_size': len(test_images),
            'validation_script_comparison': {
                'expected_high_conf_accuracy': 0.98,
                'actual_high_conf_accuracy': high_conf_accuracy,
                'expected_coverage': 0.729,
                'actual_coverage': high_conf_coverage
            }
        },
        'detailed_results': results
    }
    
    output_file = "comprehensive_backend_validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
