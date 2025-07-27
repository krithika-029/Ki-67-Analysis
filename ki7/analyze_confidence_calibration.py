import json

def analyze_confidence_calibration():
    """Analyze confidence calibration issues"""
    
    # Load validation results
    with open('/Users/chinthan/ki7/ensemble_validation_results.json', 'r') as f:
        data = json.load(f)
    
    results = data['detailed_results']
    
    print("🔍 CONFIDENCE CALIBRATION ANALYSIS")
    print("="*50)
    
    # High confidence cases (>90%)
    high_conf_cases = [r for r in results if r['model_confidence'] > 90]
    
    print(f"\n📊 HIGH CONFIDENCE CASES (>90%):")
    print(f"Total: {len(high_conf_cases)} out of {len(results)} images")
    
    for case in high_conf_cases:
        correct = "✅" if case['class_correct'] else "❌"
        print(f"\n{case['image_name']} {correct}")
        print(f"   Confidence: {case['model_confidence']:.1f}%")
        print(f"   True class: {case['true_class']} | Model: {case['model_class']}")
        print(f"   Ground truth Ki-67: {case['ground_truth_ki67']:.1f}%")
        print(f"   Model Ki-67: {case['model_ki67']:.1f}%")
        print(f"   Error: {case['ki67_error']:.1f}%")
    
    # Analyze confidence vs accuracy correlation
    print(f"\n📈 CONFIDENCE vs ACCURACY CORRELATION:")
    
    confidence_ranges = [
        (90, 100, "Very High (90-100%)"),
        (80, 90, "High (80-90%)"),
        (70, 80, "Good (70-80%)"),
        (60, 70, "Moderate (60-70%)"),
        (0, 60, "Low (<60%)")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        range_cases = [r for r in results if min_conf <= r['model_confidence'] < max_conf]
        if range_cases:
            correct_count = sum(1 for r in range_cases if r['class_correct'])
            accuracy = (correct_count / len(range_cases)) * 100
            avg_error = sum(r['ki67_error'] for r in range_cases) / len(range_cases)
            
            print(f"\n{label}:")
            print(f"   Count: {len(range_cases)} images")
            print(f"   Accuracy: {accuracy:.1f}% ({correct_count}/{len(range_cases)})")
            print(f"   Avg Ki-67 Error: {avg_error:.1f}%")
    
    # The calibration problem
    print(f"\n⚠️  CALIBRATION PROBLEM IDENTIFIED:")
    high_conf_wrong = [r for r in high_conf_cases if not r['class_correct']]
    
    if high_conf_wrong:
        print(f"   • {len(high_conf_wrong)} cases with >90% confidence were WRONG")
        print(f"   • This indicates OVERCONFIDENCE in the model")
        print(f"   • Model certainty doesn't match actual accuracy")
        
        print(f"\n❌ SPECIFIC OVERCONFIDENT MISTAKES:")
        for case in high_conf_wrong:
            print(f"   {case['image_name']}: {case['model_confidence']:.1f}% confident but WRONG")
            print(f"      True: {case['true_class']} vs Model: {case['model_class']}")
    
    print(f"\n🧠 WHY HIGH CONFIDENCE WITH WRONG RESULTS?")
    print(f"   1. MODEL OVERCONFIDENCE: Ensemble may be overconfident")
    print(f"   2. TRAINING BIAS: Models trained on different data distribution")
    print(f"   3. CALIBRATION NEEDED: Confidence scores need recalibration")
    print(f"   4. ENSEMBLE AGREEMENT: All 3 models might agree on wrong answer")
    print(f"   5. FEATURE CONFUSION: Models detecting wrong visual patterns")

if __name__ == "__main__":
    analyze_confidence_calibration()
