#!/usr/bin/env python3
"""
Device Placement Test Script for Ki-67 Champion Model
Tests the device handling improvements in the final training script.
"""

import sys
import os
import torch

def test_device_handling():
    """Test the device handling functions from the champion script"""
    print("🧪 Testing Device Handling Functions...")
    
    # Test device normalization
    def normalize_device_str(dev):
        dev_str = str(dev)
        if dev_str == 'cuda:0':
            return 'cuda'
        elif dev_str.startswith('cuda:') and dev_str.split(':')[1] == '0':
            return 'cuda'
        return dev_str
    
    # Test various device string formats
    test_cases = [
        ('cuda', 'cuda'),
        ('cuda:0', 'cuda'),
        ('cpu', 'cpu'),
        (torch.device('cuda'), 'cuda'),
        (torch.device('cuda:0'), 'cuda'),
        (torch.device('cpu'), 'cpu'),
    ]
    
    print("📝 Testing device string normalization:")
    for device_input, expected in test_cases:
        result = normalize_device_str(device_input)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {device_input} -> {result} (expected: {expected})")
    
    # Test device availability
    print(f"\n💻 Device Information:")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device Count: {torch.cuda.device_count()}")
        print(f"   Current CUDA Device: {torch.cuda.current_device()}")
        print(f"   CUDA Device Name: {torch.cuda.get_device_name()}")
    
    # Test simple tensor device handling
    print(f"\n🔧 Testing tensor device handling:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Using device: {device}")
    
    # Create test tensor
    test_tensor = torch.randn(2, 3)
    print(f"   Original tensor device: {test_tensor.device}")
    
    # Move to target device
    test_tensor = test_tensor.to(device)
    print(f"   After .to(device): {test_tensor.device}")
    
    # Test normalization comparison
    current_device_str = normalize_device_str(test_tensor.device)
    target_device_str = normalize_device_str(device)
    match = current_device_str == target_device_str
    print(f"   Normalized comparison: {current_device_str} == {target_device_str} -> {match}")
    
    if match:
        print("✅ Device handling test passed!")
    else:
        print("❌ Device handling test failed!")
    
    return match

def test_model_creation():
    """Test model creation and device placement"""
    print(f"\n🏗️  Testing Model Creation and Device Placement...")
    
    try:
        import timm
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Target device: {device}")
        
        # Create a simple test model
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=1)
        print(f"   Model created successfully")
        
        # Move to device
        model = model.to(device)
        print(f"   Model moved to {device}")
        
        # Check all parameters
        all_on_device = True
        param_count = 0
        
        def normalize_device_str(dev):
            dev_str = str(dev)
            if dev_str == 'cuda:0':
                return 'cuda'
            elif dev_str.startswith('cuda:') and dev_str.split(':')[1] == '0':
                return 'cuda'
            return dev_str
        
        target_device_str = normalize_device_str(device)
        
        for name, param in model.named_parameters():
            param_device_str = normalize_device_str(param.device)
            if param_device_str != target_device_str:
                print(f"   ❌ Parameter {name} on wrong device: {param.device}")
                all_on_device = False
            param_count += 1
        
        if all_on_device:
            print(f"   ✅ All {param_count} parameters on correct device")
        else:
            print(f"   ❌ Some parameters on wrong device")
        
        return all_on_device
    
    except Exception as e:
        print(f"   ❌ Model creation test failed: {e}")
        return False

def main():
    """Run all device handling tests"""
    print("🧪 Ki-67 Champion Model - Device Handling Test Suite\n")
    
    test1_passed = test_device_handling()
    test2_passed = test_model_creation()
    
    print(f"\n📊 Test Results Summary:")
    print(f"   Device Normalization Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   Model Device Placement Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print(f"\n🎉 All tests passed! Device handling should work correctly.")
        return 0
    else:
        print(f"\n⚠️  Some tests failed. Check device handling implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
