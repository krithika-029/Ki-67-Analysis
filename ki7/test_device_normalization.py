#!/usr/bin/env python3
"""
Test script to verify device normalization functionality
"""

import torch
import sys

def normalize_device_string(device):
    """Normalize device string to handle cuda vs cuda:0 mismatch"""
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = str(device)
    
    # Normalize cuda:0 to cuda for consistent comparison
    if device_str == 'cuda:0':
        return 'cuda'
    return device_str

def test_device_normalization():
    """Test device string normalization"""
    print("🧪 Testing device string normalization...")
    
    # Test cases
    test_cases = [
        ('cuda', 'cuda'),
        ('cuda:0', 'cuda'),
        ('cpu', 'cpu'),
        (torch.device('cuda'), 'cuda'),
        (torch.device('cuda:0'), 'cuda'),
        (torch.device('cpu'), 'cpu'),
    ]
    
    all_passed = True
    for input_device, expected in test_cases:
        result = normalize_device_string(input_device)
        status = "✅" if result == expected else "❌"
        print(f"{status} {input_device} -> {result} (expected: {expected})")
        if result != expected:
            all_passed = False
    
    print(f"\n{'✅ All tests passed!' if all_passed else '❌ Some tests failed!'}")
    return all_passed

def test_device_comparison():
    """Test device comparison with normalization"""
    print("\n🧪 Testing device comparison scenarios...")
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - skipping CUDA tests")
        return True
    
    # Create tensors on different device representations
    device_cuda = torch.device('cuda')
    device_cuda0 = torch.device('cuda:0')
    
    tensor1 = torch.randn(2, 3).to(device_cuda)
    tensor2 = torch.randn(2, 3).to(device_cuda0)
    
    print(f"Tensor 1 device: {tensor1.device}")
    print(f"Tensor 2 device: {tensor2.device}")
    print(f"Direct comparison: {tensor1.device == tensor2.device}")
    
    # Test normalized comparison
    norm1 = normalize_device_string(tensor1.device)
    norm2 = normalize_device_string(tensor2.device)
    print(f"Normalized device 1: {norm1}")
    print(f"Normalized device 2: {norm2}")
    print(f"Normalized comparison: {norm1 == norm2}")
    
    return norm1 == norm2

def main():
    """Main test function"""
    print("🔧 Device Normalization Test Suite")
    print("=" * 50)
    
    # Test normalization function
    norm_passed = test_device_normalization()
    
    # Test device comparison
    comp_passed = test_device_comparison()
    
    print("\n" + "=" * 50)
    if norm_passed and comp_passed:
        print("🎉 All tests passed! Device normalization is working correctly.")
        return 0
    else:
        print("💥 Some tests failed! Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
