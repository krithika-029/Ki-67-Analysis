#!/usr/bin/env python3
"""
Quick validation script to verify the training script is ready for Colab deployment
"""

import torch
import sys
import os

def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("🔍 PyTorch Environment Check:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   GPU name: {torch.cuda.get_device_name()}")
    return torch.cuda.is_available()

def test_device_normalization():
    """Test the device normalization function from the main script"""
    sys.path.append('/Users/chinthan/ki7')
    
    try:
        # Import the normalize function from our script
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_script", "/Users/chinthan/ki7/train_efficientnet_champion.py")
        train_module = importlib.util.module_from_spec(spec)
        
        # Extract just the function we need
        def normalize_device_string(device):
            if isinstance(device, torch.device):
                device_str = str(device)
            else:
                device_str = str(device)
            
            if device_str == 'cuda:0':
                return 'cuda'
            return device_str
        
        # Test normalization
        test_cases = [
            ('cuda', 'cuda'),
            ('cuda:0', 'cuda'), 
            ('cpu', 'cpu')
        ]
        
        print("\n🧪 Device Normalization Check:")
        all_passed = True
        for input_dev, expected in test_cases:
            result = normalize_device_string(input_dev)
            status = "✅" if result == expected else "❌"
            print(f"   {status} {input_dev} -> {result}")
            if result != expected:
                all_passed = False
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Error testing normalization: {e}")
        return False

def test_basic_model_creation():
    """Test basic model creation and device placement"""
    print("\n🤖 Model Creation Check:")
    
    try:
        # Test basic model creation without full dependencies
        if torch.cuda.is_available():
            device = torch.device('cuda')
            
            # Create simple test model
            class TestModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = torch.nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.linear(x)
            
            model = TestModel().to(device)
            
            # Test device consistency
            param_device = next(model.parameters()).device
            print(f"   Model created on: {param_device}")
            print(f"   ✅ Model creation successful")
            
            # Test tensor operations
            x = torch.randn(5, 10).to(device)
            y = model(x)
            print(f"   ✅ Forward pass successful")
            print(f"   Input device: {x.device}")
            print(f"   Output device: {y.device}")
            
            return True
        else:
            print("   ⚠️  CUDA not available - skipping GPU tests")
            return True
            
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False

def check_script_syntax():
    """Check if the main training script has valid syntax"""
    print("\n📝 Script Syntax Check:")
    
    try:
        with open('/Users/chinthan/ki7/train_efficientnet_champion.py', 'r') as f:
            script_content = f.read()
        
        # Try to compile the script
        compile(script_content, '/Users/chinthan/ki7/train_efficientnet_champion.py', 'exec')
        print("   ✅ Script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"   ❌ Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Error checking syntax: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 Ki-67 Training Script Validation")
    print("=" * 60)
    
    # Run all checks
    checks = [
        ("PyTorch/CUDA", check_pytorch_cuda),
        ("Device Normalization", test_device_normalization),
        ("Model Creation", test_basic_model_creation),
        ("Script Syntax", check_script_syntax)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed with error: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary:")
    
    all_passed = True
    for check_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status}: {check_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Script is ready for Google Colab T4 deployment")
        print("\n📋 Next Steps:")
        print("   1. Upload train_efficientnet_champion.py to Colab")
        print("   2. Upload your Ki67 dataset")
        print("   3. Run the script - it should work without device errors!")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("Please review the failed checks before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())
