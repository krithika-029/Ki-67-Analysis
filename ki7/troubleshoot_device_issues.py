#!/usr/bin/env python3
"""
Device Troubleshooting Script for Ki-67 Champion Training

This script helps identify and fix device mismatch issues in the champion training.
"""

import torch
import torch.nn as nn
import numpy as np

def test_device_compatibility():
    """Test device compatibility and identify issues"""
    print("🔍 Device Compatibility Test")
    print("=" * 50)
    
    # Check CUDA availability
    print(f"🚀 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU Count: {torch.cuda.device_count()}")
        print(f"🚀 Current Device: {torch.cuda.current_device()}")
        print(f"🚀 Device Name: {torch.cuda.get_device_name(0)}")
        print(f"🚀 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🔧 Using device: {device}")
    
    # Test basic tensor operations
    print("\n📊 Testing Basic Tensor Operations:")
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        y = torch.randn(2, 1).to(device)
        print(f"✅ Created tensors on {device}")
        print(f"   x.device: {x.device}")
        print(f"   y.device: {y.device}")
    except Exception as e:
        print(f"❌ Failed to create tensors: {e}")
        return False
    
    # Test model creation
    print("\n🏗️ Testing Model Creation:")
    try:
        # Simple test model
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 1)
        ).to(device)
        
        print(f"✅ Created model on {device}")
        print(f"   Model device: {next(model.parameters()).device}")
        
        # Test forward pass
        with torch.no_grad():
            output = model(x)
            print(f"✅ Forward pass successful")
            print(f"   Output device: {output.device}")
            print(f"   Output shape: {output.shape}")
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False
    
    # Test mixup operations
    print("\n🔄 Testing Mixup Operations:")
    try:
        from train_efficientnet_champion import mixup_data, cutmix_data
        
        # Test mixup
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.3)
        print(f"✅ Mixup successful")
        print(f"   mixed_x.device: {mixed_x.device}")
        print(f"   y_a.device: {y_a.device}")
        print(f"   y_b.device: {y_b.device}")
        
        # Test cutmix
        cut_x, cut_y_a, cut_y_b, cut_lam = cutmix_data(x.clone(), y, alpha=0.8)
        print(f"✅ CutMix successful")
        print(f"   cut_x.device: {cut_x.device}")
        print(f"   cut_y_a.device: {cut_y_a.device}")
        print(f"   cut_y_b.device: {cut_y_b.device}")
        
    except Exception as e:
        print(f"❌ Augmentation test failed: {e}")
        return False
    
    # Test mixed precision
    print("\n⚡ Testing Mixed Precision:")
    try:
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        model.train()
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                output = model(x)
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            print(f"✅ Mixed precision training successful")
        else:
            output = model(x)
            if output.dim() == 1:
                output = output.unsqueeze(1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            print(f"✅ Regular precision training successful")
            
        print(f"   Loss: {loss.item():.4f}")
        
    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False
    
    print(f"\n🎉 All device compatibility tests passed!")
    return True

def diagnose_device_issues():
    """Diagnose common device issues"""
    print("\n🔧 Device Issue Diagnosis:")
    print("-" * 30)
    
    issues_found = []
    
    # Check for CUDA initialization
    try:
        torch.cuda.init()
        print("✅ CUDA initialized successfully")
    except Exception as e:
        issues_found.append(f"CUDA initialization failed: {e}")
    
    # Check memory
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated(0)
            reserved = torch.cuda.memory_reserved(0)
            print(f"✅ GPU Memory - Allocated: {allocated/1024**2:.1f}MB, Reserved: {reserved/1024**2:.1f}MB")
        except Exception as e:
            issues_found.append(f"Memory check failed: {e}")
    
    # Check tensor creation consistency
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(5):
            t = torch.randn(10, 10, device=device)
            assert t.device == device, f"Tensor {i} on wrong device: {t.device}"
        print("✅ Consistent tensor creation")
    except Exception as e:
        issues_found.append(f"Inconsistent tensor creation: {e}")
    
    if issues_found:
        print(f"\n⚠️  Issues found:")
        for issue in issues_found:
            print(f"   • {issue}")
    else:
        print(f"\n✅ No device issues detected")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    print("🏆 Ki-67 Champion Device Troubleshooting")
    print("=" * 60)
    
    # Run compatibility tests
    compatibility_ok = test_device_compatibility()
    
    # Run diagnostics
    diagnostics_ok = diagnose_device_issues()
    
    print(f"\n📋 Summary:")
    print(f"   Compatibility Test: {'✅ PASS' if compatibility_ok else '❌ FAIL'}")
    print(f"   Diagnostics: {'✅ PASS' if diagnostics_ok else '❌ FAIL'}")
    
    if compatibility_ok and diagnostics_ok:
        print(f"\n🎉 Your system is ready for champion training!")
        print(f"💡 The device mismatch issue may be in the dataset loading or specific model components.")
    else:
        print(f"\n⚠️  Device issues detected. Please fix before training.")
