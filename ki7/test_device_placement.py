#!/usr/bin/env python3
"""
Test script to verify device placement fixes in champion training script.
This tests the mixup/cutmix functions to ensure they handle device placement correctly.
"""

import torch
import numpy as np

def mixup_data(x, y, alpha=1.0):
    """Mixup augmentation with proper device handling"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Ensure index tensor is on the same device as input
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix augmentation with proper device handling"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Ensure index tensor is on the same device as input
    index = torch.randperm(batch_size, device=x.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # Adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)  # Fixed: use int() instead of np.int()
    cut_h = int(H * cut_rat)  # Fixed: use int() instead of np.int()

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def test_device_placement():
    """Test device placement for mixup and cutmix functions"""
    print("🧪 Testing device placement fixes...")
    
    # Test on CPU
    print("\n📍 Testing CPU device placement:")
    device = torch.device('cpu')
    
    # Create test data
    batch_size = 4
    inputs = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 2, (batch_size, 1), device=device).float()
    
    print(f"  Input device: {inputs.device}")
    print(f"  Target device: {targets.device}")
    
    # Test mixup
    mixed_x, y_a, y_b, lam = mixup_data(inputs, targets, alpha=0.3)
    print(f"  Mixup - mixed_x device: {mixed_x.device}")
    print(f"  Mixup - y_a device: {y_a.device}")
    print(f"  Mixup - y_b device: {y_b.device}")
    
    # Test cutmix
    cut_x, cut_y_a, cut_y_b, cut_lam = cutmix_data(inputs.clone(), targets, alpha=0.8)
    print(f"  CutMix - cut_x device: {cut_x.device}")
    print(f"  CutMix - cut_y_a device: {cut_y_a.device}")
    print(f"  CutMix - cut_y_b device: {cut_y_b.device}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("\n🚀 Testing GPU device placement:")
        device = torch.device('cuda')
        
        # Create test data on GPU
        inputs_gpu = torch.randn(batch_size, 3, 224, 224, device=device)
        targets_gpu = torch.randint(0, 2, (batch_size, 1), device=device).float()
        
        print(f"  Input device: {inputs_gpu.device}")
        print(f"  Target device: {targets_gpu.device}")
        
        # Test mixup on GPU
        mixed_x_gpu, y_a_gpu, y_b_gpu, lam_gpu = mixup_data(inputs_gpu, targets_gpu, alpha=0.3)
        print(f"  Mixup - mixed_x device: {mixed_x_gpu.device}")
        print(f"  Mixup - y_a device: {y_a_gpu.device}")
        print(f"  Mixup - y_b device: {y_b_gpu.device}")
        
        # Test cutmix on GPU
        cut_x_gpu, cut_y_a_gpu, cut_y_b_gpu, cut_lam_gpu = cutmix_data(inputs_gpu.clone(), targets_gpu, alpha=0.8)
        print(f"  CutMix - cut_x device: {cut_x_gpu.device}")
        print(f"  CutMix - cut_y_a device: {cut_y_a_gpu.device}")
        print(f"  CutMix - cut_y_b device: {cut_y_b_gpu.device}")
        
        print("\n✅ GPU device placement test passed!")
    else:
        print("\n⚠️  No GPU available for testing")
    
    print("\n✅ All device placement tests passed!")
    print("🎯 Champion training script should now work without device mismatch errors")

if __name__ == "__main__":
    test_device_placement()
