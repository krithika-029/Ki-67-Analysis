#!/usr/bin/env python3
"""
Simplified Ki-67 Champion Training Script - Device Debug Version

This is a simplified version focused on fixing device mismatch issues.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import timm

def create_simple_champion_model(device):
    """Create a simple model for testing device issues"""
    print("🏗️ Creating Simple Champion Model for Device Testing...")
    
    try:
        # Create EfficientNet-B3 (smaller than B4)
        model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1)
        
        # Move to device immediately
        model = model.to(device)
        
        # Replace classifier with explicit device placement
        in_features = model.classifier.in_features
        new_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 1)
        )
        model.classifier = new_classifier.to(device)
        
        # Ensure entire model is on device
        model = model.to(device)
        
        # Verify device placement
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"⚠️  Parameter {name} on wrong device: {param.device}")
                param.data = param.data.to(device)
        
        print(f"✅ Simple model created and verified on {device}")
        return model
        
    except Exception as e:
        print(f"❌ Failed to create simple model: {e}")
        return None

def test_forward_pass(model, device):
    """Test a simple forward pass"""
    print("🧪 Testing forward pass...")
    
    try:
        # Create test input on device
        test_input = torch.randn(2, 3, 224, 224, device=device)
        test_target = torch.randn(2, 1, device=device)
        
        print(f"   Input device: {test_input.device}")
        print(f"   Target device: {test_target.device}")
        print(f"   Model device: {next(model.parameters()).device}")
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            print(f"   Output device: {output.device}")
            print(f"   Output shape: {output.shape}")
        
        # Test loss calculation
        criterion = nn.BCEWithLogitsLoss()
        if output.dim() == 1:
            output = output.unsqueeze(1)
        
        loss = criterion(output, test_target)
        print(f"   Loss: {loss.item():.4f}")
        print(f"✅ Forward pass successful!")
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_training_step(model, device):
    """Test a single training step"""
    print("🚀 Testing training step...")
    
    try:
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # Create training data on device
        inputs = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.randn(2, 1, device=device)
        
        print(f"   Training inputs device: {inputs.device}")
        print(f"   Training targets device: {targets.device}")
        
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        
        print(f"   Training outputs device: {outputs.device}")
        
        # Loss and backward
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f"   Training loss: {loss.item():.4f}")
        print(f"✅ Training step successful!")
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function for device testing"""
    print("🏆 Ki-67 Champion Device Testing")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"🚀 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        torch.cuda.empty_cache()
    
    # Create simple model
    model = create_simple_champion_model(device)
    if model is None:
        return
    
    # Test forward pass
    if not test_forward_pass(model, device):
        return
    
    # Test training step
    if not test_training_step(model, device):
        return
    
    print(f"\n🎉 All device tests passed!")
    print(f"✅ Your champion training should work without device issues")

if __name__ == "__main__":
    main()
