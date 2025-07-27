#!/usr/bin/env python3
"""
Quick Colab T4 Setup Verification Script

Run this in Google Colab BEFORE running the main champion training script
to verify your environment and dataset are properly set up.
"""

import os
import zipfile
from pathlib import Path

def verify_colab_environment():
    """Verify Colab environment setup"""
    print("🔍 Verifying Google Colab T4 Environment")
    print("=" * 50)
    
    # Check if in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
    except ImportError:
        print("❌ Not running in Google Colab")
        return False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ Memory: {gpu_memory:.1f} GB")
            
            if 'T4' in gpu_name:
                print("✅ Tesla T4 detected - perfect for champion training!")
            else:
                print(f"⚠️  Non-T4 GPU: {gpu_name} (may work but T4 recommended)")
        else:
            print("❌ No GPU available - Please enable GPU in Runtime settings")
            return False
    except ImportError:
        print("⚠️  PyTorch not available - will be installed by main script")
    
    return True

def verify_drive_mount():
    """Verify Google Drive is mounted"""
    print("\n📁 Verifying Google Drive Mount")
    print("-" * 30)
    
    try:
        from google.colab import drive
        
        # Try to mount drive
        drive.mount('/content/drive')
        
        if os.path.exists('/content/drive/MyDrive'):
            print("✅ Google Drive mounted successfully")
            return True
        else:
            print("❌ Google Drive mount failed")
            return False
            
    except Exception as e:
        print(f"❌ Error mounting Google Drive: {e}")
        return False

def verify_dataset():
    """Verify dataset is available and properly structured"""
    print("\n📦 Verifying Ki67 Dataset")
    print("-" * 25)
    
    # Look for dataset in common locations
    possible_paths = [
        "/content/drive/MyDrive/Ki67_Dataset_for_Colab.zip",
        "/content/drive/MyDrive/Ki67_Dataset/Ki67_Dataset_for_Colab.zip",
        "/content/drive/MyDrive/ki67_dataset.zip",
        "/content/drive/MyDrive/Ki67_Dataset.zip"
    ]
    
    dataset_path = None
    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f"✅ Found dataset: {os.path.basename(path)}")
            print(f"   Location: {path}")
            break
    
    if dataset_path is None:
        print("❌ Dataset ZIP not found in Google Drive")
        print("\n📋 Upload your dataset ZIP to one of these locations:")
        for path in possible_paths:
            print(f"   {path}")
        return False
    
    # Check dataset size
    try:
        size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
        print(f"📊 Dataset size: {size_mb:.1f} MB")
        
        if size_mb < 10:
            print("⚠️  Dataset seems very small (< 10MB)")
        elif size_mb > 5000:
            print("⚠️  Dataset is very large (> 5GB) - may take time to extract")
        else:
            print("✅ Dataset size looks reasonable")
            
    except Exception as e:
        print(f"⚠️  Could not check dataset size: {e}")
    
    # Try to peek inside the ZIP
    try:
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            files = zip_ref.namelist()[:10]  # First 10 files
            
            has_images = any('images' in f for f in files)
            has_train = any('train' in f for f in files)
            has_test = any('test' in f for f in files)
            
            print(f"📋 ZIP contents preview:")
            for f in files:
                print(f"   {f}")
            if len(zip_ref.namelist()) > 10:
                print(f"   ... and {len(zip_ref.namelist()) - 10} more files")
            
            print(f"\n🔍 Structure check:")
            print(f"   Contains 'images' folder: {'✅' if has_images else '❌'}")
            print(f"   Contains 'train' data: {'✅' if has_train else '❌'}")
            print(f"   Contains 'test' data: {'✅' if has_test else '❌'}")
            
            if has_images and has_train and has_test:
                print("✅ Dataset structure looks good!")
                return True
            else:
                print("⚠️  Dataset structure may need verification")
                return True  # Still allow to proceed
                
    except Exception as e:
        print(f"⚠️  Could not examine ZIP contents: {e}")
        return True  # Still allow to proceed
    
    return True

def check_disk_space():
    """Check available disk space"""
    print("\n💽 Checking Disk Space")
    print("-" * 20)
    
    try:
        # Check /content space (for dataset extraction)
        statvfs = os.statvfs('/content')
        free_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
        print(f"📊 Available space in /content: {free_space_gb:.1f} GB")
        
        if free_space_gb < 2:
            print("⚠️  Low disk space - may have issues extracting dataset")
        else:
            print("✅ Sufficient disk space available")
            
        # Check Drive space
        try:
            drive_statvfs = os.statvfs('/content/drive/MyDrive')
            drive_free_gb = (drive_statvfs.f_frsize * drive_statvfs.f_bavail) / (1024**3)
            print(f"📊 Available space in Google Drive: {drive_free_gb:.1f} GB")
            
            if drive_free_gb < 1:
                print("⚠️  Low Google Drive space - may have issues saving models")
            else:
                print("✅ Sufficient Google Drive space")
        except:
            print("⚠️  Could not check Google Drive space")
            
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")

def main():
    """Run all verification checks"""
    print("🚀 Google Colab T4 Champion Training - Environment Verification")
    print("=" * 65)
    
    checks = [
        ("Colab Environment", verify_colab_environment),
        ("Google Drive", verify_drive_mount),
        ("Ki67 Dataset", verify_dataset),
        ("Disk Space", check_disk_space)
    ]
    
    results = []
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 65)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 65)
    
    all_good = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if not result:
            all_good = False
    
    print("\n" + "=" * 65)
    if all_good:
        print("🎉 ALL CHECKS PASSED!")
        print("🚀 Ready to run champion training script!")
        print("\nNext step: Run the main training script:")
        print("   exec(open('train_efficientnet_champion.py').read())")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("🔧 Please fix the issues above before running champion training")
        print("\nCommon solutions:")
        print("   - Enable GPU in Runtime → Change runtime type")
        print("   - Upload dataset ZIP to Google Drive")
        print("   - Ensure sufficient storage space")

if __name__ == "__main__":
    main()
