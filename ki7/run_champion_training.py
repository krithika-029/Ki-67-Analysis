"""
🚀 ONE-CLICK COLAB CHAMPION TRAINING

Copy and paste this entire code block into a Google Colab cell and run it.
This will verify your setup and then run the champion training if everything looks good.

Make sure you have:
1. Selected T4 GPU runtime
2. Uploaded Ki67_Dataset_for_Colab.zip to Google Drive
3. Have the train_efficientnet_champion.py file uploaded to Colab

USAGE: Just paste this entire block into a Colab cell and run!
"""

# First, run the verification script
print("Step 1: Verifying Colab setup...")
exec(open('verify_colab_setup.py').read())

print("\n" + "="*80)
print("Step 2: Starting champion training...")
print("="*80)

# Then run the main training script
exec(open('train_efficientnet_champion.py').read())

print("\n" + "="*80)
print("🎉 CHAMPION TRAINING COMPLETE!")
print("="*80)
print("📁 Check your Google Drive for:")
print("   - Champion model (.pth file)")
print("   - Training results (.json file)")
print("   - Training plots (.png files)")
print("\n🎯 Next: Download the champion model and add it to your ensemble!")
