# Recovery Instructions for Ki67 Project

## Size Reduction Applied
- **Before**: 3.5GB
- **Target After**: ~500MB

## What Was Removed:
1. **.venv**: Python virtual environment (1.0GB)
2. **frontend/node_modules**: Node.js dependencies (605MB) 
3. **Excess model files**: Kept only 2 best models instead of 11 (~860MB reduction)

## To Restore Development Environment:

### Python Environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### Frontend Dependencies:
```bash
cd frontend
npm install
```

### Models Kept:
- `Ki67_STABLE_Champion_EfficientNet_B5_Champion_FINAL_90.98_20250620_142507.pth` (109MB) - Best accuracy
- `Ki67_Advanced_EfficientNet-B2_best_model_20250619_105754.pth` (30MB) - Efficient backup

### Models Removed (can retrain if needed):
- Ki67_ViT_best_model (327MB)
- Ki67_Advanced_ConvNeXt-Tiny_best_model (106MB)
- Ki67_Advanced_Swin-Tiny_best_model (105MB)
- Ki67_InceptionV3_best_model (93MB)
- Ki67_ResNet50_best_model (90MB)
- And others...

## Quick Setup Script:
Run `./setup.sh` to restore development environment.
