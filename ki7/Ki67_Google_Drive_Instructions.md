# Ki-67 Training Script - Google Drive Setup Instructions

## Dataset Setup
1. Upload your dataset ZIP file to Google Drive in this structure:
   ```
   MyDrive/
   └── ki67_Dataset/
       └── Ki67_Dataset_for_Colab.zip
   ```

## Running the Script
1. Upload `ki67_training_complete.py` to Google Colab
2. Run the script - it will automatically:
   - Mount Google Drive
   - Extract the dataset from the ZIP file
   - Train the models
   - Save results

## Output Files
After training, you'll find these files in your Google Drive:

### MyDrive (root folder) - Model Files:
- `Ki67_InceptionV3_best_model_YYYYMMDD_HHMMSS.pth`
- `Ki67_ResNet50_best_model_YYYYMMDD_HHMMSS.pth`
- `Ki67_ViT_best_model_YYYYMMDD_HHMMSS.pth`
- `Ki67_ensemble_weights_YYYYMMDD_HHMMSS.json`

### MyDrive/Ki67_Results/ - Analysis Files:
- Training histories (`.pkl` files)
- Results summaries (`.json` files)
- Confusion matrices (`.png` files)
- Detailed predictions (`.json` files)

## Loading Saved Models
```python
import torch

# Load a saved model
checkpoint = torch.load('/content/drive/MyDrive/Ki67_InceptionV3_best_model_YYYYMMDD_HHMMSS.pth')
print(f"Model performance: {checkpoint['performance_summary']}")

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
```

## Using Ensemble Weights
```python
import json

# Load ensemble weights
with open('/content/drive/MyDrive/Ki67_ensemble_weights_YYYYMMDD_HHMMSS.json', 'r') as f:
    ensemble_config = json.load(f)
    
weights = ensemble_config['weights']
model_order = ensemble_config['model_order']
print(f"Ensemble weights: {dict(zip(model_order, weights))}")
```

## Troubleshooting
- If dataset not found: Check the ZIP file path matches exactly
- If models fail to save: Ensure Google Drive has sufficient space
- If training fails: Check GPU availability and dataset format
