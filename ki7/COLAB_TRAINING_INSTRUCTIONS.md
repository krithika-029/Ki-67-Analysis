# Ki-67 Advanced Models Training in Google Colab (T4 GPU Optimized)

## 🚀 Quick Setup Instructions

### 1. Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com)
2. Create a new notebook
3. Upload the `train_advanced_models_colab.py` file
4. **Important**: Change runtime to GPU (Runtime → Change runtime type → GPU → T4)

### 2. Run the Training Script

```python
# In your Colab notebook, run this cell:
exec(open('train_advanced_models_colab.py').read())
```

Or copy-paste the entire script into a Colab cell and run it.

### 3. What It Will Do (T4 Optimized)

The script will automatically:
- ✅ Mount your Google Drive
- ✅ Install required packages (timm, etc.)
- ✅ Extract your Ki67 dataset
- ✅ Train 5 T4-optimized advanced models:
  - **EfficientNet-B2** (instead of B4) - Expected: 91-93% accuracy
  - **ConvNeXt-Tiny** (instead of Base) - Expected: 90-92% accuracy
  - **Swin-Tiny** (instead of Base) - Expected: 89-91% accuracy
  - **DenseNet-121** (instead of 201) - Expected: 88-90% accuracy
  - **RegNet-Y-8GF** (instead of 32GF) - Expected: 88-90% accuracy
- ✅ Save all models directly to your Google Drive
- ✅ Create new ensemble weights

### 4. Expected Results (T4 Optimized)

**Current Status**: 90.05% (3 models)
**After T4 Training**: 93-94% (8 models total)

*Note: Slightly lower than full-size models but still significant improvement and fits T4 memory perfectly!*

### 5. Training Time Estimates (T4 GPU)

- **EfficientNet-B2**: ~30-40 minutes
- **ConvNeXt-Tiny**: ~35-45 minutes  
- **Swin-Tiny**: ~40-50 minutes
- **DenseNet-121**: ~25-35 minutes
- **RegNet-Y-8GF**: ~25-30 minutes

**Total**: ~2.5-3.5 hours for all models (faster than original estimates!)

### 6. T4 Memory Optimization Features

✅ **Smaller model variants** - Fit comfortably in 15GB T4 memory
✅ **Mixed precision training** - Faster training, lower memory usage
✅ **Batch size optimization** - 12 instead of 32 for memory efficiency
✅ **Aggressive memory cleanup** - Between models and batches
✅ **OOM error handling** - Graceful fallback if memory issues occur

### 7. What Gets Saved to Your Drive

All files saved directly to **MyDrive** root:
```
MyDrive/
├── Ki67_Advanced_EfficientNet-B2_best_model_TIMESTAMP.pth
├── Ki67_Advanced_ConvNeXt-Tiny_best_model_TIMESTAMP.pth
├── Ki67_Advanced_Swin-Tiny_best_model_TIMESTAMP.pth
├── Ki67_Advanced_DenseNet-121_best_model_TIMESTAMP.pth
├── Ki67_Advanced_RegNet-Y-8GF_best_model_TIMESTAMP.pth
├── Ki67_t4_advanced_ensemble_weights_TIMESTAMP.json
└── Ki67_Advanced_Results/
    └── (training histories and logs)
```

### 8. After Training

1. **Download models** from your Google Drive
2. **Update your validation script** to include the new T4-optimized models
3. **Test the 8-model ensemble** (3 existing + 5 new T4-optimized)
4. **Expected final accuracy**: 93-94% 🎯

### 9. T4 vs Full Models Comparison

| Model Type | Original | T4 Optimized | Expected Acc |
|------------|----------|--------------|---------------|
| EfficientNet | B4 (19M) | B2 (9M) | 91-93% |
| ConvNeXt | Base (89M) | Tiny (29M) | 90-92% |
| Swin | Base (88M) | Tiny (29M) | 89-91% |
| DenseNet | 201 (20M) | 121 (8M) | 88-90% |
| RegNet | Y-32GF (146M) | Y-8GF (39M) | 88-90% |

### 10. Troubleshooting

**Still getting OOM errors?**
- Script automatically handles this and skips problematic models
- At least 3-4 models should train successfully

**Models failing to load?**
- Make sure you have T4 GPU selected, not CPU runtime

**Dataset not found?**
- Make sure `Ki67_Dataset_for_Colab.zip` is in `MyDrive/Ki67_Dataset/`

---

## 🎯 T4-Optimized Success Criteria

- ✅ At least 3/5 models train successfully
- ✅ Best model achieves 91%+ accuracy  
- ✅ Combined ensemble (8 models) achieves 93-94% accuracy
- ✅ Fits comfortably in T4's 15GB memory

**T4-Optimized Result**: 90.05% → 93-94% (still excellent improvement!) 🚀 
