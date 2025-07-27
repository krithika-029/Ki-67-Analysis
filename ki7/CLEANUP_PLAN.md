# Ki-67 Project Cleanup Plan

## Identified Duplicates and Redundant Files

### 1. Python Script Duplicates

#### Enhanced Validation Files (4 similar files)
- `enhanced_validation.py` (765 lines) - ❌ **REMOVE** (base version)
- `enhanced_validation_fixed.py` (764 lines) - ❌ **REMOVE** (minor fixes)
- `enhanced_validation_clean.py` (1563 lines) - ✅ **KEEP** (most comprehensive)
- `enhanced_validation_with_visualizations.py` (1469 lines) - ✅ **KEEP** (visualization-focused)

**Action**: Remove the first two files as they are earlier versions with fewer features.

#### Training Script Duplicates
- `ki67_training_complete.py` - Check for duplicates with other training files
- `train_additional_models.py` vs `train_additional_models_colab.py`
- `train_advanced_models.py` vs `train_advanced_models_colab.py`

### 2. Jupyter Notebook Duplicates
- `Copy of Ki67_Training_Google_Colab copy.ipynb` - ❌ **REMOVE** (obvious duplicate)
- `Ki67_Training_Colab_Notebook.ipynb` - ✅ **KEEP** (main notebook)
- `validate_ki67_models.ipynb` - ✅ **KEEP** (different purpose)

### 3. Result Files (JSON) - Multiple timestamped versions
Keep only the latest and most relevant results:

#### Enhanced Validation Results
- `enhanced_validation_20250619_152747.json` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_164900.json` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_174226.json` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_184934.json` - ✅ **KEEP** (latest comprehensive)
- `enhanced_validation_results_20250619_195242.json` - ❌ **REMOVE** (incomplete)
- `enhanced_validation_results_20250619_211821.json` - ✅ **KEEP** (latest)

#### Other Results
- `complete_ensemble_validation_20250619_134522.json` - ❌ **REMOVE** (older)
- `complete_ensemble_validation_20250619_135049.json` - ✅ **KEEP** (latest)

### 4. Visualization Files (PNG) - Multiple timestamped versions
Keep only the latest and most relevant visualizations:

#### Enhanced Validation Plots
- `enhanced_validation_20250619_151634.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_152644.png` - ❌ **REMOVE** (older) 
- `enhanced_validation_20250619_163705.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_164752.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_174152.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_184919.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_195245.png` - ❌ **REMOVE** (older)
- `enhanced_validation_20250619_211822.png` - ✅ **KEEP** (latest)

### 5. Validation Script Consolidation
Several validation scripts with similar functionality:
- `validate_models.py` - ✅ **KEEP** (comprehensive validation)
- `validate_complete_ensemble.py` - ✅ **KEEP** (ensemble-specific)
- `quick_validate.py` - ❌ **REMOVE** (superseded by others)
- `corrected_validation.py` - ❌ **REMOVE** (incorporated into enhanced versions)

### 6. Organize Remaining Files

#### Create organized directory structure:
```
/results/
  /archive/           # Move old timestamped results here
  /latest/           # Keep only latest results
/scripts/
  /validation/       # All validation scripts
  /training/         # All training scripts
  /analysis/         # Analysis and diagnostic scripts
/visualizations/
  /archive/          # Old visualization files
  /latest/           # Latest visualization files
```

## Summary
- **Files to remove**: ~15-20 duplicate files
- **Space to reclaim**: Estimated 50-100 MB
- **Organization**: Better structure with logical groupings
- **Maintenance**: Easier to find and use relevant files

## Next Steps
1. Create backup of current state
2. Remove identified duplicate files
3. Create organized directory structure
4. Move remaining files to appropriate locations
5. Update any scripts that reference moved files
