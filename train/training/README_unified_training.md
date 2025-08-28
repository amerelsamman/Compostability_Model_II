# Unified Training System

This unified training system replaces the individual training scripts for each property while maintaining **identical functionality and results**.

## **🚀 Quick Start**

```bash
# Train any property with default settings
python train_unified.py --property cobb --input data/cobb/polymerblends_for_ml_featurized.csv --output models/cobb/v4/

# Train with custom last N blends placement
python train_unified.py --property ts --input data/ts/polymerblends_for_ml_featurized.csv --output models/ts/v4/ --last_n_training 10
```

## **📋 Available Properties**

| Property | Target Columns | Default Strategy | Oversampling |
|----------|----------------|------------------|--------------|
| `ts` | `property1`, `property2` | Last 4 in training | 10x |
| `cobb` | `property` | Last 10 in training | None |
| `wvtr` | `property` | Last 21 in training | None |
| `otr` | `property` | Last 21 in training, Last 2 in testing | None |
| `adhesion` | `property` | Last 5 in testing | None |
| `eab` | `property1`, `property2` | Last 4 in training | 2x |
| `eol` | `max_L`, `t0` | Last 4 in training | 10x |

## **⚙️ Command Line Arguments**

- `--property`: Property to train (required)
- `--input`: Input CSV file path (required)
- `--output`: Output directory path (required)
- `--last_n_training`: Override default number of last blends in training
- `--last_n_testing`: Override default number of last blends in testing

## **🎯 Examples for Each Property**

### **Tensile Strength (TS)**
```bash
# Default: Last 4 blends in training, 10x oversampling
python train_unified.py --property ts \
  --input data/ts/polymerblends_for_ml_featurized.csv \
  --output models/ts/v4/

# Custom: Last 10 blends in training, 5x oversampling
python train_unified.py --property ts \
  --input data/ts/polymerblends_for_ml_featurized.csv \
  --output models/ts/v4/ \
  --last_n_training 10
```

### **Cobb Angle**
```bash
# Default: Last 10 blends in training
python train_unified.py --property cobb \
  --input data/cobb/polymerblends_for_ml_featurized.csv \
  --output models/cobb/v4/

# Custom: Last 5 blends in training
python train_unified.py --property cobb \
  --input data/cobb/polymerblends_for_ml_featurized.csv \
  --output models/cobb/v4/ \
  --last_n_training 5
```

### **WVTR**
```bash
# Default: Last 21 blends in training
python train_unified.py --property wvtr \
  --input data/wvtr/polymerblends_for_ml_featurized.csv \
  --output models/wvtr/v4/

# Custom: Last 10 blends in training
python train_unified.py --property wvtr \
  --input data/wvtr/polymerblends_for_ml_featurized.csv \
  --output models/wvtr/v4/ \
  --last_n_training 10
```

### **OTR**
```bash
# Default: Last 21 in training, Last 2 in testing
python train_unified.py --property otr \
  --input data/otr/polymerblends_for_ml_featurized.csv \
  --output models/otr/v4/

# Custom: Last 10 in training, Last 5 in testing
python train_unified.py --property otr \
  --input data/otr/polymerblends_for_ml_featurized.csv \
  --output models/otr/v4/ \
  --last_n_training 10 \
  --last_n_testing 5
```

### **Adhesion**
```bash
# Default: Last 5 blends in testing
python train_unified.py --property adhesion \
  --input data/adhesion/polymerblends_for_ml_featurized.csv \
  --output models/adhesion/v4/

# Custom: Last 10 blends in testing
python train_unified.py --property adhesion \
  --input data/adhesion/polymerblends_for_ml_featurized.csv \
  --output models/adhesion/v4/ \
  --last_n_testing 10
```

### **EAB (Elongation at Break)**
```bash
# Default: Last 4 blends in training, 2x oversampling
python train_unified.py --property eab \
  --input data/eab/polymerblends_for_ml_featurized.csv \
  --output models/eab/v4/

# Custom: Last 8 blends in training, 5x oversampling
python train_unified.py --property eab \
  --input data/eab/polymerblends_for_ml_featurized.csv \
  --output models/eab/v4/ \
  --last_n_training 8
```

### **EOL (Compostability)**
```bash
# Default: Last 4 blends in training, 10x oversampling
python train_unified.py --property eol \
  --input data/eol/polymerblends_for_ml_featurized.csv \
  --output models/eol/v4/

# Custom: Last 6 blends in training, 15x oversampling
python train_unified.py --property eol \
  --input data/eol/polymerblends_for_ml_featurized.csv \
  --output models/eol/v4/ \
  --last_n_training 6
```

## **📊 Output Files**

For each training run, you get:

1. **`comprehensive_polymer_model_results.png`** - Training vs test performance plots
2. **`last_N_blends_performance.png`** - Performance on last N blends (if applicable)
3. **`feature_importance_TARGET.csv`** - Feature importance rankings
4. **`comprehensive_polymer_model_TARGET.pkl`** - Trained model files

## **🔄 Migration from Individual Scripts**

The unified system produces **identical results** to the original scripts:

| Original Script | Unified Command |
|-----------------|-----------------|
| `training_ts/model.py` | `train_unified.py --property ts` |
| `training_cobb/model.py` | `train_unified.py --property cobb` |
| `training_wvtr/model.py` | `train_unified.py --property wvtr` |
| `training_otr/model.py` | `train_unified.py --property otr` |
| `training_adhesion/model.py` | `train_unified.py --property adhesion` |
| `training_eab/model.py` | `train_unified.py --property eab` |
| `training_compost/model.py` | `train_unified.py --property eol` |

## **🔧 Customization**

### **Override Default Behavior**
```bash
# Force last 20 blends in training for any property
python train_unified.py --property cobb \
  --input data/cobb/polymerblends_for_ml_featurized.csv \
  --output models/cobb/v4/ \
  --last_n_training 20

# Force last 10 blends in testing for any property
python train_unified.py --property ts \
  --input data/ts/polymerblends_for_ml_featurized.csv \
  --output models/ts/v4/ \
  --last_n_testing 10
```

### **Standard Split (No Special Handling)**
```bash
# Use standard 80/20 split without special last N handling
python train_unified.py --property cobb \
  --input data/cobb/polymerblends_for_ml_featurized.csv \
  --output models/cobb/v4/ \
  --last_n_training 0 \
  --last_n_testing 0
```

## **✅ Benefits**

1. **Identical Results**: Same functionality, same outputs, same plots
2. **Easy Control**: Command-line control over last N blends placement
3. **Unified Interface**: One script for all properties
4. **Flexible**: Override defaults for custom experiments
5. **Maintainable**: Single codebase for all training logic
6. **Future-Proof**: Easy to modify model parameters for all properties

## **⚠️ Important Notes**

- **Identical Functionality**: This system reproduces the exact behavior of the original scripts
- **Same Random Seeds**: Uses same random_state=42 for reproducible results
- **Same Model Parameters**: Identical XGBoost hyperparameters
- **Same Preprocessing**: Identical feature handling and data cleaning
- **Same Plots**: Identical visualization outputs
