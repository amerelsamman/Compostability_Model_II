# Polymer Blend Property Prediction System

A comprehensive machine learning platform for predicting polymer blend properties using molecular descriptors and advanced simulation techniques. This system combines experimental data with computational simulation to create large-scale training datasets for robust property prediction across multiple polymer types and environmental conditions.

## 🎯 Overview

This system predicts **7 key polymer blend properties** using molecular descriptors extracted from SMILES representations, combined with sophisticated data augmentation and machine learning techniques. The platform supports both single polymers and complex multi-component blends (up to 5 components) under various environmental conditions.

### Supported Properties

| Property | Unit | Environmental Parameters | Model Version | Description |
|----------|------|-------------------------|---------------|-------------|
| **WVTR** | g/m²/day | Temperature, RH, Thickness | v4 | Water Vapor Transmission Rate |
| **Tensile Strength** | MPa | Thickness | v4 | Mechanical strength (MD & TD) |
| **Elongation at Break** | % | Thickness | v4 | Material flexibility |
| **Cobb Value** | g/m² | None | v2 | Water absorption capacity |
| **OTR** | cc/m²/day | Temperature, RH, Thickness | v2 | Oxygen Transmission Rate |
| **Seal Strength** | N/15mm | Thickness | v2 | Heat sealing performance |
| **Home Compostability** | % disintegration | Thickness | v5 | Biodegradation potential |

## 🔄 Optimization Systems Overview

### When and Where to Use Optimization

The system includes **two complementary optimization systems** that can be used at different stages of the pipeline to improve prediction accuracy:

#### **1. Blend Optimization System** (`simulate_optimize_blends.py`)
- **Purpose**: Optimizes **polymer-polymer compatibility** (KI values) to minimize prediction errors
- **When to Use**: 
  - **Before model training** to improve simulation accuracy
  - **After model training** to fine-tune compatibility parameters
  - **When prediction errors are high** on validation blends
- **Pipeline Stage**: **Simulation Layer** - improves the underlying simulation rules
- **What it Optimizes**: KI values in `{property}_compatibility.yaml` files
- **Input**: Validation blends from `train/data/{property}/validationblends.csv`
- **Output**: Optimized KI values that improve MAE on validation data

#### **2. Additive Optimization System** (`simulate_optimize_additives.py`)
- **Purpose**: Optimizes **additive-polymer compatibility** (KI values) to achieve target percentage changes
- **When to Use**:
  - **When you want specific additive effects** (e.g., +20% WVTR increase)
  - **To calibrate additive behavior** to match experimental observations
  - **When additive effects don't match expectations**
- **Pipeline Stage**: **Additive Integration Layer** - fine-tunes additive effects
- **What it Optimizes**: KI values for additive-polymer pairs in compatibility files
- **Input**: Target percentage changes from `train/simulation/targets.csv`
- **Output**: Optimized KI values that achieve target additive effects

### **Pipeline Integration Flow**

```
1. Experimental Data Collection
   ↓
2. Data Generation Pipeline
   ├── Simulation Rules (can be optimized with Blend Optimization)
   ├── Additive Integration (can be optimized with Additive Optimization)
   └── Synthetic Data Generation
   ↓
3. Feature Extraction
   ↓
4. Model Training
   ↓
5. Prediction Engine
```

### **Quick Start Guide**

#### **For New Users:**
1. **Start with Blend Optimization** to improve base simulation accuracy
2. **Then use Additive Optimization** to fine-tune additive effects
3. **Retrain models** with optimized parameters
4. **Validate** on test data

#### **For Existing Users:**
- **High prediction errors?** → Use Blend Optimization
- **Additive effects wrong?** → Use Additive Optimization
- **Both issues?** → Use both systems in sequence

### **Command Quick Reference**

```bash
# Blend Optimization (improve simulation accuracy)
python simulate_optimize_blends.py --property wvtr --adaptive-lr

# Additive Optimization (calibrate additive effects)
python simulate_optimize_additives.py --property wvtr --tolerance 5.0

# Check results
git status  # See what files were updated
```

## 🏗️ System Architecture

### 1. **Data Generation Pipeline**

#### Experimental Foundation
- **Master Data**: Real experimental measurements stored in `masterdata.csv` files
- **Material Database**: 80+ polymer materials with SMILES representations
- **Property-Specific Datasets**: Separate experimental data for each property

#### Data Augmentation System (`train/simulation/`)
- **Simulation Rules**: Property-specific mathematical models for polymer blending
- **Material Compatibility**: Immiscibility rules and compatibility matrices
- **Environmental Modeling**: Temperature, humidity, and thickness effects
- **Additive Integration**: Advanced additive/filler system with UMM3 corrections
- **Pairwise Interactions**: Additive-polymer compatibility modeling

#### Synthetic Data Generation
- **Random Combinations**: Generates thousands of polymer blend combinations
- **Volume Fraction Sampling**: Uses Dirichlet distribution for realistic compositions
- **Environmental Variation**: Tests across realistic temperature/humidity/thickness ranges
- **Quality Control**: UMM3 correction and validation systems

### 2. **Feature Extraction System**

#### Molecular Descriptors (80+ features)
- **Hybridization States**: SP, SP2, SP3 for different elements (C, N, O, S, etc.)
- **Ring Systems**: Phenyls, cyclohexanes, cyclopentanes, thiophenes, heteroaromatic rings
- **Functional Groups**: 25+ chemical groups (esters, amides, alcohols, etc.)
- **Structural Features**: Branching factor, tree depth, chain lengths
- **Carbon Classification**: Primary, secondary, tertiary, quaternary carbons

#### Blend Processing
- **Volume Fraction Weighting**: Combines individual polymer features using volume fractions
- **Order Invariance**: Sorts polymers by SMILES to ensure consistent featurization
- **Environmental Integration**: Adds temperature, humidity, thickness as features

### 3. **Machine Learning Pipeline**

#### Model Architecture
- **Algorithm**: XGBoost gradient boosting models
- **Preprocessing**: Log transformations with property-specific offsets
- **Feature Engineering**: Categorical encoding and missing value handling
- **Dual Property Support**: Some properties predict multiple values (TS1+TS2, max_L+t0)

#### Training Strategy
- **Smart Data Splitting**: "Last N" strategy for temporal validation
- **Oversampling**: Property-specific oversampling for rare cases
- **Cross-Validation**: Robust model performance evaluation
- **Feature Selection**: Automatic exclusion of SMILES and metadata columns

### 4. **Prediction Engine**

#### Core Components
- **Input Validation**: Polymer/grade validation and volume fraction checking
- **Feature Extraction**: Real-time molecular descriptor calculation
- **Model Loading**: Dynamic model loading with version management
- **Error Quantification**: Uncertainty bounds and confidence intervals

#### Advanced Features
- **Curve Generation**: Time-dependent curves for compostability and sealing profiles
- **Environmental Scaling**: Property-specific environmental parameter effects
- **Multi-Property Prediction**: Simultaneous prediction of multiple properties

## 🧪 Additive Integration System

### Overview
The system includes a sophisticated additive/filler integration framework that allows for realistic modeling of how additives affect polymer blend properties. This system uses the UMM3 (Universal Material Modification Model) correction framework to apply both individual additive effects and pairwise additive-polymer interactions.

### Additive Configuration

#### 1. **Individual Additive Effects** (`train/simulation/config/ingredients.yaml`)
```yaml
ingredients:
  Glycerol:        # Additive name
    K_ts: -0.5     # Tensile strength effect (reduces strength)
    K_wvtr: 0.3    # WVTR effect (increases permeability)
    K_eab: 0.8     # Elongation effect (increases flexibility)
    K_cobb: 0.0    # Cobb effect (no effect)
    K_seal: -0.2   # Sealing effect (slightly reduces)
    K_compost: 0.1 # Compostability effect (slightly improves)
    K_otr: 0.2     # OTR effect (increases oxygen permeability)
    G: 0           # Geometry factor (0 for additives)
    type: "additive"
    description: "Glycerol plasticizer - increases flexibility, reduces strength"
    smiles: "C(C(CO)O)O"  # SMILES representation
```

#### 2. **Pairwise Compatibility** (`train/simulation/config/compatibility/`)
Property-specific compatibility files define how additives interact with different polymer families:
```yaml
# wvtr_compatibility.yaml
"Glycerol-PLA": {KI: 0.3, description: "Glycerol-PLA: Moderate compatibility, plasticizer effect increases WVTR"}
"Glycerol-PBAT": {KI: 0.2, description: "Glycerol-PBAT: Good compatibility, plasticizer effect increases WVTR"}
"Glycerol-PCL": {KI: 0.1, description: "Glycerol-PCL: Excellent compatibility, plasticizer effect increases WVTR"}
```

#### 3. **Material Dictionary** (`additives-fillers-dictionary.csv`)
```csv
Material,Grade,SMILES,Type
Glycerol,Glycerol,C(C(CO)O)O,additive
```

### UMM3 Correction System

#### Individual Corrections (K_prop)
- **Property-Specific Effects**: Each additive has K_prop values for all 7 properties
- **Transport Factors**: K_ts, K_wvtr, K_eab, K_cobb, K_seal, K_compost, K_otr
- **Realistic Ranges**: Values typically between -1.0 to +1.0 for realistic effects
- **Clipping Protection**: System automatically clips extreme values to prevent overflow

#### Pairwise Interactions (KI)
- **Additive-Polymer Compatibility**: KI values define how well additives interact with different polymer families
- **Property-Specific**: Different KI values for each property (WVTR, TS, EAB, etc.)
- **Compatibility Levels**: Excellent (KI: 0.1), Good (KI: 0.2), Moderate (KI: 0.3), Poor (KI: 0.5+)
- **Default Behavior**: Missing pairs default to KI: 0.0 with warning

### Simulation Integration

#### Additive Probability
- **Configurable Probability**: Default 30% of simulated blends include additives
- **Random Selection**: System randomly selects which blends get additives
- **Volume Fraction Sampling**: Additive concentrations sampled from realistic ranges

#### UMM3 Application
1. **Individual Corrections**: Apply K_prop values to base property values
2. **Pairwise Corrections**: Apply KI values for additive-polymer interactions
3. **Clipping**: Prevent mathematical overflow with extreme values
4. **Validation**: Ensure physically realistic results

### Streamlit Additive Explorer

#### Simple Additive App (`simple_additive_app.py`)
- **Interactive Interface**: Web-based exploration of additive effects
- **Real-time Prediction**: See immediate effects of adding Glycerol to blends
- **Property Comparison**: Compare predictions with and without additives
- **Visual Feedback**: Clear display of additive effects on all properties

#### Usage
```bash
# Launch the additive explorer
streamlit run simple_additive_app.py
```

### Training with Additives

#### Enhanced Dataset Generation
```bash
# Generate training data with additives enabled
python train/simulation/simulate.py --property wvtr --number 10000 --seed 42

# The system will:
# - Include additives in 30% of blends (default)
# - Apply UMM3 corrections for additive effects
# - Generate realistic additive-polymer interactions
```

#### Model Training
```bash
# Train models on additive-enhanced data
python train/training/train_unified_modular.py --property wvtr --input train/data/wvtr/polymerblends_for_ml_featurized.csv --output train/models/wvtr/vtest_additive/
```

### Configuration Examples

#### Realistic Additive Effects
```yaml
# Moderate plasticizer effects
Glycerol:
  K_ts: -0.3      # Slight strength reduction
  K_wvtr: 0.2     # Slight permeability increase
  K_eab: 0.5      # Moderate flexibility increase
  K_seal: -0.1    # Slight sealing reduction
```

#### Extreme Test Effects
```yaml
# For testing and demonstration
Glycerol:
  K_ts: -5.0      # Massive strength reduction
  K_wvtr: 5.0     # Massive permeability increase
  K_eab: 5.0      # Massive flexibility increase
```

## 🔧 Blend Optimization System

### Overview
The system includes an advanced blend optimization framework that automatically tunes polymer-polymer compatibility (KI) values to minimize prediction errors. This system uses gradient descent with blend-specific adaptive learning rates to optimize the entire KI vector simultaneously.

### Optimization Features

#### 1. **Dynamic KI Vector Generation**
- **Automatic Detection**: Identifies all polymer families present in validation data
- **Pair Generation**: Creates KI values for all unique polymer-polymer pairs
- **Validation-Based**: Only optimizes pairs that exist in validation blends
- **No Manual Configuration**: System automatically determines what to optimize

#### 2. **Blend-Specific Adaptive Learning Rate**
- **Personalized Learning**: Each blend gets its own learning rate based on MAE proximity to 0
- **Automatic Slowdown**: Learning rate decreases as blend approaches target
- **No Hard Limits**: KI values can go wherever needed to minimize MAE
- **Weighted Gradients**: Each KI parameter's gradient is weighted by blend learning rates

#### 3. **Gradient Descent Optimization**
- **Finite Differences**: Uses numerical differentiation to calculate gradients
- **Convergence Detection**: Stops when improvement becomes negligible
- **Stability Controls**: Gradient normalization and light clipping to prevent explosion
- **Iterative Improvement**: Continues until convergence or max iterations

### Usage Examples

#### Basic Optimization
```bash
# Optimize WVTR with adaptive learning rate
python simulate_optimize_blends.py --property wvtr --max-iterations 50 --base-lr 0.1 --adaptive-lr

# Optimize with fixed learning rate
python simulate_optimize_blends.py --property wvtr --max-iterations 50 --learning-rate 0.001 --no-adaptive-lr

# Optimize other properties
python simulate_optimize_blends.py --property ts --max-iterations 30 --base-lr 0.05 --adaptive-lr
python simulate_optimize_blends.py --property eab --max-iterations 30 --base-lr 0.05 --adaptive-lr
```

#### Advanced Configuration
```bash
# Custom iteration count and learning rate
python simulate_optimize_blends.py --property wvtr --max-iterations 100 --base-lr 0.2 --adaptive-lr

# Fixed learning rate with custom rate
python simulate_optimize_blends.py --property wvtr --max-iterations 50 --learning-rate 0.005 --no-adaptive-lr
```

### Optimization Process

#### 1. **Data Loading**
- Loads validation data from `train/data/{property}/validationblends.csv`
- Extracts polymer families and builds pair mapping
- Initializes KI vector (starts with existing YAML values or zeros)

#### 2. **Gradient Calculation**
- For each KI parameter, calculates finite difference gradient
- Tests small perturbation (ε = 1e-6) and measures MAE change
- Computes gradient = (MAE_plus - MAE_current) / ε

#### 3. **Learning Rate Adaptation**
- **Adaptive Mode**: Calculates blend-specific learning rates based on individual MAE
- **Fixed Mode**: Uses single learning rate for all parameters
- **Weighted Updates**: Combines gradients weighted by learning rates

#### 4. **KI Vector Update**
- Updates KI vector using gradient descent: `ki_new = ki_old - lr * gradient`
- Applies light clipping to prevent extreme values (-50 to +50)
- Stores results for analysis and YAML updates

### Results and Output

#### Optimization Results
- **CSV Output**: `blend_optimization_results_{property}.csv`
- **Iteration Tracking**: MAE, improvement, KI vector, learning rates per iteration
- **Convergence Info**: Final MAE, improvement, convergence status

#### YAML Updates
- **Automatic Updates**: Updates `{property}_compatibility.yaml` with optimized KI values
- **Backup Safety**: Original values preserved in git
- **Validation**: Ensures updated values are physically reasonable

### Performance Characteristics

#### Adaptive Learning Rate
- **Robust**: Less sensitive to learning rate choice
- **Personalized**: Each blend gets appropriate attention
- **Stable**: Consistent improvement across iterations
- **Efficient**: Automatically adjusts step size

#### Fixed Learning Rate
- **Simple**: Straightforward gradient descent
- **Tunable**: Requires careful learning rate selection
- **Fast**: Can converge quickly with right parameters
- **Sensitive**: Performance depends heavily on learning rate choice

### Integration with Existing System

#### Compatibility with UMM3
- **KI Overrides**: Optimized values take precedence over YAML defaults
- **Fallback System**: Falls back to YAML values for non-optimized pairs
- **Hybrid Approach**: Combines optimized and default values seamlessly

#### Model Training Integration
- **Pre-Training**: Run optimization before model training
- **Post-Training**: Run optimization after model training for fine-tuning
- **Iterative**: Can be run multiple times for continuous improvement

### Best Practices

#### When to Use Adaptive Learning Rate
- **New Properties**: When optimizing for the first time
- **Complex Blends**: When dealing with many polymer families
- **Robust Optimization**: When you want consistent results

#### When to Use Fixed Learning Rate
- **Fine-Tuning**: When you have a good starting point
- **Quick Tests**: When you want to test different parameters
- **Custom Control**: When you want precise control over step size

#### Optimization Strategy
1. **Start with Adaptive**: Use adaptive learning rate for initial optimization
2. **Analyze Results**: Check convergence and final MAE
3. **Fine-Tune**: Use fixed learning rate with smaller steps if needed
4. **Validate**: Test updated models on validation data
5. **Iterate**: Repeat if further improvement is needed

## 🧪 Additive Optimization System

### Overview
The system includes a dedicated additive optimization framework that tunes additive-polymer compatibility (KI) values to achieve specific target percentage changes in properties. This system uses representative blends and gradient descent to optimize additive effects across different polymer families.

### Additive Optimization Features

#### 1. **Target-Based Optimization**
- **CSV-Driven Targets**: Loads optimization targets from `train/simulation/targets.csv`
- **Percentage-Based**: Optimizes for specific percentage changes (e.g., +20% WVTR increase)
- **Polymer-Specific**: Different targets for different polymer families
- **Additive-Specific**: Different targets for different additives (e.g., Glycerol)

#### 2. **Representative Blend Generation**
- **Deterministic Blends**: Creates consistent representative blends for each polymer family
- **Volume Fraction Sampling**: Uses realistic volume fractions (0.3-0.7 for primary polymer)
- **Environmental Parameters**: Tests across realistic temperature/humidity/thickness ranges
- **Additive Integration**: Includes additive at realistic concentrations (0.1-0.3)

#### 3. **Gradient Descent Optimization**
- **Individual Gradients**: Each additive-polymer pair gets its own gradient
- **Adaptive Learning**: Learning rate adjusts based on individual target achievement
- **Tolerance-Based**: Stops when all targets are within specified tolerance
- **Convergence Detection**: Monitors individual and total errors

### Usage Examples

#### Basic Additive Optimization
```bash
# Optimize Glycerol effects on WVTR
python simulate_optimize_additives.py --property wvtr --tolerance 5.0 --max-iterations 50

# Optimize Glycerol effects on Tensile Strength
python simulate_optimize_additives.py --property ts --tolerance 10.0 --max-iterations 30

# Optimize with custom parameters
python simulate_optimize_additives.py --property eab --tolerance 3.0 --max-iterations 100 --learning-rate 0.05
```

#### Advanced Configuration
```bash
# High precision optimization
python simulate_optimize_additives.py --property wvtr --tolerance 1.0 --max-iterations 200 --learning-rate 0.01

# Quick optimization for testing
python simulate_optimize_additives.py --property otr --tolerance 15.0 --max-iterations 20 --learning-rate 0.2
```

### Target Configuration

#### Targets CSV Format (`train/simulation/targets.csv`)
```csv
Polymer Family,Additive,Target (average % increase on representative blends)
PLA,Glycerol,20.0
PBAT,Glycerol,15.0
PBS,Glycerol,25.0
PHAa,Glycerol,18.0
PHAs,Glycerol,22.0
PHBV,Glycerol,16.0
```

#### Target Interpretation
- **Positive Values**: Increase in property (e.g., +20% WVTR increase)
- **Negative Values**: Decrease in property (e.g., -10% Tensile Strength decrease)
- **Percentage-Based**: Targets are percentage changes, not absolute values
- **Representative Blends**: Targets are achieved on consistent test blends

### Optimization Process

#### 1. **Target Loading**
- Loads targets from `targets.csv`
- Validates target values and polymer-additive combinations
- Creates optimization vector for each target

#### 2. **Representative Blend Creation**
- Generates deterministic blends for each polymer family
- Includes additive at realistic concentrations
- Tests across environmental parameter ranges
- Ensures consistent baseline for optimization

#### 3. **Gradient Calculation**
- Calculates individual gradients for each additive-polymer pair
- Uses finite differences to measure KI sensitivity
- Tracks individual percentage changes vs targets
- Computes total error across all targets

#### 4. **KI Vector Update**
- Updates each KI value based on its individual gradient
- Uses adaptive learning rate based on target achievement
- Applies gradient clipping to prevent extreme values
- Monitors convergence and tolerance achievement

### Results and Output

#### Optimization Results
- **Console Output**: Real-time progress with individual target tracking
- **Convergence Info**: Final errors, target achievement, iteration count
- **KI Updates**: Shows KI value changes for each additive-polymer pair

#### YAML Updates
- **Automatic Updates**: Updates `{property}_compatibility.yaml` with optimized KI values
- **Additive-Polymer Pairs**: Updates specific additive-polymer compatibility values
- **Validation**: Ensures updated values are physically reasonable

### Integration with UMM3 System

#### KI Value Hierarchy
1. **Optimized Values**: Values from additive optimization take highest priority
2. **YAML Defaults**: Falls back to YAML values for non-optimized pairs
3. **System Defaults**: Uses 0.0 for completely missing pairs

#### UMM3 Application
- **Individual Corrections**: K_prop values from `ingredients.yaml`
- **Pairwise Corrections**: Optimized KI values from compatibility files
- **Combined Effects**: Both individual and pairwise effects applied together

### Performance Characteristics

#### Optimization Efficiency
- **Target-Driven**: Focuses on specific percentage changes
- **Representative Testing**: Uses consistent blends for reliable optimization
- **Individual Tracking**: Each additive-polymer pair optimized independently
- **Tolerance-Based**: Stops when targets are achieved within tolerance

#### Convergence Behavior
- **Fast Initial Progress**: Large improvements in early iterations
- **Fine-Tuning**: Slower progress as targets are approached
- **Tolerance Achievement**: Stops when all targets within specified tolerance
- **Robust**: Handles multiple targets simultaneously

### Best Practices

#### Target Setting
- **Realistic Targets**: Set achievable percentage changes based on literature
- **Property-Specific**: Different tolerances for different properties
- **Additive-Specific**: Consider additive type when setting targets
- **Validation**: Test targets on representative blends before optimization

#### Optimization Strategy
1. **Start Conservative**: Use higher tolerance (10-15%) for initial optimization
2. **Refine Targets**: Lower tolerance (3-5%) for fine-tuning
3. **Validate Results**: Test optimized values on validation blends
4. **Iterate**: Repeat optimization if targets not achieved

#### Integration Workflow
1. **Set Targets**: Define target percentage changes in `targets.csv`
2. **Run Optimization**: Execute additive optimization for each property
3. **Validate Results**: Test optimized KI values on validation data
4. **Update Models**: Retrain models with optimized additive effects
5. **Test Performance**: Evaluate model performance with new additive effects

## 📊 Data Generation Process

### Step 1: Experimental Data Collection
```
Real Measurements → Master Data Files → Material Database
```

### Step 2: Simulation and Augmentation
```bash
# Generate synthetic data for any property
python train/simulation/simulate.py --property wvtr --number 10000

# Generate data for all properties
python train/simulation/simulate.py --all --number 5000
```

### Step 3: Feature Extraction
```bash
# Convert raw data to ML-ready features
python train/run_blend_featurization.py input.csv output.csv
```

### Step 4: Model Training
```bash
# Train models using modular pipeline
python train/training/train_unified_modular.py --property wvtr --input featurized_data.csv --output models/wvtr/v1
```

### Step 5: Database Generation
```bash
# Generate comprehensive prediction database
python train/databasegeneration.py --max_materials 20 --random_samples 5
```

## 🧬 Molecular Feature System

### Feature Categories

#### 1. **Hybridization Features** (19 features)
- **SP Hybridization**: SP_C, SP_N
- **SP2 Hybridization**: SP2_C, SP2_N, SP2_O, SP2_S, SP2_B
- **SP3 Hybridization**: SP3_C, SP3_N, SP3_O, SP3_S, SP3_P, SP3_Si, SP3_B
- **Halogens**: SP3_F, SP3_Cl, SP3_Br, SP3_I
- **Special**: SP3D2_S

#### 2. **Ring System Features** (12 features)
- **Aromatic Rings**: Phenyls, thiophenes, aromatic rings with N/O/S
- **Aliphatic Rings**: Cyclohexanes, cyclopentanes, cyclopentenes
- **Heteroaromatic**: Aromatic rings with nitrogen, oxygen, sulfur
- **Mixed Systems**: Aromatic rings with both N and O

#### 3. **Functional Group Features** (25+ features)
- **Carbonyl Groups**: Carboxylic acids, esters, amides, ketones, aldehydes
- **Nitrogen Groups**: Amines (primary, secondary, tertiary, quaternary), imines, nitriles
- **Sulfur Groups**: Thiols, thioethers, sulfones, sulfoxides
- **Special Groups**: Imides, ureas, carbamates, azides, azo compounds

#### 4. **Structural Features** (6 features)
- **Branching**: Branching factor calculation
- **Depth**: Tree depth from terminal atoms
- **Chains**: Ethyl, propyl, butyl, long chain detection

### Feature Processing
- **SMILES Parsing**: Uses RDKit for molecular structure analysis
- **Order Invariance**: Consistent feature extraction regardless of input order
- **Missing Value Handling**: Robust handling of invalid SMILES
- **Volume Weighting**: Features weighted by polymer volume fractions in blends

## 🔬 Simulation Rules and Blending Models

### Property-Specific Blending Rules

#### WVTR (Water Vapor Transmission Rate)
- **Blending Rule**: Inverse rule of mixtures (1/Σ(ci/wi))
- **Environmental Effects**: Temperature (Arrhenius), humidity (power law), thickness (power law)
- **Scaling**: Dynamic thickness scaling based on polymer properties

#### Tensile Strength
- **Blending Rule**: Rule of mixtures with material type considerations
- **Dual Properties**: Machine direction (MD) and transverse direction (TD)
- **Environmental Effects**: Thickness-dependent scaling

#### Elongation at Break
- **Blending Rule**: Volume-weighted average with non-linear corrections
- **Dual Properties**: EAB1 and EAB2 predictions
- **Material Compatibility**: Immiscibility rules for certain polymer combinations

#### Cobb Value
- **Blending Rule**: Simple rule of mixtures
- **Environmental Independence**: No temperature/humidity effects
- **Material Specific**: Intrinsic property of polymer composition

#### OTR (Oxygen Transmission Rate)
- **Blending Rule**: Similar to WVTR with oxygen-specific parameters
- **Environmental Effects**: Temperature and humidity scaling
- **Thickness Scaling**: Power law relationship

#### Seal Strength
- **Blending Rule**: Volume-weighted average of individual polymer strengths
- **Temperature Calculation**: Rule of mixtures for melting temperatures
- **Curve Generation**: Cubic polynomial interpolation for temperature profiles

#### Home Compostability
- **Dual Properties**: Maximum disintegration (max_L) and time to 50% (t0)
- **Curve Generation**: Sigmoid curves for disintegration, quintic curves for biodegradation
- **Environmental Effects**: Thickness-dependent degradation kinetics

## 🎯 Model Training and Validation

### Training Pipeline

#### 1. **Data Preparation**
- **Log Transformations**: Property-specific log scaling with offsets
- **Zero Handling**: Removal of zero targets for certain properties
- **Missing Value Treatment**: NaN handling for WVTR/OTR properties
- **Feature Exclusion**: Automatic removal of SMILES and metadata columns

#### 2. **Data Splitting Strategy**
- **Last N Training**: Specified blends always go to training set
- **Last N Testing**: Can override last N training for validation
- **Automatic 80/20 Split**: For remaining data
- **Temporal Validation**: Ensures realistic model evaluation

#### 3. **Model Configuration**
- **Algorithm**: XGBoost with property-specific hyperparameters
- **Preprocessing**: Categorical encoding and feature scaling
- **Oversampling**: Configurable oversampling for rare cases
- **Dual Property Support**: Separate models for multi-target properties

#### 4. **Validation and Testing**
- **Cross-Validation**: K-fold validation for robust performance assessment
- **Last N Analysis**: Special validation on most recent blends
- **Error Quantification**: Model error bounds and uncertainty estimates
- **Performance Metrics**: MAE, RMSE, R² for each property

### Model Versions and Performance

#### Current Model Versions
- **WVTR**: v4 (XGBoost)
- **Tensile Strength**: v4 (XGBoost, dual property)
- **Elongation at Break**: v4 (XGBoost, dual property)
- **Cobb Value**: v2 (XGBoost)
- **OTR**: v2 (XGBoost)
- **Seal Strength**: v2 (XGBoost)
- **Home Compostability**: v5 (XGBoost, dual property)

#### Performance Characteristics
- **High Accuracy**: R² > 0.8 for most properties
- **Robust Validation**: Consistent performance across different blend types
- **Error Bounds**: Quantified uncertainty for all predictions
- **Scalability**: Handles up to 5-component blends efficiently

## 🚀 Usage Examples

### Command Line Interface

#### Single Property Prediction
```bash
# WVTR prediction
python predict_blend_properties.py wvtr "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5" 25 60 100

# Tensile strength prediction
python predict_blend_properties.py ts "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5" 100

# All properties prediction
python predict_blend_properties.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5" 25 60 100
```

#### Web Interface
```bash
# Launch main Streamlit web app
streamlit run predict_blend_properties_app.py

# Launch additive explorer app
streamlit run simple_additive_app.py
```

### Python API

#### Basic Property Prediction
```python
from train.modules.prediction_engine import predict_blend_property

# WVTR prediction for PLA/PBAT blend
result = predict_blend_property(
    property_type='wvtr',
    polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
    available_env_params={'Temperature (C)': 25, 'RH (%)': 60, 'Thickness (um)': 100},
    material_dict=material_dict
)

if result['success']:
    print(f"WVTR: {result['prediction']:.2f} {result['unit']}")
```

#### Multi-Property Prediction
```python
# Predict all properties simultaneously
properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']
results = []

for prop in properties:
    result = predict_blend_property(prop, polymers, env_params, material_dict)
    if result:
        results.append(result)

# Display results
for result in results:
    print(f"{result['name']}: {result['prediction']:.2f} {result['unit']}")
```

#### Additive-Enhanced Prediction
```python
# Predict properties with additives
polymers_with_additive = [
    ("PLA", "4032D", 0.4),
    ("PBAT", "Ecoworld", 0.4),
    ("Glycerol", "Glycerol", 0.2)  # 20% Glycerol additive
]

# Predict with additive effects
result = predict_blend_property(
    property_type='wvtr',
    polymers=polymers_with_additive,
    available_env_params={'Temperature (C)': 25, 'RH (%)': 60, 'Thickness (um)': 100},
    material_dict=material_dict
)

if result['success']:
    print(f"WVTR with Glycerol: {result['prediction']:.2f} {result['unit']}")
    print(f"Additive effects applied via UMM3 corrections")
```

## 📁 Project Structure

```
Polymer-Blends-Model/
├── predict_blend_properties_app.py      # Main Streamlit application
├── simple_additive_app.py               # Additive explorer application
├── predict_blend_properties.py          # Command-line interface
├── material-smiles-dictionary.csv       # Polymer database
├── additives-fillers-dictionary.csv    # Additives and fillers database
├── train/                               # Training and simulation modules
│   ├── simulation/                      # Data augmentation system
│   │   ├── simulate.py                 # Main simulation script
│   │   ├── simulation_common.py        # Common simulation functions
│   │   ├── simulation_rules.py         # Property-specific rules
│   │   ├── umm3_correction.py          # UMM3 correction system
│   │   ├── config/                     # Configuration files
│   │   │   ├── ingredients.yaml        # Additive K_prop parameters
│   │   │   ├── ingredient_polymer.yaml # Additive-polymer compatibility
│   │   │   └── compatibility/          # Property-specific compatibility
│   │   │       ├── wvtr_compatibility.yaml
│   │   │       ├── ts_compatibility.yaml
│   │   │       ├── eab_compatibility.yaml
│   │   │       ├── cobb_compatibility.yaml
│   │   │       ├── otr_compatibility.yaml
│   │   │       ├── seal_compatibility.yaml
│   │   │       └── compost_compatibility.yaml
│   │   └── rules/                      # Individual property rules
│   ├── modules/                         # Core prediction modules
│   │   ├── feature_extractor.py        # Molecular feature extraction
│   │   ├── blend_feature_extractor.py  # Blend processing
│   │   ├── prediction_engine.py        # Model prediction
│   │   ├── prediction_utils.py         # Utility functions
│   │   └── error_calculator.py         # Uncertainty quantification
│   ├── training/                        # Model training pipeline
│   │   ├── train_unified_modular.py    # Main training script
│   │   └── training_modules/           # Modular training components
│   ├── data/                           # Training and validation data
│   │   ├── wvtr/                       # WVTR datasets
│   │   ├── ts/                         # Tensile strength datasets
│   │   ├── eab/                        # Elongation datasets
│   │   ├── cobb/                       # Cobb value datasets
│   │   ├── otr/                        # OTR datasets
│   │   ├── seal/                       # Seal strength datasets
│   │   └── eol/                        # Compostability datasets
│   ├── models/                         # Trained ML models
│   │   ├── wvtr/v4/                   # WVTR models
│   │   ├── ts/v4/                     # Tensile strength models
│   │   ├── eab/v4/                    # Elongation models
│   │   ├── cobb/v2/                   # Cobb value models
│   │   ├── otr/v2/                    # OTR models
│   │   ├── seal/v2/                   # Seal strength models
│   │   └── eol/v5/                    # Compostability models
│   └── databasegeneration.py           # Database generation script
├── train/modules_home/                 # Compostability curve generation
│   ├── curve_generator.py             # Curve generation functions
│   └── utils.py                       # Utility functions
├── train/modules_sealing/              # Sealing profile generation
│   ├── curve_generator.py             # Sealing curve generation
│   └── utils.py                       # Utility functions
├── validation/                         # Model validation results
└── test_results/                       # Prediction outputs
```

## 🔧 Installation and Setup

### Dependencies
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn rdkit scipy joblib xgboost
```

### Required Files
- `material-smiles-dictionary.csv`: Polymer database with SMILES representations
- `modelrrors.csv`: Model error data for uncertainty quantification
- Trained model files in `train/models/` directories

### Quick Start
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch main web app**: `streamlit run predict_blend_properties_app.py`
3. **Launch additive explorer**: `streamlit run simple_additive_app.py`
4. **Command line**: `python predict_blend_properties.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"`

## 🧪 Data Generation and Training

### Complete Training Pipeline

#### 1. Generate Synthetic Data
```bash
# Generate 10,000 samples for WVTR
python train/simulation/simulate.py --property wvtr --number 10000

# Generate data for all properties
python train/simulation/simulate.py --all --number 5000
```

#### 2. Featurize Data
```bash
# Convert to ML-ready features
python train/run_blend_featurization.py train/data/wvtr/polymerblends_for_ml.csv train/data/wvtr/polymerblends_for_ml_featurized.csv
```

#### 3. Train Models
```bash
# Train WVTR model
python train/training/train_unified_modular.py --property wvtr --input train/data/wvtr/polymerblends_for_ml_featurized.csv --output train/models/wvtr/v1

# Train with custom parameters
python train/training/train_unified_modular.py --property ts --input train/data/ts/polymerblends_for_ml_featurized.csv --output train/models/ts/v1 --last_n_training 10 --last_n_testing 3
```

#### 4. Generate Prediction Database
```bash
# Generate comprehensive database
python train/databasegeneration.py --max_materials 20 --random_samples 5 --n_polymers 2 3 4
```

## 📈 Advanced Features

### Curve Generation

#### Compostability Curves
- **Disintegration Curves**: Sigmoid functions for disintegration over time
- **Biodegradation Curves**: Quintic functions for biodegradation kinetics
- **Time Profiles**: Day-by-day predictions up to 400 days
- **Thickness Scaling**: Material thickness effects on degradation

#### Sealing Profile Curves
- **Temperature Profiles**: Sealing strength vs temperature curves
- **Cubic Interpolation**: Smooth curves through boundary points
- **Boundary Points**: Initial sealing, polymer max, blend predicted, degradation
- **Material Properties**: Melt temperature and degradation temperature effects

### Error Quantification
- **Model Errors**: Property-specific error bounds
- **Experimental Uncertainty**: Standard deviations from experimental data
- **Confidence Intervals**: Upper and lower bounds for predictions
- **Uncertainty Propagation**: Error bounds for derived quantities

### Environmental Effects
- **Temperature Scaling**: Arrhenius and power law relationships
- **Humidity Effects**: Moisture-dependent property changes
- **Thickness Scaling**: Dynamic and fixed thickness scaling models
- **Material Interactions**: Polymer-specific environmental responses

## 🔬 Scientific Methodology

### Data Quality Assurance
- **Experimental Validation**: All models trained on real experimental data
- **Cross-Validation**: Robust validation across different blend types
- **Error Tracking**: Comprehensive error analysis and reporting
- **Reproducibility**: Fixed random seeds and version control

### Physical Realism
- **Material Compatibility**: Immiscibility rules based on polymer chemistry
- **Environmental Physics**: Realistic temperature and humidity effects
- **Blending Rules**: Physically meaningful mathematical models
- **Constraint Validation**: Volume fraction and material property constraints

### Model Interpretability
- **Feature Importance**: Detailed analysis of molecular descriptor contributions
- **Blending Rules**: Transparent mathematical models for property prediction
- **Error Analysis**: Clear understanding of model limitations
- **Validation Results**: Comprehensive performance metrics and visualizations

## 📊 Performance and Validation

### Model Performance
- **High Accuracy**: R² > 0.8 for most properties
- **Robust Validation**: Consistent performance across blend types
- **Error Quantification**: Bounded uncertainty for all predictions
- **Scalability**: Efficient prediction for complex multi-component blends

### Validation Methodology
- **Temporal Validation**: Last N blends strategy for realistic evaluation
- **Cross-Property Validation**: Consistent performance across all properties
- **Environmental Validation**: Performance across temperature/humidity ranges
- **Material Validation**: Performance across different polymer types

## 🤝 Contributing

### Adding New Materials
1. Add SMILES representation to `material-smiles-dictionary.csv`
2. Validate SMILES syntax with RDKit
3. Test predictions with the new material
4. Update documentation

### Adding New Properties
1. Create property-specific simulation rules
2. Add experimental data to `train/data/`
3. Configure training parameters
4. Train and validate models
5. Update prediction engine

### Code Structure
- **Modular Design**: Clear separation of concerns
- **Version Control**: Model versioning and management
- **Documentation**: Comprehensive inline documentation
- **Testing**: Validation and error handling throughout

## 📄 License

This project is for research and development purposes. Please ensure proper attribution when using the models and methodology.

## 🙏 Acknowledgments

- **RDKit**: Molecular structure analysis and SMILES processing
- **XGBoost**: Gradient boosting machine learning models
- **Streamlit**: Web application framework
- **Experimental Data**: Real polymer property measurements
- **Scientific Community**: Polymer science and materials engineering research

---

*This comprehensive system represents a significant advancement in polymer blend property prediction, combining experimental data with computational simulation to create robust, accurate, and interpretable models for materials science applications.*
