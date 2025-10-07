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
