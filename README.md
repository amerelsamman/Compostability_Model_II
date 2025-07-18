# Polymer Blend Property Prediction Model

A comprehensive machine learning system for predicting polymer blend properties including Water Vapor Transmission Rate (WVTR), Tensile Strength (TS), Elongation at Break (EAB), Cobb Value, and Home Compostability.

## Overview

This system uses molecular descriptors extracted from SMILES (Simplified Molecular Input Line Entry System) representations of polymers to predict blend properties. The model architecture combines individual polymer features weighted by volume fractions to predict blend behavior under various environmental conditions.

## Architecture

### Core Components

#### 1. **Main Interface** (`polymer_blend_predictor.py`)
- Primary API for property prediction
- Handles input validation and error management
- Supports single property and multi-property predictions
- Provides material discovery and property information functions

#### 2. **Feature Extraction** (`modules/feature_extractor.py`)
- Extracts 80+ molecular descriptors from SMILES strings
- Features include:
  - **Hybridization states**: SP, SP2, SP3 for different elements (C, N, O, S, etc.)
  - **Ring systems**: Phenyls, cyclohexanes, cyclopentanes, thiophenes, and heteroaromatic rings
  - **Functional groups**: 25+ chemical groups (esters, amides, alcohols, etc.)
  - **Structural features**: Branching factor, tree depth, chain lengths
- Uses RDKit for molecular structure analysis

#### 3. **Blend Processing** (`modules/blend_feature_extractor.py`)
- Combines individual polymer features using volume fraction weighting
- Supports up to 5-component blends
- Validates volume fraction constraints (must sum to 1.0)
- Handles single polymers and complex blends

#### 4. **Prediction Engine** (`modules/prediction_engine.py`)
- Loads trained machine learning models
- Processes environmental parameters (temperature, humidity, thickness)
- Returns predictions with optional error bounds

#### 5. **Home Compostability Model** (`homecompost_modules/`)
- Specialized model for predicting home compostability

### Supported Properties

| Property | Unit | Required Parameters | Model Version |
|----------|------|-------------------|---------------|
| WVTR | g/m²/day | Temperature, RH, Thickness | v2 |
| Tensile Strength | MPa | Thickness | v2 |
| Elongation at Break | % | Thickness | v1 |
| Cobb Value | g/m² | None | v2 |
| Home Compostability | % disintegration | Thickness | Custom |

## Data Structure

### Material Dictionary (`material-smiles-dictionary.csv`)
Contains polymer materials with their SMILES representations:
```csv
Material,Grade,SMILES
PLA,Ingeo 4032D,*OC(=O)C(*)C
PBAT,Ecoworld,*OCCCCOC(=O)CCCCC(=O)OCCCCC(=O)c1ccc(C(*)=O)cc1
```

### Supported Materials
- **Biopolymers**: PLA, PBAT, PHA, PHB, PBS, PCL, PGA
- **Petroleum-based**: LDPE, PP, PET, PVDC, PA, EVOH
- **Others**: Starch-based materials

## Usage Examples

### Basic Property Prediction

```python
from polymer_blend_predictor import predict_property

# WVTR prediction for PLA/PBAT blend
result = predict_property(
    polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
    property_name="wvtr",
    temperature=25,
    rh=60,
    thickness=100
)

if result['success']:
    print(f"WVTR: {result['prediction']:.2f} {result['unit']}")
```

### Multi-Property Prediction

```python
from polymer_blend_predictor import predict_all_properties

# Predict all properties
results = predict_all_properties(
    polymers=[("PLA", "4032D", 0.6), ("PBAT", "Ecoworld", 0.4)],
    temperature=23,
    rh=50,
    thickness=75
)

for prop_name, result in results.items():
    if result['success']:
        print(f"{result['property_name']}: {result['prediction']:.2f} {result['unit']}")
```

### Single Polymer Prediction

```python
# Cobb value for single polymer
result = predict_property(
    polymers=[("PLA", "4032D", 1.0)],
    property_name="cobb"
)
```

### Three-Component Blend

```python
# Complex blend prediction
result = predict_property(
    polymers=[
        ("PLA", "4032D", 0.5),
        ("PBAT", "Ecoworld", 0.3),
        ("PCL", "Capa 6500", 0.2)
    ],
    property_name="eab",
    thickness=75
)
```

## Model Training and Validation

### Feature Engineering Process

1. **Molecular Descriptor Extraction**: Each polymer's SMILES is converted to 80+ molecular features
2. **Blend Feature Calculation**: Features are weighted by volume fractions and combined
3. **Environmental Parameter Integration**: Temperature, humidity, and thickness are added as features
4. **Model Training**: Gradient boosting models are trained on experimental data

### Model Performance

Models are trained on experimental data with cross-validation and feature importance analysis. Each property has its own specialized model optimized for that specific prediction task.

## Environmental Dependencies

### WVTR (Water Vapor Transmission Rate)
- **Temperature**: Affects molecular mobility and diffusion
- **Relative Humidity**: Drives moisture gradient across film
- **Thickness**: Directly proportional to resistance

### Mechanical Properties (TS, EAB)
- **Thickness**: Affects stress distribution and failure modes
- Temperature and humidity effects are material-dependent

### Cobb Value
- **No environmental parameters**: Intrinsic material property
- Measures water absorption capacity

### Home Compostability
- **Thickness**: Affects degradation kinetics
- Uses sigmoid model with thickness scaling
- Incorporates certification data and synergistic effects

## Error Handling

The system includes comprehensive error handling for:
- Invalid material/grade combinations
- Volume fractions that don't sum to 1.0
- Missing required environmental parameters
- SMILES parsing errors
- Model loading failures

## File Structure

```
Polymer-Blends-Model/
├── polymer_blend_predictor.py      # Main API
├── modules/                        # Core functionality
│   ├── feature_extractor.py       # Molecular feature extraction
│   ├── blend_feature_extractor.py # Blend processing
│   ├── prediction_engine.py       # Model prediction
│   ├── prediction_utils.py        # Utilities and configs
│   ├── input_parser.py           # Input validation
│   └── output_formatter.py       # Result formatting
├── homecompost_modules/           # Compostability model
│   ├── core_model.py             # Compostability logic
│   ├── blend_generator.py        # Blend curve generation
│   └── plotting.py               # Visualization
├── models/                        # Trained ML models
│   ├── wvtr/v2/                  # WVTR model
│   ├── ts/v2/                    # Tensile strength model
│   ├── eab/v1/                   # Elongation model
│   └── cobb/v2/                  # Cobb value model
├── material-smiles-dictionary.csv # Polymer database
├── usage_examples.py             # Usage examples
└── README.md                     # This file
```

## Dependencies

- **RDKit**: Molecular structure analysis
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning models
- **joblib**: Model serialization
- **scipy**: Scientific computing (for compostability model)

## Installation

1. Install required packages:
```bash
pip install rdkit-python pandas numpy scikit-learn joblib scipy
```

2. Ensure the material dictionary and trained models are in place
3. Run usage examples to verify installation

## Contributing

When adding new materials:
1. Add SMILES representation to `material-smiles-dictionary.csv`
2. Validate SMILES syntax with RDKit
3. Test predictions with the new material

## License

This project is for research and development purposes. Please ensure proper attribution when using the models and methodology.
