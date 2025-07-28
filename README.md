# Polymer Blend Prediction Model

A Streamlit web application for predicting biodegradation and disintegration properties of polymer blends using machine learning.

## Features

- **Blend Prediction**: Input polymer compositions and get predictions for max_L (disintegration level) and t0 (time to 50%)
- **Sigmoid Curve Generation**: Visualize disintegration and biodegradation curves
- **PLA Rule**: Special rule for PLA + compostable polymer blends
- **Multiple Models**: Support for different trained models

## Usage

1. Select polymers from the dropdown menus
2. Set volume fractions (must sum to 1.0)
3. Choose model directory
4. Click "Generate Prediction"
5. View results and download curves

## Technical Details

- Uses Differentiable Label Optimization (DLO) for predictions
- Molecular features extracted from SMILES strings
- Volume fraction-based blend calculations
- Special rules for known polymer combinations

## Files

- `predict_blend_cli_old_majorityrule.py`: Main Streamlit app
- `modules/`: Core prediction modules
- `models/v1/`: Trained model files
- `polymer_properties_reference.csv`: Polymer database
- `data/`: Training data 