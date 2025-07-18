#!/usr/bin/env python3
"""
WVTR Blend Prediction Script
Takes polymer names, grades, volume fractions, temperature, RH, and thickness as command line inputs
and predicts WVTR using the trained v1 model.

Usage: python predict_wvtr_blend.py "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5, 38, 90, 100"

Format: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Temperature, RH, Thickness"
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
import argparse
import tempfile
import logging
from modules.blend_feature_extractor import process_blend_features
from modules.feature_extractor import FeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_material_dictionary(dict_path='material-smiles-dictionary.csv'):
    """Load the material-SMILES dictionary."""
    try:
        df = pd.read_csv(dict_path)
        # Create a dictionary with (Material, Grade) as key and SMILES as value
        material_dict = {}
        for _, row in df.iterrows():
            key = (row['Material'].strip(), row['Grade'].strip())
            material_dict[key] = row['SMILES'].strip()
        
        logger.info(f"✅ Material dictionary loaded with {len(material_dict)} entries")
        return material_dict
    except Exception as e:
        logger.error(f"❌ Error loading material dictionary: {e}")
        return None

def parse_command_line_input(input_string):
    """
    Parse the command line input string.
    
    Expected format: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Temperature, RH, Thickness"
    
    Returns:
    - list of tuples: [(Material, Grade, vol_fraction), ...]
    - temperature, rh, thickness
    """
    try:
        # Split by commas and strip whitespace
        parts = [part.strip() for part in input_string.split(',')]
        
        if len(parts) < 6:  # Minimum: 2 polymers (6 parts) + 3 environmental
            raise ValueError("Input must have at least 2 polymers and 3 environmental parameters")
        
        # Extract environmental parameters (last 3 values)
        temperature = float(parts[-3])
        rh = float(parts[-2])
        thickness = float(parts[-1])
        
        # Extract polymer information (everything except last 3)
        polymer_parts = parts[:-3]
        
        if len(polymer_parts) % 3 != 0:
            raise ValueError("Polymer information must be in groups of 3: Material, Grade, Volume_Fraction")
        
        polymers = []
        for i in range(0, len(polymer_parts), 3):
            material = polymer_parts[i]
            grade = polymer_parts[i + 1]
            vol_fraction = float(polymer_parts[i + 2])
            polymers.append((material, grade, vol_fraction))
        
        # Validate volume fractions sum to 1.0
        total_fraction = sum(vol_fraction for _, _, vol_fraction in polymers)
        if not np.isclose(total_fraction, 1.0, atol=1e-5):
            raise ValueError(f"Volume fractions must sum to 1.0, got {total_fraction}")
        
        logger.info(f"✅ Parsed {len(polymers)} polymers with total volume fraction: {total_fraction}")
        logger.info(f"✅ Environmental parameters: T={temperature}°C, RH={rh}%, Thickness={thickness}μm")
        
        return polymers, temperature, rh, thickness
        
    except Exception as e:
        logger.error(f"❌ Error parsing input: {e}")
        return None, None, None, None

def convert_polymers_to_smiles(polymers, material_dict):
    """
    Convert polymer names and grades to SMILES using the material dictionary.
    
    Args:
        polymers: list of tuples (Material, Grade, vol_fraction)
        material_dict: dictionary mapping (Material, Grade) to SMILES
    
    Returns:
        list of tuples (SMILES, vol_fraction) or None if error
    """
    try:
        smiles_list = []
        for material, grade, vol_fraction in polymers:
            key = (material, grade)
            if key in material_dict:
                smiles = material_dict[key]
                smiles_list.append((smiles, vol_fraction))
                logger.info(f"✅ {material} {grade} -> {smiles} (vol_fraction: {vol_fraction})")
            else:
                logger.error(f"❌ Material/Grade combination not found: {material} {grade}")
                logger.info(f"Available combinations: {list(material_dict.keys())}")
                return None
        
        return smiles_list
        
    except Exception as e:
        logger.error(f"❌ Error converting polymers to SMILES: {e}")
        return None

def create_input_dataframe(smiles_list, temperature, rh, thickness, polymers):
    """
    Create a DataFrame in the exact format expected by the blend feature extractor.
    
    Args:
        smiles_list: list of tuples (SMILES, vol_fraction)
        temperature: temperature in Celsius
        rh: relative humidity in percent
        thickness: thickness in micrometers
        polymers: original polymer list for metadata
    
    Returns:
        DataFrame ready for featurization
    """
    try:
        # Create the basic structure
        data = {
            'Materials': ', '.join([f"{mat} {grade}" for mat, grade, _ in polymers]),
            'Polymer Grade 1': polymers[0][1] if len(polymers) > 0 else 'Unknown',
            'Polymer Grade 2': polymers[1][1] if len(polymers) > 1 else 'Unknown',
            'Polymer Grade 3': polymers[2][1] if len(polymers) > 2 else 'Unknown',
            'Polymer Grade 4': polymers[3][1] if len(polymers) > 3 else 'Unknown',
            'Polymer Grade 5': polymers[4][1] if len(polymers) > 4 else 'Unknown',
            'SMILES1': smiles_list[0][0] if len(smiles_list) > 0 else '',
            'SMILES2': smiles_list[1][0] if len(smiles_list) > 1 else '',
            'SMILES3': smiles_list[2][0] if len(smiles_list) > 2 else '',
            'SMILES4': smiles_list[3][0] if len(smiles_list) > 3 else '',
            'SMILES5': smiles_list[4][0] if len(smiles_list) > 4 else '',
            'vol_fraction1': smiles_list[0][1] if len(smiles_list) > 0 else 0.0,
            'vol_fraction2': smiles_list[1][1] if len(smiles_list) > 1 else 0.0,
            'vol_fraction3': smiles_list[2][1] if len(smiles_list) > 2 else 0.0,
            'vol_fraction4': smiles_list[3][1] if len(smiles_list) > 3 else 0.0,
            'vol_fraction5': smiles_list[4][1] if len(smiles_list) > 4 else 0.0,
            'Temperature (C)': temperature,
            'RH (%)': rh,
            'Thickness (um)': thickness
        }
        
        df = pd.DataFrame([data])
        logger.info(f"✅ Created input DataFrame with shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error creating input DataFrame: {e}")
        return None

def load_wvtr_model(model_path='models/wvtr/v1/comprehensive_polymer_model.pkl'):
    """Load the trained WVTR model."""
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ WVTR model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Error loading WVTR model: {e}")
        return None

def prepare_features_for_prediction(featurized_df, model):
    """
    Prepare featurized data exactly as expected by the trained WVTR model.
    
    Args:
        featurized_df: DataFrame with featurized blend data
        model: loaded trained model
    
    Returns:
        prepared DataFrame ready for prediction
    """
    try:
        logger.info(f"Preparing features for prediction...")
        logger.info(f"Input shape: {featurized_df.shape}")
        logger.info(f"Input columns: {list(featurized_df.columns)}")
        
        # Separate features and target (same logic as training script)
        target = 'property'
        smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
        # EXCLUDE Materials column from features (same as training script)
        excluded_cols = [target] + smiles_cols + ['Materials']
        
        # Remove excluded columns if they exist
        X = featurized_df.drop(columns=[col for col in excluded_cols if col in featurized_df.columns])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Excluded columns: {excluded_cols}")
        logger.info(f"Feature columns: {list(X.columns)}")
        
        # Identify categorical and numerical features (same logic as training script)
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
        logger.info(f"Numerical features ({len(numerical_features)}): {numerical_features}")
        
        # Handle missing values (same logic as training script)
        logger.info("Handling missing values...")
        for col in categorical_features:
            X[col] = X[col].fillna('Unknown')
        for col in numerical_features:
            X[col] = X[col].fillna(0)
        
        return X
        
    except Exception as e:
        logger.error(f"❌ Error preparing features: {e}")
        return None

def predict_wvtr(features_df, model):
    """
    Make WVTR predictions using the trained model.
    
    Args:
        features_df: prepared DataFrame
        model: loaded trained model
    
    Returns:
        prediction in log scale
    """
    try:
        # Make prediction (log scale)
        prediction_log = model.predict(features_df)[0]
        logger.info(f"✅ WVTR prediction made successfully")
        return prediction_log
        
    except Exception as e:
        logger.error(f"❌ Error making WVTR prediction: {e}")
        return None

def main():
    """Main function to run the WVTR blend prediction."""
    parser = argparse.ArgumentParser(description='Predict WVTR for polymer blends')
    parser.add_argument('input', type=str, 
                       help='Input string: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Temperature, RH, Thickness"')
    parser.add_argument('--model', type=str, default='models/wvtr/v1/comprehensive_polymer_model.pkl',
                       help='Path to WVTR model (default: models/wvtr/v1/comprehensive_polymer_model.pkl)')
    parser.add_argument('--dict', type=str, default='material-smiles-dictionary.csv',
                       help='Path to material dictionary (default: material-smiles-dictionary.csv)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("=== WVTR BLEND PREDICTION SCRIPT ===")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Dictionary: {args.dict}")
    
    # Load material dictionary
    print("\n=== LOADING MATERIAL DICTIONARY ===")
    material_dict = load_material_dictionary(args.dict)
    if material_dict is None:
        return
    
    # Parse command line input
    print("\n=== PARSING INPUT ===")
    polymers, temperature, rh, thickness = parse_command_line_input(args.input)
    if polymers is None:
        return
    
    # Convert polymers to SMILES
    print("\n=== CONVERTING POLYMERS TO SMILES ===")
    smiles_list = convert_polymers_to_smiles(polymers, material_dict)
    if smiles_list is None:
        return
    
    # Create input DataFrame
    print("\n=== CREATING INPUT DATAFRAME ===")
    input_df = create_input_dataframe(smiles_list, temperature, rh, thickness, polymers)
    if input_df is None:
        return
    
    # Save input to temporary file for featurization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        input_df.to_csv(f.name, index=False)
        temp_input_file = f.name
    
    try:
        # Featurize the blend
        print("\n=== FEATURIZING BLEND ===")
        temp_output_file = temp_input_file.replace('.csv', '_featurized.csv')
        featurized_df = process_blend_features(temp_input_file, temp_output_file)
        
        if featurized_df is None or len(featurized_df) == 0:
            logger.error("❌ Featurization failed or produced empty result")
            return
        
        logger.info(f"✅ Blend featurized successfully with shape: {featurized_df.shape}")
        
        # Load WVTR model
        print("\n=== LOADING WVTR MODEL ===")
        model = load_wvtr_model(args.model)
        if model is None:
            return
        
        # Prepare features for prediction
        print("\n=== PREPARING FEATURES FOR PREDICTION ===")
        features_df = prepare_features_for_prediction(featurized_df, model)
        if features_df is None:
            return
        
        # Make prediction
        print("\n=== MAKING WVTR PREDICTION ===")
        prediction_log = predict_wvtr(features_df, model)
        if prediction_log is None:
            return
        
        # Convert to original scale
        prediction_original = np.exp(prediction_log)
        
        # Print results
        print("\n=== WVTR PREDICTION RESULTS ===")
        print(f"Log-scale prediction: {prediction_log:.4f}")
        print(f"Original-scale prediction: {prediction_original:.2f} g/m²/day")
        
        # Print input summary
        print(f"\n=== INPUT SUMMARY ===")
        print(f"Number of polymers: {len(polymers)}")
        for i, (material, grade, vol_fraction) in enumerate(polymers, 1):
            print(f"  Polymer {i}: {material} {grade} ({vol_fraction:.2f})")
        print(f"Temperature: {temperature}°C")
        print(f"Relative humidity: {rh}%")
        print(f"Thickness: {thickness} μm")
        
        # Print feature summary
        print(f"\n=== FEATURE SUMMARY ===")
        print(f"Total features: {len(features_df.columns)}")
        print(f"Environmental features: Temperature, RH, Thickness")
        print(f"Molecular features: {len(features_df.columns) - 3} descriptors")
        
        return prediction_original
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input_file)
            if os.path.exists(temp_input_file.replace('.csv', '_featurized.csv')):
                os.unlink(temp_input_file.replace('.csv', '_featurized.csv'))
        except:
            pass

if __name__ == "__main__":
    main() 