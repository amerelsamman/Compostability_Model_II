#!/usr/bin/env python3
"""
Prediction utility functions for polymer blend property prediction.
"""

import os
import pandas as pd
import numpy as np
import joblib
import logging
import tempfile
from modules.blend_feature_extractor import process_blend_features

# Set up logging
logger = logging.getLogger(__name__)

# Property configurations
PROPERTY_CONFIGS = {
    'wvtr': {
        'name': 'WVTR',
        'unit': 'g/m²/day',
        'model_path': 'models/wvtr/v3/comprehensive_polymer_model.pkl',
        'env_params': ['Temperature (C)', 'RH (%)', 'Thickness (um)'],
        'min_parts': 6,  # 2 polymers (6 parts) + 3 environmental
        'log_scale': True
    },
    'ts': {
        'name': 'Tensile Strength',
        'unit': 'MPa',
        'model_path': 'models/ts/v3/comprehensive_polymer_model.pkl',
        'env_params': ['Thickness (um)'],
        'min_parts': 4,  # 2 polymers (6 parts) + 1 environmental
        'log_scale': True
    },
    'eab': {
        'name': 'Elongation at Break',
        'unit': '%',
        'model_path': 'models/eab/v3/comprehensive_polymer_model.pkl',
        'env_params': ['Thickness (um)'],
        'min_parts': 4,  # 2 polymers (6 parts) + 1 environmental
        'log_scale': True
    },
    'cobb': {
        'name': 'Cobb Value',
        'unit': 'g/m²',
        'model_path': 'models/cobb/v3/comprehensive_polymer_model.pkl',
        'env_params': [],
        'min_parts': 3,  # 1 polymer (3 parts)
        'log_scale': True
    },
    'adhesion': {
        'name': 'Adhesion',
        'unit': 'N/15mm',
        'model_path': 'models/adhesion/v3/comprehensive_polymer_model.pkl',
        'env_params': ['Thickness (um)', 'Sealing Temperature (C)'],
        'min_parts': 4,  # 2 polymers (6 parts) + 2 environmental
        'log_scale': True
    },
    'compost': {
        'name': 'Home Compostability',
        'unit': '% disintegration',
        'model_path': None,  # Uses home-compost modules instead
        'env_params': ['Thickness (um)'],
        'min_parts': 3,  # 1 polymer (3 parts)
        'log_scale': False
    }
}

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

def parse_command_line_input(input_string, property_type):
    """
    Parse the command line input string based on property type.
    
    Args:
        input_string: comma-separated input string
        property_type: one of 'wvtr', 'ts', 'eab', 'cobb', 'all'
    
    Returns:
        - list of tuples: [(Material, Grade, vol_fraction), ...]
        - dict of environmental parameters or None
        - dict of available environmental parameters for 'all' mode
    """
    try:
        parts = [part.strip() for part in input_string.split(',')]
        
        # If the number of parts is divisible by 3, treat all as polymer info (no env params)
        if len(parts) % 3 == 0:
            polymer_parts = parts
            available_env_params = {}
        else:
            # Otherwise, use the previous logic for extracting env params
            available_env_params = {}
            polymer_parts = parts
            if property_type == 'all' and len(parts) >= 6:
                try:
                    temp_val = float(parts[-3])
                    rh_val = float(parts[-2])
                    thickness_val = float(parts[-1])
                    available_env_params['Temperature (C)'] = temp_val
                    available_env_params['RH (%)'] = rh_val
                    available_env_params['Thickness (um)'] = thickness_val
                    polymer_parts = parts[:-3]
                except Exception:
                    try:
                        thickness_val = float(parts[-1])
                        available_env_params['Thickness (um)'] = thickness_val
                        polymer_parts = parts[:-1]
                    except Exception:
                        polymer_parts = parts
            elif property_type == 'wvtr' and len(parts) >= 6:
                available_env_params['Temperature (C)'] = float(parts[-3])
                available_env_params['RH (%)'] = float(parts[-2])
                available_env_params['Thickness (um)'] = float(parts[-1])
                polymer_parts = parts[:-3]
            elif property_type in ['ts', 'eab'] and len(parts) >= 4:
                available_env_params['Thickness (um)'] = float(parts[-1])
                polymer_parts = parts[:-1]
            elif property_type == 'cobb':
                polymer_parts = parts
        
        # Build polymers list, using defaults for missing values
        polymers = []
        for i in range(0, len(polymer_parts), 3):
            material = polymer_parts[i] if i < len(polymer_parts) else 'Unknown'
            grade = polymer_parts[i + 1] if i + 1 < len(polymer_parts) else 'Unknown'
            try:
                vol_fraction = float(polymer_parts[i + 2]) if i + 2 < len(polymer_parts) else 0.0
            except Exception:
                vol_fraction = 0.0
            polymers.append((material, grade, vol_fraction))
        
        # Validate volume fractions sum to 1.0 (only if all are present)
        total_fraction = sum(vol_fraction for _, _, vol_fraction in polymers)
        if not np.isclose(total_fraction, 1.0, atol=1e-5):
            logger.warning(f"⚠️ Volume fractions do not sum to 1.0, got {total_fraction}")
        
        logger.info(f"✅ Parsed {len(polymers)} polymers with total volume fraction: {total_fraction}")
        if available_env_params:
            logger.info(f"✅ Available environmental parameters: {available_env_params}")
        
        return polymers, available_env_params
        
    except Exception as e:
        logger.error(f"❌ Error parsing input: {e}")
        return None, None

def get_env_params_for_property(available_env_params, property_type):
    """
    Get the environmental parameters needed for a specific property type.
    
    Args:
        available_env_params: dict of all available environmental parameters
        property_type: property type to get parameters for
    
    Returns:
        dict of environmental parameters for this property type (can include missing values)
    """
    config = PROPERTY_CONFIGS[property_type]
    required_params = config['env_params']
    
    env_params = {}
    
    for param in required_params:
        if param in available_env_params:
            env_params[param] = available_env_params[param]
        else:
            # Let the model handle missing values - don't skip the property
            env_params[param] = np.nan
    
    return env_params

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
            else:
                logger.error(f"❌ Material/Grade combination not found: {material} {grade}")
                return None
        
        return smiles_list
        
    except Exception as e:
        logger.error(f"❌ Error converting polymers to SMILES: {e}")
        return None

def create_input_dataframe(smiles_list, polymers, env_params=None):
    """
    Create a DataFrame in the exact format expected by the blend feature extractor.
    
    Args:
        smiles_list: list of tuples (SMILES, vol_fraction)
        polymers: original polymer list for metadata
        env_params: dictionary of environmental parameters
    
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
            'vol_fraction5': smiles_list[4][1] if len(smiles_list) > 4 else 0.0
        }
        # Always include all possible environmental columns
        all_env_cols = ['Temperature (C)', 'RH (%)', 'Thickness (um)', 'Sealing Temperature (C)']
        for col in all_env_cols:
            if env_params and col in env_params:
                data[col] = env_params[col]
            else:
                data[col] = np.nan
        
        df = pd.DataFrame([data])
        return df
        
    except Exception as e:
        logger.error(f"❌ Error creating input DataFrame: {e}")
        return None

def load_model(property_type, model_path=None):
    """Load the trained model for the specified property type."""
    try:
        config = PROPERTY_CONFIGS[property_type]
        if model_path is None:
            model_path = config['model_path']
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"❌ Error loading {property_type.upper()} model: {e}")
        return None

def prepare_features_for_prediction(featurized_df, model, property_type):
    """
    Prepare featurized data exactly as expected by the trained model.
    
    Args:
        featurized_df: DataFrame with featurized blend data
        model: loaded trained model
        property_type: property type for configuration
    
    Returns:
        prepared DataFrame ready for prediction
    """
    try:
        # Separate features and target (same logic as training script)
        target = 'property'
        smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
        # EXCLUDE Materials column from features (same as training script)
        excluded_cols = [target] + smiles_cols + ['Materials']
        
        # Remove excluded columns if they exist
        X = featurized_df.drop(columns=[col for col in excluded_cols if col in featurized_df.columns])
        
        # Identify categorical and numerical features (same logic as training script)
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Handle missing values (same logic as training script)
        for col in categorical_features:
            X[col] = X[col].fillna('Unknown')
        for col in numerical_features:
            X[col] = X[col].fillna(0)
        
        # Convert categorical features to categorical dtype for XGBoost
        for col in categorical_features:
            X[col] = X[col].astype('category')
        
        return X
        
    except Exception as e:
        logger.error(f"❌ Error preparing features: {e}")
        return None

def predict_property(features_df, model, property_type):
    """
    Make property predictions using the trained model.
    
    Args:
        features_df: prepared feature DataFrame
        model: loaded trained model
        property_type: property type for configuration
    
    Returns:
        prediction value or None if error
    """
    try:
        config = PROPERTY_CONFIGS[property_type]
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Convert from log scale if needed
        if config['log_scale']:
            prediction_original = np.exp(prediction)
            prediction = prediction_original
        else:
            prediction = prediction
        
        # Special handling for WVTR: convert from normalized to unnormalized (g/m2/day)
        if property_type == 'wvtr':
            # Get thickness from features (in um)
            thickness_um = features_df['Thickness (um)'].iloc[0]
            if thickness_um > 0:
                # Model output is in g·μm/m²/day, divide by thickness (μm) to get g/m²/day
                prediction = prediction / thickness_um
            else:
                logger.warning(f"⚠️ Thickness is zero or missing for WVTR prediction")
        
        return prediction
        
    except Exception as e:
        logger.error(f"❌ Error making {property_type.upper()} prediction: {e}")
        return None 