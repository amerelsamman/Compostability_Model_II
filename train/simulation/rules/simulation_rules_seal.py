#!/usr/bin/env python3
"""Sealing blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import sys
import os
import tempfile
import joblib

# Import from local modules copy (same directory as rules)
try:
    from .modules.blend_feature_extractor import process_blend_features
    from .modules.prediction_utils import load_model, prepare_features_for_prediction, predict_property
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import from local modules: {e}")
    print("TS prediction will use fallback values")
    MODULES_AVAILABLE = False

def predict_tensile_strength_for_blend(polymers: List[Dict], compositions: List[float], thickness: float) -> float:
    """Predict tensile strength for a blend using the actual TS model and feature extraction"""
    if not MODULES_AVAILABLE:
        print(f"Modules not available, using fallback TS value")
        return 50.0
    
    try:
        # Create a temporary blend row for TS prediction (same format as prediction_engine.py)
        blend_data = {
            'Materials': 'temp_blend',
            'Polymer Grade 1': polymers[0]['grade'],
            'Polymer Grade 2': polymers[1]['grade'] if len(polymers) > 1 else 'Unknown',
            'Polymer Grade 3': polymers[2]['grade'] if len(polymers) > 2 else 'Unknown',
            'Polymer Grade 4': polymers[3]['grade'] if len(polymers) > 3 else 'Unknown',
            'Polymer Grade 5': polymers[4]['grade'] if len(polymers) > 4 else 'Unknown',
            'SMILES1': polymers[0]['smiles'],
            'SMILES2': polymers[1]['smiles'] if len(polymers) > 1 else '',
            'SMILES3': polymers[2]['smiles'] if len(polymers) > 2 else '',
            'SMILES4': polymers[3]['smiles'] if len(polymers) > 3 else '',
            'SMILES5': polymers[4]['smiles'] if len(polymers) > 4 else '',
            'vol_fraction1': compositions[0],
            'vol_fraction2': compositions[1] if len(compositions) > 1 else 0,
            'vol_fraction3': compositions[2] if len(compositions) > 2 else 0,
            'vol_fraction4': compositions[3] if len(compositions) > 3 else 0,
            'vol_fraction5': compositions[4] if len(compositions) > 4 else 0,
            'Thickness (um)': thickness
        }
        
        # Create temporary files for feature extraction (same as prediction_engine.py)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_input:
            temp_input_path = temp_input.name
            pd.DataFrame([blend_data]).to_csv(temp_input_path, index=False)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Extract features for the blend (same as prediction_engine.py)
            featurized_df = process_blend_features(temp_input_path, temp_output_path)
            
            if featurized_df is None or len(featurized_df) == 0:
                print(f"Warning: Could not extract features for TS prediction, using fallback value")
                return 50.0
            
            # Load TS model (same as prediction_engine.py)
            ts_model = load_model('ts')
            if ts_model is None:
                print(f"Warning: Could not load TS model, using fallback value")
                return 50.0
            
            # Prepare features for prediction (same as prediction_engine.py)
            features_df = prepare_features_for_prediction(featurized_df, ts_model, 'ts')
            if features_df is None:
                print(f"Warning: Could not prepare features for TS prediction, using fallback value")
                return 50.0
            
            # Make TS prediction (same as prediction_engine.py)
            ts_prediction = predict_property(features_df, ts_model, 'ts')
            
            if ts_prediction is None or ts_prediction <= 0:
                print(f"Warning: Invalid TS prediction, using fallback value")
                return 50.0
            
            return ts_prediction
            
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
            except:
                pass
                
    except Exception as e:
        print(f"Error in TS prediction: {e}, using fallback value")
        return 50.0  # Fallback TS value in MPa


def load_seal_data():
    """Load seal data"""
    seal_data = pd.read_csv('train/data/seal/masterdata.csv')  # Correct path from root directory
    
    return seal_data


def rule_of_mixtures(compositions: List[float], seal_values: List[float]) -> float:
    """Calculate seal using rule of mixtures weighted by volume fraction"""
    if len(compositions) != len(seal_values):
        raise ValueError("Compositions and seal values must have same length")
    
    # Calculate seal weighted by volume fraction
    blend_seal = 0
    for comp, seal in zip(compositions, seal_values):
        blend_seal += comp * seal
    
    return blend_seal


def inverse_rule_of_mixtures(compositions: List[float], seal_values: List[float]) -> float:
    """Calculate seal using inverse rule of mixtures"""
    if len(compositions) != len(seal_values):
        raise ValueError("Compositions and seal values must have same length")
    
    # Calculate inverse seal
    inverse_seal_sum = 0
    for comp, seal in zip(compositions, seal_values):
        if seal > 0:  # Avoid division by zero
            inverse_seal_sum += comp / seal
    
    # Return the final seal
    if inverse_seal_sum > 0:
        return 1 / inverse_seal_sum
    else:
        return 0


def combined_rule_of_mixtures(compositions: List[float], seal_values: List[float], thickness: float) -> float:
    """Calculate seal using combined rule: 50% rule of mixtures + 50% inverse rule of mixtures for thin films"""
    if thickness < 30:
        # Use 50/50 combination for thin films
        rom_seal = rule_of_mixtures(compositions, seal_values)
        inv_rom_seal = inverse_rule_of_mixtures(compositions, seal_values)
        combined_seal = 0.5 * rom_seal + 0.5 * inv_rom_seal
        return combined_seal
    else:
        # Use standard rule of mixtures for thicker films
        return rule_of_mixtures(compositions, seal_values)


def apply_seal_blending_rules(polymers: List[Dict], compositions: List[float], thickness: float, selected_rules: Dict[str, bool] = None) -> float:
    """Apply seal blending rules based on selected rules configuration"""
    seal_values = [p['seal'] for p in polymers]
    
    # If no rules specified, use default behavior (all rules enabled)
    if selected_rules is None:
        return combined_rule_of_mixtures(compositions, seal_values, thickness)
    
    # Check which rules are enabled
    use_combined_rom_thin = selected_rules.get('combined_rom_thin', True)
    use_standard_rom_thick = selected_rules.get('standard_rom_thick', True)
    
    # Apply rules based on thickness and enabled rules
    if thickness < 30 and use_combined_rom_thin:
        # Use combined rule for thin films
        return combined_rule_of_mixtures(compositions, seal_values, thickness)
    elif thickness >= 30 and use_standard_rom_thick:
        # Use standard rule for thicker films
        return rule_of_mixtures(compositions, seal_values)
    else:
        # Fallback to standard rule if no specific rule is enabled
        return rule_of_mixtures(compositions, seal_values)


def scale_seal_with_thickness_and_ts_cap(base_seal: float, thickness: float, 
                                           ts_limit: float, reference_thickness: float = 20,
                                           thin_film_threshold: float = 50.0,
                                           thin_film_exponent: float = 0.5,
                                           thick_film_exponent: float = 0.25) -> float:
    """Scale seal with thickness scaling, capped at tensile strength limit"""
    # Dynamic scaling using environmental parameters
    if thickness <= thin_film_threshold:
        empirical_exponent = thin_film_exponent  # Use thin film exponent
    else:
        # Linear interpolation from thin_film_exponent at threshold to thick_film_exponent at 300μm
        empirical_exponent = thin_film_exponent - (thin_film_exponent - thick_film_exponent) * ((thickness - thin_film_threshold) / (300 - thin_film_threshold))
        empirical_exponent = max(thick_film_exponent, empirical_exponent)  # Don't go below thick_film_exponent
    
    # Scale seal with thickness
    scaled_seal = base_seal * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))
    
    # No conversion needed! MPa and N/15mm are directly comparable:
    # - MPa = N/mm² (tensile strength)
    # - N/15mm = N/mm² when normalized by test width (peel strength)
    # So we can directly compare ts_limit (MPa) with seal (N/15mm)
    ts_limit_n_per_15mm = ts_limit
    
    # Cap seal at the tensile strength limit
    capped_seal = min(scaled_seal, ts_limit_n_per_15mm)
    
    return capped_seal


def create_seal_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int, rule_tracker=None, selected_rules: Dict[str, bool] = None, environmental_config: Dict[str, Any] = None, disable_ts_model: bool = False) -> Dict[str, Any]:
    """Create seal blend row with thickness scaling - sealing strength only"""
    
    # Generate random thickness using environmental controls
    if environmental_config and 'seal' in environmental_config:
        env_params = environmental_config['seal']
        thickness_config = env_params['thickness']
        thickness = np.random.uniform(thickness_config['min'], thickness_config['max'])
    else:
        # Fallback to original hardcoded values
        thickness = np.random.uniform(10, 300)  # Thickness between 10-300 μm
    
    # Use selected rules for blending
    blend_seal = apply_seal_blending_rules(polymers, compositions, thickness, selected_rules)
    
    # Track rule usage based on selected rules and thickness
    if rule_tracker is not None:
        if selected_rules is None:
            # Default behavior - track based on thickness
            if thickness < 30:
                rule_tracker.record_rule_usage("Combined Rule of Mixtures (thin films < 30μm)")
            else:
                rule_tracker.record_rule_usage("Standard Rule of Mixtures (thick films ≥ 30μm)")
        else:
            # Track based on which rules are actually enabled and used
            if thickness < 30 and selected_rules.get('combined_rom_thin', True):
                rule_tracker.record_rule_usage("Combined Rule of Mixtures (thin films < 30μm)")
            elif thickness >= 30 and selected_rules.get('standard_rom_thick', True):
                rule_tracker.record_rule_usage("Standard Rule of Mixtures (thick films ≥ 30μm)")
            else:
                # Fallback rule
                rule_tracker.record_rule_usage("Standard Rule of Mixtures (fallback)")
    
    # Predict tensile strength to cap seal scaling (if not disabled)
    if disable_ts_model:
        ts_limit = 1000.0  # Use high default value when TS model is disabled
    else:
        ts_limit = predict_tensile_strength_for_blend(polymers, compositions, thickness)
    
    # Scale seal based on thickness using environmental parameters, capped at TS limit
    if environmental_config and 'seal' in environmental_config:
        env_params = environmental_config['seal']
        thickness_config = env_params['thickness']
        reference_thickness = thickness_config['reference']
        
        # Get dynamic scaling parameters
        if 'dynamic_scaling' in thickness_config:
            dynamic_config = thickness_config['dynamic_scaling']
            thin_film_threshold = dynamic_config.get('thin_film_threshold', 50.0)
            thin_film_exponent = dynamic_config.get('thin_film_exponent', 0.5)
            thick_film_exponent = dynamic_config.get('thick_film_exponent', 0.25)
        else:
            thin_film_threshold = 50.0
            thin_film_exponent = 0.5
            thick_film_exponent = 0.25
    else:
        reference_thickness = 20.0  # Fallback to original value
        thin_film_threshold = 50.0
        thin_film_exponent = 0.5
        thick_film_exponent = 0.25
    
    blend_seal = scale_seal_with_thickness_and_ts_cap(blend_seal, thickness, ts_limit, 
                                                     reference_thickness=reference_thickness,
                                                     thin_film_threshold=thin_film_threshold,
                                                     thin_film_exponent=thin_film_exponent,
                                                     thick_film_exponent=thick_film_exponent)
    
    # No noise added - clean simulation
    blend_seal_final = blend_seal
    
    # Check for invalid property values
    if pd.isna(blend_seal_final) or blend_seal_final <= 0:
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            print(f"WARNING: Invalid property value for blend {blend_number}: {blend_seal_final}")
        blend_seal_final = 0.5  # Fallback value
    
    # Fill polymer grades
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(compositions))
    
    # Fill volume fractions
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Create complete row with all required columns - single property output
    row = {
        'Materials': str(blend_number),  # Use blend number for Materials column
        'Polymer Grade 1': grades[0],
        'Polymer Grade 2': grades[1],
        'Polymer Grade 3': grades[2],
        'Polymer Grade 4': grades[3],
        'Polymer Grade 5': grades[4],
        'SMILES1': smiles[0],
        'SMILES2': smiles[1],
        'SMILES3': smiles[2],
        'SMILES4': smiles[3],
        'SMILES5': smiles[4],
        'vol_fraction1': vol_fractions[0],
        'vol_fraction2': vol_fractions[1],
        'vol_fraction3': vol_fractions[2],
        'vol_fraction4': vol_fractions[3],
        'vol_fraction5': vol_fractions[4],
        'Thickness (um)': thickness,
        'property': blend_seal_final  # Sealing strength (seal strength) - single property
    }
    
    return row
