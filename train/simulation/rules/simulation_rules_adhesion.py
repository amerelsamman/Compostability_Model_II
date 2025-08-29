#!/usr/bin/env python3
"""Adhesion blending rules only - everything else is common"""

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
            
            print(f"TS prediction for blend: {ts_prediction:.2f} MPa")
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


def load_adhesion_data():
    """Load adhesion data"""
    adhesion_data = pd.read_csv('train/data/adhesion/masterdata.csv')  # Correct path from root directory
    
    return adhesion_data


def rule_of_mixtures(compositions: List[float], adhesion_values: List[float]) -> float:
    """Calculate adhesion using rule of mixtures weighted by volume fraction"""
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate adhesion weighted by volume fraction
    blend_adhesion = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        blend_adhesion += comp * adhesion
    
    return blend_adhesion


def inverse_rule_of_mixtures(compositions: List[float], adhesion_values: List[float]) -> float:
    """Calculate adhesion using inverse rule of mixtures"""
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate inverse adhesion
    inverse_adhesion_sum = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        if adhesion > 0:  # Avoid division by zero
            inverse_adhesion_sum += comp / adhesion
    
    # Return the final adhesion
    if inverse_adhesion_sum > 0:
        return 1 / inverse_adhesion_sum
    else:
        return 0


def combined_rule_of_mixtures(compositions: List[float], adhesion_values: List[float], thickness: float) -> float:
    """Calculate adhesion using combined rule: 50% rule of mixtures + 50% inverse rule of mixtures for thin films"""
    if thickness < 30:
        # Use 50/50 combination for thin films
        rom_adhesion = rule_of_mixtures(compositions, adhesion_values)
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, adhesion_values)
        combined_adhesion = 0.5 * rom_adhesion + 0.5 * inv_rom_adhesion
        return combined_adhesion
    else:
        # Use standard rule of mixtures for thicker films
        return rule_of_mixtures(compositions, adhesion_values)


def apply_adhesion_blending_rules(polymers: List[Dict], compositions: List[float], thickness: float) -> float:
    """Apply adhesion blending rules - combined rule for thin films, standard for thicker films"""
    adhesion_values = [p['adhesion'] for p in polymers]
    return combined_rule_of_mixtures(compositions, adhesion_values, thickness)


def scale_adhesion_with_thickness_and_ts_cap(base_adhesion: float, thickness: float, 
                                           ts_limit: float, reference_thickness: float = 20) -> float:
    """Scale adhesion with thickness scaling, capped at tensile strength limit"""
    empirical_exponent = 0.5  # Moderate scaling for balanced thickness sensitivity
    
    # Scale adhesion with thickness
    scaled_adhesion = base_adhesion * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))
    
    # No conversion needed! MPa and N/15mm are directly comparable:
    # - MPa = N/mm² (tensile strength)
    # - N/15mm = N/mm² when normalized by test width (peel strength)
    # So we can directly compare ts_limit (MPa) with adhesion (N/15mm)
    ts_limit_n_per_15mm = ts_limit
    
    # Cap adhesion at the tensile strength limit
    capped_adhesion = min(scaled_adhesion, ts_limit_n_per_15mm)
    
    if capped_adhesion < scaled_adhesion:
        print(f"Adhesion capped at TS limit: {capped_adhesion:.3f} N/15mm (TS: {ts_limit:.2f} MPa)")
    
    return capped_adhesion


def create_adhesion_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create adhesion blend row with thickness scaling and sealing temperature - EXACTLY as original"""
    # Generate random thickness - EXACTLY as original
    thickness = np.random.uniform(10, 300)  # Thickness between 10-300 μm - EXACTLY as original
    
    # Calculate blend sealing temperature using lowest melt temperature in the blend
    sealing_temps = [p.get('sealing_temp', 23.0) for p in polymers]  # Get sealing temperatures from polymers
    blend_sealing_temp = min(sealing_temps)  # Use lowest melt temperature in the blend
    
    # Use the calculated blend sealing temperature
    blend_temperature = blend_sealing_temp
    
    # Use combined rule of mixtures for thin films (< 30 μm), standard rule for thicker films - EXACTLY as original
    blend_adhesion = apply_adhesion_blending_rules(polymers, compositions, thickness)
    
    # Debug: Show which rule was used - EXACTLY as original
    if thickness < 30:
        rom_adhesion = rule_of_mixtures(compositions, [p['adhesion'] for p in polymers])
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, [p['adhesion'] for p in polymers])
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm < 30 μm - Using 50/50 combined rule")
        print(f"  Rule of Mixtures: {rom_adhesion:.3f}, Inverse Rule: {inv_rom_adhesion:.3f}, Combined: {blend_adhesion:.3f}")
    else:
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm ≥ 30 μm - Using standard rule of mixtures: {blend_adhesion:.3f}")
    
    # NEW: Predict tensile strength to cap adhesion scaling
    ts_limit = predict_tensile_strength_for_blend(polymers, compositions, thickness)
    
    # Scale adhesion based on thickness using fixed 20 μm reference, capped at TS limit
    blend_adhesion = scale_adhesion_with_thickness_and_ts_cap(blend_adhesion, thickness, ts_limit, reference_thickness=20)
    
    # No temperature scaling needed - we're at the optimal sealing temperature - EXACTLY as original
    
    # Add noise - EXACTLY as original
    noise_level = 0.05  # 5% noise - EXACTLY as original
    blend_adhesion_noisy = blend_adhesion * (1 + np.random.normal(0, noise_level))
    
    # Ensure the results stay positive - EXACTLY as original
    blend_adhesion_noisy = max(blend_adhesion_noisy, 0.1)  # Minimum adhesion of 0.1
    
    # DEBUG: Print the property value to ensure it's not NaN - EXACTLY as original
    if pd.isna(blend_adhesion_noisy) or blend_adhesion_noisy <= 0:
        print(f"WARNING: Invalid property value for blend {blend_number}: {blend_adhesion_noisy}")
        blend_adhesion_noisy = 0.5  # Fallback value
    
    # Fill polymer grades - EXACTLY as original
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES - EXACTLY as original
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(compositions))
    
    # Fill volume fractions - EXACTLY as original
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Create complete row with all required columns - EXACTLY as original
    row = {
        'Materials': str(blend_number),  # Use blend number for Materials column - EXACTLY as original
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
        'property1': blend_temperature,  # Sealing temperature (property1)
        'property2': blend_adhesion_noisy,  # Adhesion strength (property2)
        'unit': 'N/15mm'  # Default unit for adhesion - EXACTLY as original
    }
    
    return row
