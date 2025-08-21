#!/usr/bin/env python3
"""
Modular predictor for polymer blend compostability using the differentiable label optimization model.
This module extracts the core prediction logic from predict_blend_cli.py for reuse.
"""

import pandas as pd
import numpy as np
import os
import warnings
import random
from contextlib import redirect_stdout, redirect_stderr
import io
import sys

from .optimizer import DifferentiableLabelOptimizer
from .blend_feature_extractor import process_blend_features
from .utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves, generate_quintic_biodegradation_curve

warnings.filterwarnings('ignore')

def parse_blend_string(blend_string):
    """
    Parse a blend string like "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5"
    Returns a list of tuples: [(polymer_name, grade, fraction), ...]
    """
    parts = [part.strip() for part in blend_string.split(',')]
    polymers = []
    
    i = 0
    while i < len(parts):
        if i + 2 < len(parts):
            polymer_name = parts[i]
            grade = parts[i + 1]
            try:
                fraction = float(parts[i + 2])
                polymers.append((polymer_name, grade, fraction))
                i += 3
            except ValueError:
                i += 3
        else:
            break
    
    return polymers

def create_blend_data_from_string(blend_string):
    """
    Create a DataFrame for the specified blend using polymer properties reference.
    """
    # Parse the blend string
    polymers = parse_blend_string(blend_string)
    
    # Load polymer properties reference
    polymer_ref = pd.read_csv('polymer_properties_reference.csv')
    
    # Create blend data
    blend_data = {
        'Materials': blend_string,
        'Polymer Grade 1': 'Unknown',
        'Polymer Grade 2': 'Unknown', 
        'Polymer Grade 3': 'Unknown',
        'Polymer Grade 4': 'Unknown',
        'Polymer Grade 5': 'Unknown',
        'SMILES1': '',
        'SMILES2': '',
        'SMILES3': '',
        'SMILES4': '',
        'SMILES5': '',
        'vol_fraction1': 0.0,
        'vol_fraction2': 0.0,
        'vol_fraction3': 0.0,
        'vol_fraction4': 0.0,
        'vol_fraction5': 0.0,
        'Thickness_certification': ''
    }
    
    # Fill in polymer data
    for i, (polymer_name, grade, fraction) in enumerate(polymers):
        if i >= 5:  # Only support up to 5 polymers
            break
            
        # Find the polymer in the reference
        mask = (polymer_ref['Polymer Grade 1'] == grade)
        if mask.any():
            polymer_row = polymer_ref[mask].iloc[0]
            blend_data[f'Polymer Grade {i+1}'] = grade
            blend_data[f'SMILES{i+1}'] = polymer_row['SMILES1']
            blend_data[f'vol_fraction{i+1}'] = fraction
        else:
            blend_data[f'Polymer Grade {i+1}'] = grade
            blend_data[f'vol_fraction{i+1}'] = fraction
    
    blend_df = pd.DataFrame([blend_data])
    return blend_df

def prepare_features_for_prediction(processed_df):
    """
    Prepare features for prediction using the same process as training.
    """
    # Filter out metadata columns
    exclude_cols = ['Materials', 'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5']
    processed_df = processed_df.drop(columns=exclude_cols, errors='ignore')
    
    # Molecular features only (90 features)
    feature_order = [
        'vol_fraction1', 'vol_fraction2', 'vol_fraction3', 'vol_fraction4', 'vol_fraction5',
        'SP_C', 'SP_N', 'SP2_C', 'SP2_N', 'SP2_O', 'SP2_S', 'SP2_B',
        'SP3_C', 'SP3_N', 'SP3_O', 'SP3_S', 'SP3_P', 'SP3_Si', 'SP3_B',
        'SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I', 'SP3D2_S',
        'phenyls', 'cyclohexanes', 'cyclopentanes', 'cyclopentenes', 'thiophenes',
        'aromatic_rings_with_n', 'aromatic_rings_with_o', 'aromatic_rings_with_n_o',
        'aromatic_rings_with_s', 'aliphatic_rings_with_n', 'aliphatic_rings_with_o',
        'aliphatic_rings_with_n_o', 'aliphatic_rings_with_s', 'other_rings',
        'carboxylic_acid', 'anhydride', 'acyl_halide', 'carbamide', 'urea',
        'carbamate', 'thioamide', 'amide', 'ester', 'sulfonamide', 'sulfone',
        'sulfoxide', 'phosphate', 'nitro', 'acetal', 'ketal', 'isocyanate',
        'thiocyanate', 'azide', 'azo', 'imide', 'sulfonyl_halide', 'phosphonate',
        'thiourea', 'guanidine', 'silicon_4_coord', 'boron_3_coord', 'vinyl',
        'vinyl_halide', 'allene', 'alcohol', 'ether', 'aldehyde', 'ketone',
        'thiol', 'thioether', 'primary_amine', 'secondary_amine', 'tertiary_amine',
        'quaternary_amine', 'imine', 'nitrile', 'primary_carbon', 'secondary_carbon',
        'tertiary_carbon', 'quaternary_carbon', 'branching_factor', 'tree_depth',
        'ethyl_chain', 'propyl_chain', 'butyl_chain', 'long_chain'
    ]
    
    # Extract only the feature columns in the correct order
    features = processed_df[feature_order].values
    
    return features, feature_order

def predict_compostability_core(blend_string, actual_thickness=None, model_dir="models/eol/v4/", 
                               suppress_output=True, save_plots=False, output_dir="."):
    """
    Core prediction function that replicates the exact logic from predict_blend_cli.py
    
    Args:
        blend_string: Blend specification string
        actual_thickness: Thickness in mm (default: 0.050 mm = 50 μm)
        model_dir: Directory containing the trained model
        suppress_output: Whether to suppress verbose output
        save_plots: Whether to save plots and CSV files
        output_dir: Directory to save output files
    
    Returns:
        dict: Prediction results with max_disintegration, max_biodegradation, etc.
    """
    if actual_thickness is None:
        actual_thickness = 0.050  # Default 50 μm
    
    try:
        # Suppress output if requested
        if suppress_output:
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            stdout_redirect = redirect_stdout(stdout_capture)
            stderr_redirect = redirect_stderr(stderr_capture)
        else:
            stdout_redirect = redirect_stdout(sys.stdout)
            stderr_redirect = redirect_stderr(sys.stderr)
        
        with stdout_redirect, stderr_redirect:
            # Step 1: Create blend data from blend string
            blend_df = create_blend_data_from_string(blend_string)
            
            # NEW RULE: Check if blend contains only polymers with known max_L values
            polymer_props = pd.read_csv('polymer_properties_reference.csv')
            blend_components = parse_blend_string(blend_string)
            
            all_polymers_have_max_L = True
            polymer_max_L_values = []
            polymer_t0_values = []
            total_fraction = 0.0
            
            for material, grade, fraction in blend_components:
                total_fraction += fraction
                # Find the polymer in the reference
                polymer_row = polymer_props[polymer_props['Polymer Grade 1'] == grade]
                if not polymer_row.empty:
                    max_L = polymer_row.iloc[0]['property1']  # max_L value
                    t0 = polymer_row.iloc[0]['property2']     # t0 value
                    if pd.notna(max_L) and pd.notna(t0):
                        polymer_max_L_values.append(max_L)
                        polymer_t0_values.append(t0)
                    else:
                        all_polymers_have_max_L = False
                else:
                    all_polymers_have_max_L = False
            
            # Check if all polymers have max_L and total fraction is 1.0
            if all_polymers_have_max_L and abs(total_fraction - 1.0) < 0.01 and len(polymer_max_L_values) > 0:
                # Check if all polymers are home-compostable (max_L > 90)
                all_home_compostable = all(max_L > 90 for max_L in polymer_max_L_values)
                
                if all_home_compostable:
                    # For purely home-compostable blends, use random max_L between 90-95
                    max_L_pred = random.uniform(90.0, 95.0)
                    
                    # Calculate weighted average t0
                    weighted_t0 = 0.0
                    for i, (material, grade, fraction) in enumerate(blend_components):
                        weighted_t0 += fraction * polymer_t0_values[i]
                    t0_pred = weighted_t0
                else:
                    # Calculate weighted averages based on volume fractions for mixed blends
                    weighted_max_L = 0.0
                    weighted_t0 = 0.0
                    
                    for i, (material, grade, fraction) in enumerate(blend_components):
                        weighted_max_L += fraction * polymer_max_L_values[i]
                        weighted_t0 += fraction * polymer_t0_values[i]
                    
                    # Use the weighted averages instead of model prediction
                    max_L_pred = weighted_max_L
                    t0_pred = weighted_t0
                
                skip_model_prediction = True
            else:
                skip_model_prediction = False
            
            # NEW PLA RULE: Check for PLA + compostable polymer rule
            has_PLA = False
            has_compostable_polymer = False
            compostable_polymer_fraction = 0.0
            non_compostable_polymer_fraction = 0.0
            blend_t0_values = []
            blend_fractions = []
            
            for material, grade, fraction in blend_components:
                # Find the polymer in the reference
                polymer_row = polymer_props[polymer_props['Polymer Grade 1'] == grade]
                if not polymer_row.empty:
                    max_L = polymer_row.iloc[0]['property1']
                    t0 = polymer_row.iloc[0]['property2']
                    
                    # Check if it's PLA
                    if 'PLA' in material.upper():
                        has_PLA = True
                    
                    # Check if it's a compostable polymer (max_L > 90)
                    if max_L > 90:
                        has_compostable_polymer = True
                        compostable_polymer_fraction += fraction
                    else:
                        # Check if it's non-compostable (max_L < 90) and NOT PLA
                        if 'PLA' not in material.upper():
                            if fraction > 0.20:
                                non_compostable_polymer_fraction += fraction
                    
                    # Store t0 and fraction for weighted average calculation
                    if pd.notna(t0):
                        blend_t0_values.append(t0)
                        blend_fractions.append(fraction)
            
            # Apply PLA rule if conditions are met
            if (has_PLA and has_compostable_polymer and 
                compostable_polymer_fraction >= 0.15 and 
                non_compostable_polymer_fraction <= 0.20):
                
                # Set max_L = random value between 90-95 and calculate weighted average t0
                max_L_pred = random.uniform(90.0, 95.0)
                weighted_t0 = 0.0
                
                for i in range(len(blend_t0_values)):
                    weighted_t0 += blend_fractions[i] * blend_t0_values[i]
                
                t0_pred = weighted_t0
                
                # Skip model prediction and go directly to k0 calculation
                skip_model_prediction = True
            elif not skip_model_prediction:
                # Use model prediction
                pass
            
            # Step 2: Process blend features using the same process as training
            # Save blend data to temporary file first
            temp_input_file = "temp_blend_input.csv"
            temp_features_file = "temp_blend_features.csv"
            blend_df.to_csv(temp_input_file, index=False)
            processed_df = process_blend_features(temp_input_file, temp_features_file)
            
            # Step 3: Prepare features for prediction
            features, feature_order = prepare_features_for_prediction(processed_df)
            
            # Step 4: Load the trained model and make predictions (if needed)
            if not skip_model_prediction:
                dlo = DifferentiableLabelOptimizer(device='cpu')
                metadata = dlo.load_model(save_dir=model_dir, model_name='dlo_model')
                
                # Step 5: Make predictions
                predictions = dlo.predict(features, use_scaler=True)
                
                # Extract predictions
                max_L_pred = predictions[0, 0]  # property1 = max_L
                t0_pred = predictions[0, 1]     # property2 = t0
            
            # Step 6: Calculate k0 values
            # Determine majority polymer behavior for k0 selection
            # Get the majority polymer from the blend
            majority_polymer = None
            majority_fraction = 0.0
            
            for material, grade, fraction in blend_components:
                if fraction > majority_fraction:
                    majority_fraction = fraction
                    majority_polymer = grade
            
            # Find the majority polymer's max_L value
            majority_max_L = None
            if majority_polymer:
                polymer_row = polymer_props[polymer_props['Polymer Grade 1'] == majority_polymer]
                if not polymer_row.empty:
                    majority_max_L = polymer_row.iloc[0]['property1']  # This is the max_L value
            
            # Determine if majority polymer has high or low disintegration
            majority_high_disintegration = None
            if majority_max_L is not None:
                majority_high_disintegration = majority_max_L > 5
            
            k0_disintegration = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred, t_max=200.0, 
                                                               majority_polymer_high_disintegration=majority_high_disintegration,
                                                               actual_thickness=actual_thickness)
            k0_biodegradation = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred * 2.0, t_max=400.0, 
                                                               majority_polymer_high_disintegration=majority_high_disintegration,
                                                               actual_thickness=actual_thickness)
            
            # Step 7: Generate sigmoid curves
            if save_plots:
                # Save plots in the current directory
                prediction_output_dir = output_dir
                
                # Disintegration curves (200 days)
                disintegration_df = generate_sigmoid_curves(
                    np.array([max_L_pred]), 
                    np.array([t0_pred]), 
                    np.array([k0_disintegration]), 
                    days=200, 
                    curve_type='disintegration',
                    save_dir=prediction_output_dir,
                    actual_thickness=actual_thickness
                )
                
                # Biodegradation curves (400 days, using quintic polynomial based on disintegration)
                biodegradation_df = generate_quintic_biodegradation_curve(
                    disintegration_df, 
                    t0_pred, 
                    max_L_pred, 
                    days=400, 
                    save_dir=prediction_output_dir,
                    actual_thickness=actual_thickness
                )
            else:
                # Generate curves without saving files
                disintegration_df = generate_sigmoid_curves(
                    np.array([max_L_pred]), 
                    np.array([t0_pred]), 
                    np.array([k0_disintegration]), 
                    days=200, 
                    curve_type='disintegration',
                    save_csv=False,
                    save_plot=False,
                    save_dir='.',
                    actual_thickness=actual_thickness
                )
                
                biodegradation_df = generate_quintic_biodegradation_curve(
                    disintegration_df, 
                    t0_pred, 
                    max_L_pred, 
                    days=400, 
                    save_csv=False,
                    save_plot=False,
                    save_dir='.',
                    actual_thickness=actual_thickness
                )
            
            # Get max values from curves (use capped values from quintic curves)
            max_disintegration = disintegration_df['disintegration'].max() if not disintegration_df.empty else min(max_L_pred, 95.0)
            max_biodegradation = biodegradation_df['biodegradation'].max() if not biodegradation_df.empty else min(max_L_pred, 95.0)
            
            # Clean up temporary files
            if os.path.exists(temp_input_file):
                os.remove(temp_input_file)
            if os.path.exists(temp_features_file):
                os.remove(temp_features_file)
            
            # Return results
            results = {
                'max_disintegration': max_disintegration,
                'max_biodegradation': max_biodegradation,
                't0_pred': t0_pred,
                'k0_disintegration': k0_disintegration,
                'k0_biodegradation': k0_biodegradation,
                'disintegration_curve': disintegration_df,
                'biodegradation_curve': biodegradation_df,
                'blend_string': blend_string,
                'actual_thickness': actual_thickness
            }
            
            return results
            
    except Exception as e:
        return None 