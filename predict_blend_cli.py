#!/usr/bin/env python3
"""
Command line interface for polymer blend prediction.
Usage: python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5" --output_prefix test_prediction
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append('modules')

from modules.optimizer import DifferentiableLabelOptimizer
from modules.blend_feature_extractor import process_blend_features
from modules.utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves

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
                print(f"Warning: Could not parse fraction '{parts[i + 2]}' for {polymer_name} {grade}")
                i += 3
        else:
            break
    
    return polymers

def create_blend_data_from_string(blend_string):
    """
    Create a DataFrame for the specified blend using polymer properties reference.
    """
    print(f"Parsing blend: {blend_string}")
    
    # Parse the blend string
    polymers = parse_blend_string(blend_string)
    print(f"Found {len(polymers)} polymers: {polymers}")
    
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
            print(f"Warning: Only first 5 polymers will be used")
            break
            
        # Find the polymer in the reference
        mask = (polymer_ref['Polymer Grade 1'] == grade)
        if mask.any():
            polymer_row = polymer_ref[mask].iloc[0]
            blend_data[f'Polymer Grade {i+1}'] = grade
            blend_data[f'SMILES{i+1}'] = polymer_row['SMILES1']
            blend_data[f'vol_fraction{i+1}'] = fraction
            
            print(f"  Found {polymer_name} {grade}: SMILES = {polymer_row['SMILES1']}, fraction = {fraction}")
        else:
            print(f"  Warning: Could not find polymer grade '{grade}' in reference")
    
    return pd.DataFrame([blend_data])

def prepare_features_for_prediction(processed_df):
    """
    Prepare features for prediction, ensuring the same order as training.
    """
    # Define the exact feature order (90 molecular features only)
    feature_order = [
        'MolWt1', 'MolWt2', 'MolWt3', 'MolWt4', 'MolWt5',
        'LogP1', 'LogP2', 'LogP3', 'LogP4', 'LogP5',
        'NumRotatableBonds1', 'NumRotatableBonds2', 'NumRotatableBonds3', 'NumRotatableBonds4', 'NumRotatableBonds5',
        'NumHAcceptors1', 'NumHAcceptors2', 'NumHAcceptors3', 'NumHAcceptors4', 'NumHAcceptors5',
        'NumHDonors1', 'NumHDonors2', 'NumHDonors3', 'NumHDonors4', 'NumHDonors5',
        'NumAromaticRings1', 'NumAromaticRings2', 'NumAromaticRings3', 'NumAromaticRings4', 'NumAromaticRings5',
        'NumSaturatedRings1', 'NumSaturatedRings2', 'NumSaturatedRings3', 'NumSaturatedRings4', 'NumSaturatedRings5',
        'FractionCsp31', 'FractionCsp32', 'FractionCsp33', 'FractionCsp34', 'FractionCsp35',
        'FractionCsp21', 'FractionCsp22', 'FractionCsp23', 'FractionCsp24', 'FractionCsp25',
        'FractionCsp1', 'FractionCsp2', 'FractionCsp3', 'FractionCsp4', 'FractionCsp5',
        'FractionCsp01', 'FractionCsp02', 'FractionCsp03', 'FractionCsp04', 'FractionCsp05',
        'FractionCsp11', 'FractionCsp12', 'FractionCsp13', 'FractionCsp14', 'FractionCsp15',
        'FractionCsp21', 'FractionCsp22', 'FractionCsp23', 'FractionCsp24', 'FractionCsp25',
        'FractionCsp31', 'FractionCsp32', 'FractionCsp33', 'FractionCsp34', 'FractionCsp35',
        'FractionCsp41', 'FractionCsp42', 'FractionCsp43', 'FractionCsp44', 'FractionCsp45',
        'FractionCsp51', 'FractionCsp52', 'FractionCsp53', 'FractionCsp54', 'FractionCsp55',
        'FractionCsp61', 'FractionCsp62', 'FractionCsp63', 'FractionCsp64', 'FractionCsp65',
        'FractionCsp71', 'FractionCsp72', 'FractionCsp73', 'FractionCsp74', 'FractionCsp75',
        'FractionCsp81', 'FractionCsp82', 'FractionCsp83', 'FractionCsp84', 'FractionCsp85'
    ]
    
    # Extract features in the correct order
    features = []
    for feature in feature_order:
        if feature in processed_df.columns:
            features.append(processed_df[feature].iloc[0])
        else:
            print(f"Warning: Feature '{feature}' not found in processed data, using 0.0")
            features.append(0.0)
    
    return np.array([features]), feature_order

def calculate_polymer_k0(max_L: float, t0: float, t_max: float = 200.0) -> float:
    """
    Calculate k0 for a single polymer using the sigmoid equation.
    
    Args:
        max_L: Maximum disintegration level for the polymer
        t0: Time at 50% disintegration for the polymer
        t_max: Time at which max_L should be reached (default 200 days)
        
    Returns:
        k0: Rate constant for the polymer
    """
    if t0 <= 0 or t_max <= t0:
        return 0.1  # Default value if parameters are invalid
    
    try:
        # Calculate k0 from both boundary conditions (same as in utils.py)
        k0_from_start = np.log(999) / t0
        k0_from_end = -np.log(1/0.999 - 1) / (t_max - t0)
        
        # Use the maximum to satisfy both conditions
        k0 = max(k0_from_start, k0_from_end)
        
        # Ensure k0 is positive and reasonable
        return max(0.01, min(5.0, k0))
    except (ValueError, ZeroDivisionError):
        return 0.1  # Default value if calculation fails

def calculate_max_L_from_k0_t0(k0: float, t0: float, target_y: float, target_t: float = 200.0) -> float:
    """
    Calculate max_L from k0, t0, and target y value at target time.
    
    Args:
        k0: Rate constant
        t0: Time at 50% disintegration
        target_y: Target y value at target_t
        target_t: Time at which target_y should be reached (default 200 days)
        
    Returns:
        max_L: Maximum disintegration level
    """
    # SIGMOID FUNCTION: y = max_L / (1 + exp(-k0 * (t - t0)))
    # Solving for max_L: max_L = y * (1 + exp(-k0 * (t - t0)))
    
    try:
        max_L = target_y * (1 + np.exp(-k0 * (target_t - t0)))
        return max_L
    except (ValueError, OverflowError):
        return 95.0  # Default value if calculation fails

def predict_blend(blend_string, output_prefix="cli_prediction", model_dir="models/v1/"):
    """Main prediction function."""
    print("="*60)
    print("POLYMER BLEND PREDICTION - CLI INTERFACE")
    print("="*60)
    print(f"Blend: {blend_string}")
    print(f"Output prefix: {output_prefix}")
    print(f"Model directory: {model_dir}")
    
    try:
        # Step 1: Create blend data from blend string
        blend_df = create_blend_data_from_string(blend_string)
        print(f"Created blend data with shape: {blend_df.shape}")
        
        # NEW RULE: Check if blend contains only polymers with known max_L values
        print("\nChecking if blend contains only polymers with known max_L values...")
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
                    print(f"  Found {grade}: max_L = {max_L:.2f}, t0 = {t0:.2f}")
                else:
                    all_polymers_have_max_L = False
                    print(f"  {grade}: Missing max_L or t0 values")
            else:
                all_polymers_have_max_L = False
                print(f"  {grade}: Not found in polymer reference")
        
        # Check if all polymers have max_L and total fraction is 1.0
        if all_polymers_have_max_L and abs(total_fraction - 1.0) < 0.01 and len(polymer_max_L_values) > 0:
            # Calculate weighted averages based on volume fractions
            weighted_max_L = 0.0
            weighted_t0 = 0.0
            
            for i, (material, grade, fraction) in enumerate(blend_components):
                weighted_max_L += fraction * polymer_max_L_values[i]
                weighted_t0 += fraction * polymer_t0_values[i]
            
            print(f"\nUsing weighted average of known polymer values:")
            print(f"  Weighted max_L: {weighted_max_L:.2f}")
            print(f"  Weighted t0: {weighted_t0:.2f}")
            
            # Use the weighted averages instead of model prediction
            max_L_pred = weighted_max_L
            t0_pred = weighted_t0
            
            print(f"\nFinal Properties (from known polymer values):")
            print(f"Max_L (Disintegration Level): {max_L_pred:.2f}")
            print(f"t0 (Time to 50%): {t0_pred:.2f} days")
            
            # Skip model prediction and go directly to k0 calculation
            skip_model_prediction = True
        else:
            print(f"  Not all polymers have known max_L values or total fraction != 1.0, using model prediction")
            skip_model_prediction = False
        
        # NEW RULE-BASED APPROACH: Calculate t0 using inverse rule of mixtures and k0 as highest in blend
        print("\nChecking for PLA-containing blends for new rule-based approach...")
        has_PLA = False
        blend_t0_values = []
        blend_fractions = []
        polymer_k0_values = []
        polymer_fractions_for_k0 = []
        
        for material, grade, fraction in blend_components:
            # Find the polymer in the reference
            polymer_row = polymer_props[polymer_props['Polymer Grade 1'] == grade]
            if not polymer_row.empty:
                max_L = polymer_row.iloc[0]['property1']
                t0 = polymer_row.iloc[0]['property2']
                
                # Check if it's PLA
                if 'PLA' in material.upper():
                    has_PLA = True
                    print(f"  Found PLA: {material} ({grade})")
                
                # Store t0 and fraction for inverse rule of mixtures calculation
                if pd.notna(t0) and t0 > 0:
                    blend_t0_values.append(t0)
                    blend_fractions.append(fraction)
                
                # Calculate k0 for this polymer and store if fraction >= 15%
                if pd.notna(max_L) and pd.notna(t0) and max_L > 0 and t0 > 0 and fraction >= 0.15:
                    polymer_k0 = calculate_polymer_k0(max_L, t0)
                    polymer_k0_values.append(polymer_k0)
                    polymer_fractions_for_k0.append(fraction)
                    print(f"  {grade}: max_L = {max_L:.2f}, t0 = {t0:.2f}, k0 = {polymer_k0:.4f}, fraction = {fraction:.2f}")
        
        # Apply new rule-based approach if PLA is present and we have valid data
        if has_PLA and len(blend_t0_values) > 0 and len(polymer_k0_values) > 0:
            print(f"\nNew rule-based approach applies (PLA present):")
            
            # Step 1: Calculate t0 using inverse rule of mixtures
            # Inverse rule: 1/t0_blend = Σ(fraction_i / t0_i)
            inverse_t0_sum = 0.0
            total_fraction = 0.0
            
            for i in range(len(blend_t0_values)):
                inverse_t0_sum += blend_fractions[i] / blend_t0_values[i]
                total_fraction += blend_fractions[i]
            
            if inverse_t0_sum > 0:
                t0_pred = total_fraction / inverse_t0_sum
            else:
                # Fallback to weighted average if inverse calculation fails
                weighted_t0 = 0.0
                for i in range(len(blend_t0_values)):
                    weighted_t0 += blend_fractions[i] * blend_t0_values[i]
                t0_pred = weighted_t0 / total_fraction if total_fraction > 0 else 50.0
            
            # Step 2: Find the highest k0 among polymers with ≥15% volume fraction
            if len(polymer_k0_values) > 0:
                max_k0_idx = np.argmax(polymer_k0_values)
                k0_pred = polymer_k0_values[max_k0_idx]
                k0_polymer_fraction = polymer_fractions_for_k0[max_k0_idx]
                print(f"  Selected highest k0: {k0_pred:.4f} (fraction: {k0_polymer_fraction:.2f})")
            else:
                k0_pred = 0.1  # Default if no valid k0 values
                print(f"  No valid k0 values found, using default: {k0_pred}")
            
            # Step 3: Calculate max_L using sigmoid equation at t=200
            # Target y value at t=200: Let's use 95% as the target
            target_y_at_200 = 95.0
            max_L_pred = calculate_max_L_from_k0_t0(k0_pred, t0_pred, target_y_at_200, 200.0)
            
            print(f"\nFinal Properties (New Rule-Based Approach):")
            print(f"t0 (Inverse Rule of Mixtures): {t0_pred:.2f} days")
            print(f"k0 (Highest in blend): {k0_pred:.4f}")
            print(f"Max_L (Calculated at t=200): {max_L_pred:.2f}")
            
            # Skip model prediction and go directly to k0 calculation
            skip_model_prediction = True
        else:
            print(f"  New rule-based approach does not apply:")
            print(f"    - PLA present: {has_PLA}")
            print(f"    - Valid t0 values: {len(blend_t0_values)}")
            print(f"    - Valid k0 values: {len(polymer_k0_values)}")
            if not skip_model_prediction:
                print(f"  Using model prediction")
        
        # Step 2: Process blend features using the same process as training
        print("\nProcessing blend features...")
        # Save blend data to temporary file first
        temp_input_file = f"{output_prefix}_input.csv"
        temp_features_file = f"{output_prefix}_features.csv"
        blend_df.to_csv(temp_input_file, index=False)
        processed_df = process_blend_features(temp_input_file, temp_features_file)
        print(f"Processed blend features shape: {processed_df.shape}")
        
        # Step 3: Prepare features for prediction
        print("\nProcessed blend feature columns:", list(processed_df.columns))
        features, feature_order = prepare_features_for_prediction(processed_df)
        
        # Step 4: Load the trained model and make predictions (if needed)
        if not skip_model_prediction:
            print("\nLoading trained model...")
            dlo = DifferentiableLabelOptimizer(device='cpu')
            metadata = dlo.load_model(save_dir=model_dir, model_name='dlo_model')
            
            # Step 5: Make predictions
            print("\nMaking predictions...")
            predictions = dlo.predict(features, use_scaler=True)
            
            # Extract predictions
            max_L_pred = predictions[0, 0]  # property1 = max_L
            t0_pred = predictions[0, 1]     # property2 = t0
            
            print(f"\nPredicted Properties:")
            print(f"Max_L (Disintegration Level): {max_L_pred:.2f}")
            print(f"t0 (Time to 50%): {t0_pred:.2f} days")
        else:
            print("\nSkipping model prediction - using known polymer values")
        
        # Step 6: Calculate k0 values
        print("\nCalculating rate constants...")
        
        if skip_model_prediction and has_PLA:
            # For the new rule-based approach, we already have k0_pred
            k0_disintegration = k0_pred
            k0_biodegradation = k0_pred  # Use same k0 for biodegradation
            print(f"Using rule-based k0 values:")
            print(f"k0 (Disintegration): {k0_disintegration:.4f}")
            print(f"k0 (Biodegradation): {k0_biodegradation:.4f}")
        else:
            # Use the original approach for non-PLA blends or when model prediction is used
            # Determine majority polymer behavior for k0 selection
            # Load polymer properties to check max_L values
            polymer_props = pd.read_csv('polymer_properties_reference.csv')
            
            # Get the majority polymer from the blend
            blend_components = parse_blend_string(blend_string)
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
                print(f"Majority polymer '{majority_polymer}' has max_L = {majority_max_L:.1f} ({'high' if majority_high_disintegration else 'low'} disintegration)")
            else:
                print(f"Could not determine majority polymer behavior, using default k0 selection")
            
            k0_disintegration = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred, t_max=200.0, 
                                                               majority_polymer_high_disintegration=majority_high_disintegration)
            k0_biodegradation = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred * 2.0, t_max=400.0, 
                                                               majority_polymer_high_disintegration=majority_high_disintegration)
            
            print(f"k0 (Disintegration): {k0_disintegration:.4f}")
            print(f"k0 (Biodegradation): {k0_biodegradation:.4f}")
        
        # Step 7: Generate sigmoid curves
        print("\nGenerating sigmoid curves...")
        
        # Create output directory for this prediction
        prediction_output_dir = f"test_results/{output_prefix}"
        os.makedirs(prediction_output_dir, exist_ok=True)
        
        # Disintegration curves (200 days)
        disintegration_df = generate_sigmoid_curves(
            np.array([max_L_pred]), 
            np.array([t0_pred]), 
            np.array([k0_disintegration]), 
            days=200, 
            curve_type='disintegration',
            save_dir=prediction_output_dir
        )
        
        # Biodegradation curves (400 days, t0 doubled)
        biodegradation_df = generate_sigmoid_curves(
            np.array([max_L_pred]), 
            np.array([t0_pred * 2.0]), 
            np.array([k0_biodegradation]), 
            days=400, 
            curve_type='biodegradation',
            save_dir=prediction_output_dir
        )
        
        # Step 8: Save detailed results
        print("\nSaving detailed results...")
        
        # Create results summary
        results_summary = {
            'Blend': blend_string,
            'Max_L_Predicted': max_L_pred,
            't0_Predicted': t0_pred,
            'k0_Disintegration': k0_disintegration,
            'k0_Biodegradation': k0_biodegradation,
            'Number_of_Features': len(feature_order)
        }
        
        # Save results summary
        results_file = os.path.join(prediction_output_dir, f"{output_prefix}_results.csv")
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv(results_file, index=False)
        print(f"Results summary saved to: {results_file}")
        
        # Save feature values
        feature_file = os.path.join(prediction_output_dir, f"{output_prefix}_feature_values.csv")
        feature_df = pd.DataFrame({
            'Feature': feature_order,
            'Value': features[0]
        })
        feature_df.to_csv(feature_file, index=False)
        print(f"Feature values saved to: {feature_file}")
        
        # Sigmoid curves are now saved directly to the prediction output directory
        
        # Clean up temporary files
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        if os.path.exists(temp_features_file):
            os.remove(temp_features_file)
        
        print("\n" + "="*60)
        print("PREDICTION COMPLETE")
        print("="*60)
        print(f"Files generated in: {prediction_output_dir}")
        print(f"- {os.path.basename(results_file)} (summary)")
        print(f"- {os.path.basename(feature_file)} (feature values)")
        print(f"- {output_prefix}_disintegration_curves.csv (disintegration data)")
        print(f"- {output_prefix}_disintegration_curves.png (disintegration plot)")
        print(f"- {output_prefix}_biodegradation_curves.csv (biodegradation data)")
        print(f"- {output_prefix}_biodegradation_curves.png (biodegradation plot)")
        
        return results_summary, disintegration_df, biodegradation_df
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your blend specification and try again.")
        return None, None, None

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Predict polymer blend properties')
    parser.add_argument('blend_string', type=str, help='Blend string like "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5"')
    parser.add_argument('--output_prefix', type=str, default='cli_prediction', help='Output file prefix')
    parser.add_argument('--model_dir', type=str, default='models/v1/', help='Model directory')
    
    args = parser.parse_args()
    
    # Run prediction
    results, disintegration_df, biodegradation_df = predict_blend(
        args.blend_string, 
        args.output_prefix, 
        args.model_dir
    )
    
    if results:
        print(f"\n✅ Prediction completed successfully!")
        print(f"Max_L: {results['Max_L_Predicted']:.2f}")
        print(f"t0: {results['t0_Predicted']:.2f} days")
        print(f"k0 (Disintegration): {results['k0_Disintegration']:.4f}")
        print(f"k0 (Biodegradation): {results['k0_Biodegradation']:.4f}")
    else:
        print(f"\n❌ Prediction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 