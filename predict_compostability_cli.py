#!/usr/bin/env python3
"""
Command line interface for polymer blend prediction.
Usage: python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5" thickness=50
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import warnings
import random
import argparse
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append('modules')

from train.modules_home.optimizer import DifferentiableLabelOptimizer
from train.modules_home.blend_feature_extractor import process_blend_features
from train.modules_home.utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves, generate_quintic_biodegradation_curve

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
            blend_data[f'Polymer Grade {i+1}'] = grade
            blend_data[f'vol_fraction{i+1}'] = fraction
    
    # No surrogate models needed - using only molecular features and volume fractions
    
    blend_df = pd.DataFrame([blend_data])
    return blend_df

def prepare_features_for_prediction(processed_df):
    """
    Prepare features for prediction using the same process as training.
    """
    print("\nPreparing features for prediction...")
    
    # Filter out metadata columns
    exclude_cols = ['Materials', 'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5']
    processed_df = processed_df.drop(columns=exclude_cols, errors='ignore')
    
    print(f"Columns after filtering: {list(processed_df.columns)}")
    print(f"Number of columns after filtering: {len(processed_df.columns)}")
    
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
    
    print(f"Feature shape: {features.shape}")
    print(f"Number of features: {len(feature_order)}")
    
    return features, feature_order

def predict_blend(blend_string, output_prefix="cli_prediction", model_dir="models/eol/v4/", actual_thickness=None):
    """Main prediction function with optional thickness scaling."""
    print("="*60)
    print("POLYMER BLEND PREDICTION - COMMAND LINE INTERFACE")
    print("="*60)
    print(f"Blend: {blend_string}")
    print(f"Output prefix: {output_prefix}")
    print(f"Model directory: {model_dir}")
    if actual_thickness:
        print(f"Thickness: {actual_thickness} μm")
    
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
            # Check if all polymers are home-compostable (max_L > 90)
            all_home_compostable = all(max_L > 90 for max_L in polymer_max_L_values)
            
            if all_home_compostable:
                # For purely home-compostable blends, use random max_L between 90-95
                print(f"\nAll polymers are home-compostable (max_L > 90) - using random max_L between 90-95")
                import random
                max_L_pred = random.uniform(90.0, 95.0)
                
                # Calculate weighted average t0
                weighted_t0 = 0.0
                for i, (material, grade, fraction) in enumerate(blend_components):
                    weighted_t0 += fraction * polymer_t0_values[i]
                t0_pred = weighted_t0
                
                print(f"  Random max_L: {max_L_pred:.2f}")
                print(f"  Weighted t0: {t0_pred:.2f}")
            else:
                # Calculate weighted averages based on volume fractions for mixed blends
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
        
        # NEW PLA RULE: Check for PLA + compostable polymer rule
        print("\nChecking PLA + compostable polymer rule...")
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
                    print(f"  Found PLA: {material} ({grade})")
                
                # Check if it's a compostable polymer (max_L > 90)
                if max_L > 90:
                    has_compostable_polymer = True
                    compostable_polymer_fraction += fraction
                    print(f"  Found compostable polymer: {grade} (max_L = {max_L:.2f}, fraction = {fraction:.2f})")
                else:
                    # Check if it's non-compostable (max_L < 90) and NOT PLA
                    if 'PLA' not in material.upper():
                        if fraction > 0.20:
                            non_compostable_polymer_fraction += fraction
                            print(f"  Found non-PLA non-compostable polymer > 20%: {grade} (max_L = {max_L:.2f}, fraction = {fraction:.2f})")
                        else:
                            print(f"  Found non-PLA non-compostable polymer ≤ 20%: {grade} (max_L = {max_L:.2f}, fraction = {fraction:.2f})")
                    else:
                        print(f"  Found PLA (non-compostable but doesn't count toward 20% limit): {grade} (max_L = {max_L:.2f}, fraction = {fraction:.2f})")
                
                # Store t0 and fraction for weighted average calculation
                if pd.notna(t0):
                    blend_t0_values.append(t0)
                    blend_fractions.append(fraction)
        
        # Apply PLA rule if conditions are met
        if (has_PLA and has_compostable_polymer and 
            compostable_polymer_fraction >= 0.15 and 
            non_compostable_polymer_fraction <= 0.20):
            
            print(f"\nPLA + compostable polymer rule applies:")
            print(f"  - PLA present: {has_PLA}")
            print(f"  - Compostable polymer fraction: {compostable_polymer_fraction:.2f} (>= 0.15)")
            print(f"  - Non-compostable polymer fraction: {non_compostable_polymer_fraction:.2f} (<= 0.20)")
            
            # Set max_L = random value between 90-95 and calculate weighted average t0
            import random
            max_L_pred = random.uniform(90.0, 95.0)
            weighted_t0 = 0.0
            
            for i in range(len(blend_t0_values)):
                weighted_t0 += blend_fractions[i] * blend_t0_values[i]
            
            t0_pred = weighted_t0
            
            print(f"\nFinal Properties (PLA rule):")
            print(f"Max_L (Disintegration Level): {max_L_pred:.2f}")
            print(f"t0 (Time to 50%): {t0_pred:.2f} days")
            print(f"Random value generated: {max_L_pred:.2f}")
            
            # Skip model prediction and go directly to k0 calculation
            skip_model_prediction = True
        else:
            print(f"  PLA rule does not apply:")
            print(f"    - PLA present: {has_PLA}")
            print(f"    - Compostable polymer present: {has_compostable_polymer}")
            print(f"    - Compostable polymer fraction: {compostable_polymer_fraction:.2f}")
            print(f"    - Non-compostable polymer fraction: {non_compostable_polymer_fraction:.2f}")
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
                                                           majority_polymer_high_disintegration=majority_high_disintegration,
                                                           actual_thickness=actual_thickness)
        k0_biodegradation = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred * 2.0, t_max=400.0, 
                                                           majority_polymer_high_disintegration=majority_high_disintegration,
                                                           actual_thickness=actual_thickness)
        
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
        print(f"- {output_prefix}_quintic_biodegradation_curves.png (biodegradation plot)")
        
        return results_summary, disintegration_df, biodegradation_df
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your blend specification and try again.")
        return None, None, None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict polymer blend biodegradation properties",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5"
  python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5" thickness=50
  python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5" thickness=100 output=my_prediction
        """
    )
    
    parser.add_argument(
        'blend_string',
        type=str,
        help='Blend specification in format "Polymer1, Grade1, Fraction1, Polymer2, Grade2, Fraction2, ..."'
    )
    
    parser.add_argument(
        '--thickness', '-t',
        type=float,
        default=50.0,
        help='Material thickness in micrometers (μm). Default: 50.0'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='cli_prediction',
        help='Output prefix for generated files. Default: cli_prediction'
    )
    
    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        default='models/eol/v4/',
        help='Model directory. Default: models/eol/v4/'
    )
    
    return parser.parse_args()

def main():
    """Main function to handle command line interface."""
    args = parse_arguments()
    
    # Parse thickness from command line arguments
    thickness_um = args.thickness
    actual_thickness_mm = thickness_um / 1000.0  # Convert to mm
    
    print(f"Starting prediction with:")
    print(f"  Blend: {args.blend_string}")
    print(f"  Thickness: {thickness_um} μm ({actual_thickness_mm:.3f} mm)")
    print(f"  Output prefix: {args.output}")
    print(f"  Model directory: {args.model_dir}")
    print()
    
    # Run prediction
    results, dis_df, bio_df = predict_blend(
        args.blend_string, 
        args.output, 
        args.model_dir, 
        actual_thickness_mm
    )
    
    if results is not None:
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Blend: {results['Blend']}")
        print(f"Max_L (Disintegration Level): {results['Max_L_Predicted']:.2f}")
        print(f"t0 (Time to 50%): {results['t0_Predicted']:.2f} days")
        print(f"k0 (Disintegration): {results['k0_Disintegration']:.4f}")
        print(f"k0 (Biodegradation): {results['k0_Biodegradation']:.4f}")
        print(f"Number of Features: {results['Number_of_Features']}")
        print("="*60)
    else:
        print("Prediction failed. Check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 