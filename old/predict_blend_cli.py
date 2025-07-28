#!/usr/bin/env python3
"""
Command-line interface and Streamlit app for polymer blend prediction.
Usage: 
  CLI: python predict_blend_cli.py "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5"
  Streamlit: streamlit run predict_blend_cli.py
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import Streamlit only when needed
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

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
        'Thickness_certification': '',
        'wa': 0.0,
        'wvtr': 0.0,
        'Xc': 0.0,
        'Tg': 0.0,
        'ts': 0.0,
        'eab': 0.0,
        'enzyme kinetics': 'med'
    }
    
    # Fill in polymer data and collect enzyme kinetics and physical properties for majority rule
    enzyme_kinetics_data = []
    physical_properties_data = []
    
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
            
            # Collect enzyme kinetics data for majority rule
            if 'enzyme kinetics' in polymer_row:
                enzyme_kinetics_data.append({
                    'kinetics': polymer_row['enzyme kinetics'],
                    'fraction': fraction,
                    'polymer_grade': grade
                })
            
            # Collect physical properties data for majority rule
            physical_props = ['wa', 'wvtr', 'Xc', 'Tg', 'ts', 'eab']
            for prop in physical_props:
                if prop in polymer_row:
                    physical_properties_data.append({
                        'property': prop,
                        'value': polymer_row[prop],
                        'fraction': fraction
                    })
            
            print(f"  Found {polymer_name} {grade}: SMILES = {polymer_row['SMILES1']}, fraction = {fraction}, kinetics = {polymer_row.get('enzyme kinetics', 'unknown')}")
        else:
            print(f"  Warning: Could not find polymer grade '{grade}' in reference")
            blend_data[f'Polymer Grade {i+1}'] = grade
            blend_data[f'vol_fraction{i+1}'] = fraction
    
    # Determine majority polymer (highest volume fraction, or highest max_L if equal)
    majority_polymer = None
    majority_fraction = 0.0
    majority_max_L = -1  # Track max_L for tie-breaking
    
    for polymer_name, grade, fraction in polymers:
        # Get max_L for this polymer
        polymer_mask = (polymer_ref['Polymer Grade 1'] == grade)
        polymer_max_L = 0.0
        if polymer_mask.any():
            polymer_max_L = polymer_ref[polymer_mask].iloc[0]['property1']
        
        # Update majority if fraction is higher, or if equal fraction but higher max_L
        if (fraction > majority_fraction) or (fraction == majority_fraction and polymer_max_L > majority_max_L):
            majority_fraction = fraction
            majority_polymer = grade
            majority_max_L = polymer_max_L
    
    # Calculate physical properties using rule of mixtures
    print("  Calculating physical properties using rule of mixtures:")
    
    # Initialize property sums for rule of mixtures
    property_sums = {
        'wa': 0.0,      # Rule of mixtures
        'wvtr': 0.0,    # Rule of mixtures
        'ts': 0.0,      # Inverse rule of mixtures
        'eab': 0.0,     # Inverse rule of mixtures
        'Tg': 0.0,      # Inverse rule of mixtures (convert to K first)
        'Xc': 0.0       # Majority rule (will be set later)
    }
    
    # Track inverse sums for inverse rule of mixtures
    inverse_sums = {
        'ts': 0.0,
        'eab': 0.0,
        'Tg': 0.0
    }
    
    # Collect all polymer data for rule of mixtures
    polymer_data = []
    for polymer_name, grade, fraction in polymers:
        polymer_mask = (polymer_ref['Polymer Grade 1'] == grade)
        if polymer_mask.any():
            polymer_row = polymer_ref[polymer_mask].iloc[0]
            polymer_data.append({
                'grade': grade,
                'fraction': fraction,
                'row': polymer_row
            })
    
    # Apply rule of mixtures for wa and wvtr
    for data in polymer_data:
        fraction = data['fraction']
        polymer_row = data['row']
        
        # Rule of mixtures (linear combination)
        for prop in ['wa', 'wvtr']:
            if prop in polymer_row and polymer_row[prop] is not None:
                property_sums[prop] += fraction * polymer_row[prop]
        
        # Inverse rule of mixtures (for properties where lower is better)
        for prop in ['ts', 'eab']:
            if prop in polymer_row and polymer_row[prop] is not None and polymer_row[prop] > 0:
                inverse_sums[prop] += fraction / polymer_row[prop]
        
        # Inverse rule of mixtures for Tg (convert to K first)
        if 'Tg' in polymer_row and polymer_row['Tg'] is not None:
            # Convert Tg from Celsius to Kelvin
            tg_kelvin = polymer_row['Tg'] + 273.15
            if tg_kelvin > 0:
                inverse_sums['Tg'] += fraction / tg_kelvin
    
    # Set the calculated properties
    blend_data['wa'] = property_sums['wa']
    blend_data['wvtr'] = property_sums['wvtr']
    
    # Calculate inverse rule of mixtures
    for prop in ['ts', 'eab']:
        if inverse_sums[prop] > 0:
            blend_data[prop] = 1.0 / inverse_sums[prop]
        else:
            blend_data[prop] = 0.0
    
    # Calculate inverse rule of mixtures for Tg (convert back to Celsius)
    if inverse_sums['Tg'] > 0:
        tg_kelvin_result = 1.0 / inverse_sums['Tg']
        blend_data['Tg'] = tg_kelvin_result - 273.15  # Convert back to Celsius
    else:
        blend_data['Tg'] = 0.0
    
    # Use majority rule for Xc only
    if majority_polymer:
        majority_mask = (polymer_ref['Polymer Grade 1'] == majority_polymer)
        if majority_mask.any():
            majority_row = polymer_ref[majority_mask].iloc[0]
            if 'Xc' in majority_row:
                blend_data['Xc'] = majority_row['Xc']
            else:
                blend_data['Xc'] = 0.0
            print(f"  Using majority rule for Xc from '{majority_polymer}' (fraction: {majority_fraction})")
        else:
            blend_data['Xc'] = 0.0
            print(f"  Warning: Majority polymer '{majority_polymer}' not found in reference, using default values")
    else:
        blend_data['Xc'] = 0.0
        print(f"  No polymers found, using default values for Xc")
    
    print(f"  Rule of mixtures results:")
    print(f"    wa: {blend_data['wa']:.4f}")
    print(f"    wvtr: {blend_data['wvtr']:.4f}")
    print(f"    ts (inverse): {blend_data['ts']:.4f}")
    print(f"    eab (inverse): {blend_data['eab']:.4f}")
    print(f"    Tg (inverse): {blend_data['Tg']:.4f}°C")
    print(f"    Xc (majority): {blend_data['Xc']:.4f}")
    
    # Apply majority rule for enzyme kinetics
    if enzyme_kinetics_data:
        # Group by kinetics and sum fractions, also track max_L for tie-breaking
        kinetics_totals = {}
        kinetics_max_L = {}  # Track max_L for each kinetics type
        
        for data in enzyme_kinetics_data:
            kinetics = data['kinetics']
            fraction = data['fraction']
            polymer_grade = data['polymer_grade']
            
            # Get max_L for this polymer
            polymer_mask = (polymer_ref['Polymer Grade 1'] == polymer_grade)
            polymer_max_L = 0.0
            if polymer_mask.any():
                polymer_max_L = polymer_ref[polymer_mask].iloc[0]['property1']
            
            if kinetics in kinetics_totals:
                kinetics_totals[kinetics] += fraction
                # Update max_L if this polymer has higher max_L
                if polymer_max_L > kinetics_max_L[kinetics]:
                    kinetics_max_L[kinetics] = polymer_max_L
            else:
                kinetics_totals[kinetics] = fraction
                kinetics_max_L[kinetics] = polymer_max_L
        
        # Find the kinetics with the highest total fraction, or highest max_L if equal
        max_fraction = max(kinetics_totals.values())
        max_fraction_kinetics = [k for k, v in kinetics_totals.items() if v == max_fraction]
        
        if len(max_fraction_kinetics) == 1:
            majority_kinetics = max_fraction_kinetics[0]
        else:
            # If equal fractions, choose the one with highest max_L
            majority_kinetics = max(max_fraction_kinetics, key=lambda k: kinetics_max_L.get(k, 0))
        
        blend_data['enzyme kinetics'] = majority_kinetics
        print(f"  Majority enzyme kinetics: {majority_kinetics} (fractions: {kinetics_totals})")
    else:
        # Default if no kinetics data found
        blend_data['enzyme kinetics'] = 'med'
        print(f"  No enzyme kinetics data found, using default: med")
    
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
    
    # One-hot encode categorical features to match training
    from modules.utils import one_hot_encode_categorical_features
    processed_df = one_hot_encode_categorical_features(processed_df)
    
    print(f"Columns after filtering: {list(processed_df.columns)}")
    print(f"Number of columns after filtering: {len(processed_df.columns)}")
    
    # Get the complete feature order including categorical features
    from modules.utils import get_categorical_feature_names
    
    # Molecular features (90 features)
    molecular_features = [
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
    
    # Physical properties (6 features)
    physical_properties = ['Xc', 'wa', 'wvtr', 'eab', 'ts', 'Tg']
    
    # Categorical features (63 features)
    categorical_features = get_categorical_feature_names()
    
    # Complete feature order
    feature_order = molecular_features + physical_properties + categorical_features
    
    # Extract only the feature columns in the correct order
    features = processed_df[feature_order].values
    
    print(f"Feature shape: {features.shape}")
    print(f"Number of features: {len(feature_order)}")
    
    return features, feature_order

def main(args=None):
    """Main function with command-line interface."""
    if args is None:
        parser = argparse.ArgumentParser(description='Predict properties for a polymer blend')
        parser.add_argument('blend', type=str, 
                           help='Blend specification (e.g., "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5")')
        parser.add_argument('--output_prefix', type=str, default='blend_prediction',
                           help='Prefix for output files (default: blend_prediction)')
        parser.add_argument('--model_dir', type=str, default='models/v1/',
                           help='Directory containing trained model (default: models/v1/)')
        
        args = parser.parse_args()
    
    print("="*60)
    print("POLYMER BLEND PREDICTION - COMMAND LINE INTERFACE")
    print("="*60)
    print(f"Blend: {args.blend}")
    print(f"Output prefix: {args.output_prefix}")
    print(f"Model directory: {args.model_dir}")
    
    try:
        # Step 1: Create blend data from command line argument
        blend_df = create_blend_data_from_string(args.blend)
        print(f"Created blend data with shape: {blend_df.shape}")
        
        # Step 2: Process blend features using the same process as training
        print("\nProcessing blend features...")
        # Save blend data to temporary file first
        temp_input_file = f"{args.output_prefix}_input.csv"
        temp_features_file = f"{args.output_prefix}_features.csv"
        blend_df.to_csv(temp_input_file, index=False)
        processed_df = process_blend_features(temp_input_file, temp_features_file)
        print(f"Processed blend features shape: {processed_df.shape}")
        
        # Step 3: Prepare features for prediction
        print("\nProcessed blend feature columns:", list(processed_df.columns))
        features, feature_order = prepare_features_for_prediction(processed_df)
        
        # Step 4: Load the trained model
        print("\nLoading trained model...")
        dlo = DifferentiableLabelOptimizer(device='cpu')
        metadata = dlo.load_model(save_dir=args.model_dir, model_name='dlo_model')
        
        # Step 5: Make predictions
        print("\nMaking predictions...")
        predictions = dlo.predict(features, use_scaler=True)
        
        # Extract predictions
        max_L_pred = predictions[0, 0]  # property1 = max_L
        t0_pred = predictions[0, 1]     # property2 = t0
        
        print(f"\nPredicted Properties:")
        print(f"Max_L (Disintegration Level): {max_L_pred:.2f}")
        print(f"t0 (Time to 50%): {t0_pred:.2f} days")
        
        # Step 6: Calculate k0 values
        print("\nCalculating rate constants...")
        
        # Determine majority polymer behavior for k0 selection
        # Load polymer properties to check max_L values
        polymer_props = pd.read_csv('polymer_properties_reference.csv')
        
        # Get the majority polymer from the blend
        blend_components = parse_blend_string(args.blend)
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
        prediction_output_dir = f"test_results/{args.output_prefix}"
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
            'Blend': args.blend,
            'Max_L_Predicted': max_L_pred,
            't0_Predicted': t0_pred,
            'k0_Disintegration': k0_disintegration,
            'k0_Biodegradation': k0_biodegradation,
            'Number_of_Features': len(feature_order)
        }
        
        # Save results summary
        results_file = os.path.join(prediction_output_dir, f"{args.output_prefix}_results.csv")
        results_df = pd.DataFrame([results_summary])
        results_df.to_csv(results_file, index=False)
        print(f"Results summary saved to: {results_file}")
        
        # Save feature values
        feature_file = os.path.join(prediction_output_dir, f"{args.output_prefix}_feature_values.csv")
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
        print(f"- {args.output_prefix}_disintegration_curves.csv (disintegration data)")
        print(f"- {args.output_prefix}_disintegration_curves.png (disintegration plot)")
        print(f"- {args.output_prefix}_biodegradation_curves.csv (biodegradation data)")
        print(f"- {args.output_prefix}_biodegradation_curves.png (biodegradation plot)")
        
        return results_summary, disintegration_df, biodegradation_df
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Please check your blend specification and try again.")
        return None, None, None


def run_streamlit_app():
    """Run the Streamlit app for blend prediction."""
    
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit is not available. Please install it with: pip install streamlit")
        return
    
    # THIS MUST BE FIRST!
    st.set_page_config(
        page_title="Polymer Blend Prediction Model",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Set dark background for the whole app
    st.markdown("""
    <style>
    body, .stApp, .main, .block-container, .css-18e3th9, .css-1d391kg, .css-1v0mbdj, .css-1dp5vir, .css-1cpxqw2, .css-ffhzg2, .css-1outpf7, .css-1lcbmhc, .css-1vq4p4l, .css-1wrcr25, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb, .css-1vzeuhh, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb, .css-1vzeuhh, .css-1b2g3b5, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0, .css-1vzeuhh, .css-1cypcdb {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stMetric, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stAlert, .stSuccess, .stError, .stWarning, .stInfo, .stRadio, .stCheckbox, .stSlider, .stTextInput, .stTextArea, .stDateInput, .stTimeInput, .stColorPicker, .stFileUploader, .stImage, .stAudio, .stVideo, .stJson, .stCode, .stException, .stHelp, .stExpander, .stTabs, .stTab, .stSidebar, .stSidebarContent, .stSidebarNav, .stSidebarNavItem, .stSidebarNavLink, .stSidebarNavLinkActive, .stSidebarNavLinkInactive, .stSidebarNavLinkSelected, .stSidebarNavLinkUnselected, .stSidebarNavLinkDisabled, .stSidebarNavLinkIcon, .stSidebarNavLinkLabel, .stSidebarNavLinkLabelText, .stSidebarNavLinkLabelIcon, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled {
        color: #FFFFFF !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Aggressive green button CSS
    st.markdown("""
    <style>
    button[kind='primary'],
    button[kind='primary']:hover,
    button[kind='primary']:focus,
    button[kind='primary']:active,
    button[kind='primary']:focus-visible,
    button[kind='primary']:focus-within,
    button[kind='primary']:focus:not(:active),
    .stButton > button[kind='primary'],
    .stButton > button[kind='primary']:hover,
    .stButton > button[kind='primary']:focus,
    .stButton > button[kind='primary']:active,
    .stButton > button[kind='primary']:focus-visible,
    .stButton > button[kind='primary']:focus-within,
    .stButton > button[kind='primary']:focus:not(:active),
    div[data-testid='stButton'] > button[kind='primary'],
    div[data-testid='stButton'] > button[kind='primary']:hover,
    div[data-testid='stButton'] > button[kind='primary']:focus,
    div[data-testid='stButton'] > button[kind='primary']:active,
    div[data-testid='stButton'] > button[kind='primary']:focus-visible,
    div[data-testid='stButton'] > button[kind='primary']:focus-within,
    div[data-testid='stButton'] > button[kind='primary']:focus:not(:active) {
        background-color: #2E8B57 !important;
        border-color: #2E8B57 !important;
        color: white !important;
        font-weight: 600 !important;
        outline: none !important;
        box-shadow: none !important;
        border: 2px solid #2E8B57 !important;
    }
    button[kind='primary']:hover,
    .stButton > button[kind='primary']:hover,
    div[data-testid='stButton'] > button[kind='primary']:hover {
        background-color: #3CB371 !important;
        border-color: #3CB371 !important;
        border: 2px solid #3CB371 !important;
    }
    button[kind='primary']:active,
    .stButton > button[kind='primary']:active,
    div[data-testid='stButton'] > button[kind='primary']:active {
        background-color: #228B22 !important;
        border-color: #228B22 !important;
        border: 2px solid #228B22 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Force all Streamlit widget labels to be white
    st.markdown("""
    <style>
    label, .stSelectbox label, .stNumberInput label, .css-1cpxqw2, .css-1v3fvcr, .css-1q8dd3e, .css-1r6slb0 {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'prediction_generated' not in st.session_state:
        st.session_state['prediction_generated'] = False
    
    # Professional header
    st.markdown('<h1 class="main-header">🧬 Polymer Blend Prediction Model</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced DLO-based modeling for polymer blend biodegradation prediction</p>', unsafe_allow_html=True)
    
    # Load available polymers from reference
    try:
        polymer_ref = pd.read_csv('polymer_properties_reference.csv')
        polymer_options = []
        for _, row in polymer_ref.iterrows():
            grade = str(row.get('Polymer Grade 1', ''))
            if pd.notna(grade) and grade.strip():
                polymer_options.append({
                    'display': grade,
                    'grade': grade,
                    'row_data': row
                })
    except Exception as e:
        st.error(f"Error loading polymer reference: {str(e)}")
        return
    
    # Main interface - 35% left for selection, 65% right for results
    col1, col2 = st.columns([35, 65])
    
    with col1:
        st.markdown("### Blend Configuration")
        
        selected_polymers = []
        volume_fractions = []
        
        # Polymer selection interface
        for i in range(5):
            cols = st.columns([3, 1])
            
            with cols[0]:
                polymer_selection = st.selectbox(
                    f"Polymer {i+1}",
                    options=[""] + [opt['display'] for opt in polymer_options],
                    key=f"polymer_{i}"
                )
            
            with cols[1]:
                vol_frac = st.number_input(
                    "Fraction",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.0,
                    step=0.05,
                    key=f"vol_frac_{i}"
                )
            
            if polymer_selection and vol_frac > 0:
                selected_polymer = next(opt for opt in polymer_options if opt['display'] == polymer_selection)
                selected_polymers.append(selected_polymer)
                volume_fractions.append(vol_frac)
        
        # Volume fraction validation
        total_vol_frac = sum(volume_fractions)
        if selected_polymers:
            if abs(total_vol_frac - 1.0) <= 0.01:
                st.success(f"✅ Total: {total_vol_frac:.2f}")
            else:
                st.error(f"❌ Total: {total_vol_frac:.2f} (should be 1.0)")
        
        # Model directory selection
        model_dir = st.selectbox(
            "Model Directory",
            options=["models/v1/", "models/v2/", "models/v3/"],
            index=0,
            help="Select the directory containing the trained model"
        )
        
        # Output prefix
        output_prefix = st.text_input(
            "Output Prefix",
            value="blend_prediction",
            help="Prefix for output files"
        )
        
        # Generate button
        generate_clicked = st.button("🚀 Generate Prediction", type="primary")
        
        if generate_clicked:
            if not selected_polymers:
                st.error("Please select at least one polymer.")
            elif abs(total_vol_frac - 1.0) > 0.01:
                st.error("Volume fractions must sum to 1.0.")
            else:
                # Create blend string in the format expected by parse_blend_string
                blend_parts = []
                for polymer, vol_frac in zip(selected_polymers, volume_fractions):
                    # For this dataset, the polymer name and grade are the same
                    blend_parts.extend([polymer['grade'], polymer['grade'], str(vol_frac)])
                blend_string = ",".join(blend_parts)
                
                try:
                    # Create temporary output directory
                    temp_output_dir = f"test_results/{output_prefix}"
                    os.makedirs(temp_output_dir, exist_ok=True)
                    
                    # Debug: Show the blend string being created
                    st.info(f"Creating blend string: {blend_string}")
                    
                    # Run prediction using the existing CLI logic
                    with st.spinner("Generating prediction..."):
                        # Create a mock args object for the CLI function
                        class MockArgs:
                            def __init__(self, blend, output_prefix, model_dir):
                                self.blend = blend
                                self.output_prefix = output_prefix
                                self.model_dir = model_dir
                        
                        mock_args = MockArgs(blend_string, output_prefix, model_dir)
                        
                        # Run the existing CLI logic
                        results, dis_df, bio_df = main(mock_args)
                        
                        if results is not None:
                            # Store results in session state
                            st.session_state['prediction_generated'] = True
                            st.session_state['prediction_results'] = results
                            st.session_state['prediction_output_dir'] = temp_output_dir
                        else:
                            st.error("Prediction failed. Check the console for details.")
                        
                except Exception as e:
                    st.error(f"Error generating prediction: {str(e)}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")

    with col2:
        st.markdown("### Results")
        
        if st.session_state.get('prediction_generated', False):
            results = st.session_state['prediction_results']
            output_dir = st.session_state['prediction_output_dir']
            
            # Display prediction parameters
            st.markdown("#### Prediction Parameters")
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Max L", f"{results['Max_L_Predicted']:.2f}")
            with col_b:
                st.metric("t0", f"{results['t0_Predicted']:.2f}")
            with col_c:
                st.metric("k0 (Disintegration)", f"{results['k0_Disintegration']:.4f}")
            with col_d:
                st.metric("k0 (Biodegradation)", f"{results['k0_Biodegradation']:.4f}")
            
            # Display sigmoid curves
            st.markdown("#### Sigmoid Curves")
            
            # Load and display the generated plots
            disintegration_png = os.path.join(output_dir, "sigmoid_disintegration_curves.png")
            biodegradation_png = os.path.join(output_dir, "sigmoid_biodegradation_curves.png")
            
            if os.path.exists(disintegration_png):
                st.image(disintegration_png, use_container_width=True, caption="Disintegration Curves")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(disintegration_png, "rb") as file:
                        st.download_button(
                            label="📥 Download Disintegration Plot",
                            data=file.read(),
                            file_name=f"{output_prefix}_disintegration_curves.png",
                            mime="image/png"
                        )
                
                with col2:
                    disintegration_csv = os.path.join(output_dir, "sigmoid_disintegration_curves.csv")
                    if os.path.exists(disintegration_csv):
                        with open(disintegration_csv, "rb") as file:
                            st.download_button(
                                label="📥 Download Disintegration CSV",
                                data=file.read(),
                                file_name=f"{output_prefix}_disintegration_curves.csv",
                                mime="text/csv"
                            )
            
            if os.path.exists(biodegradation_png):
                st.image(biodegradation_png, use_container_width=True, caption="Biodegradation Curves")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(biodegradation_png, "rb") as file:
                        st.download_button(
                            label="📥 Download Biodegradation Plot",
                            data=file.read(),
                            file_name=f"{output_prefix}_biodegradation_curves.png",
                            mime="image/png"
                        )
                
                with col2:
                    biodegradation_csv = os.path.join(output_dir, "sigmoid_biodegradation_curves.csv")
                    if os.path.exists(biodegradation_csv):
                        with open(biodegradation_csv, "rb") as file:
                            st.download_button(
                                label="📥 Download Biodegradation CSV",
                                data=file.read(),
                                file_name=f"{output_prefix}_biodegradation_curves.csv",
                                mime="text/csv"
                            )
            
            # Display blend information
            st.markdown("#### Blend Information")
            st.info(f"**Blend:** {results['Blend']}")
            st.info(f"**Output Directory:** {output_dir}")
            st.info(f"**Number of Features:** {results['Number_of_Features']}")
            
        else:
            st.markdown("""
            **Select polymers and generate a prediction to see results here.**
            
            The prediction results and sigmoid curves will appear in this area once you click "Generate Prediction".
            """)


if __name__ == "__main__":
    # Check if running as Streamlit app
    if len(sys.argv) > 1 and sys.argv[1] == '--streamlit':
        run_streamlit_app()
    else:
        # Run as CLI
        results, dis_df, bio_df = main() 