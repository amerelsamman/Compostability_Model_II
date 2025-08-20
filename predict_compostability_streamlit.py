#!/usr/bin/env python3
"""
Streamlit web application for polymer blend prediction.
Usage: streamlit run predict_blend_cli_old_majorityrule.py
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import warnings
import random
warnings.filterwarnings('ignore')

# Add modules to path
sys.path.append('modules')

from modules_home.optimizer import DifferentiableLabelOptimizer
from modules_home.blend_feature_extractor import process_blend_features
from modules_home.utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves, generate_quintic_biodegradation_curve

# Streamlit import
import streamlit as st

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

def run_streamlit_app():
    """Run the Streamlit app for blend prediction."""
    
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
            options=["models/eol/v3/", "models/eol/v4/"],
            index=1,  # Default to v4 (latest)
            help="Select the directory containing the trained model"
        )
        
        # Thickness input
        thickness_um = st.number_input(
            "Material Thickness (μm)",
            min_value=1.0,
            max_value=1000.0,
            value=50.0,
            step=1.0,
            help="Enter the actual thickness of the material in micrometers (μm). Default is 50μm."
        )
        actual_thickness_mm = thickness_um / 1000.0  # Convert to mm
        
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
                    # Include both material name and grade
                    material = polymer['row_data']['Materials']
                    grade = polymer['grade']
                    blend_parts.extend([material, grade, str(vol_frac)])
                blend_string = ",".join(blend_parts)
                
                try:
                    # Create temporary output directory
                    temp_output_dir = f"test_results/{output_prefix}"
                    os.makedirs(temp_output_dir, exist_ok=True)
                    
                    # Debug: Show the blend string being created
                    st.info(f"Creating blend string: {blend_string}")
                    
                    # Run prediction using the existing CLI logic
                    with st.spinner("Generating prediction..."):
                        # Run the existing prediction logic
                        results, dis_df, bio_df = predict_blend(blend_string, output_prefix, model_dir, actual_thickness_mm)
                        
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
            
            # Display quintic biodegradation curve parameters
            st.markdown("#### Quintic Biodegradation Curve Parameters")
            
            # Load the quintic curve data to extract parameters
            quintic_csv = os.path.join(output_dir, "quintic_biodegradation_curves.csv")
            if os.path.exists(quintic_csv):
                quintic_df = pd.read_csv(quintic_csv)
                
                # Extract constraint points from the data
                day_0 = quintic_df[quintic_df['day'] == 0]['biodegradation'].iloc[0]
                
                # Find closest day value for t0+10 since it might be a float
                t0_plus_10 = results['t0_Predicted'] + 10
                day_values = quintic_df['day'].values
                closest_day_idx = np.argmin(np.abs(day_values - t0_plus_10))
                closest_day = day_values[closest_day_idx]
                day_t0_plus_10 = quintic_df.iloc[closest_day_idx]['biodegradation']
                
                day_400 = quintic_df[quintic_df['day'] == 400]['biodegradation'].iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Constraint Point 1", f"(0, {day_0:.2f}%)")
                with col2:
                    st.metric("Constraint Point 2", f"({closest_day:.0f}, {day_t0_plus_10:.2f}%)")
                with col3:
                    st.metric("Constraint Point 3", f"(400, {day_400:.2f}%)")
                
                # Display the quintic function form
                st.info(f"**Quintic Function:** y = ax⁵ + bx⁴ + cx³ + dx² + ex")
            
            # Display disintegration and biodegradation curves
            st.markdown("#### Disintegration and Biodegradation Curves")
            
            # Load and display the generated plots
            disintegration_png = os.path.join(output_dir, "sigmoid_disintegration_curves.png")
            biodegradation_png = os.path.join(output_dir, "quintic_biodegradation_curves.png")
            
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
                    biodegradation_csv = os.path.join(output_dir, "quintic_biodegradation_curves.csv")
                    if os.path.exists(biodegradation_csv):
                        with open(biodegradation_csv, "rb") as file:
                            st.download_button(
                                label="📥 Download Biodegradation CSV",
                                data=file.read(),
                                file_name=f"{output_prefix}_quintic_biodegradation_curves.csv",
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

def predict_blend(blend_string, output_prefix="streamlit_prediction", model_dir="models/eol/v4/", actual_thickness=None):
    """Main prediction function with optional thickness scaling."""
    print("="*60)
    print("POLYMER BLEND PREDICTION - STREAMLIT INTERFACE")
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

if __name__ == "__main__":
    run_streamlit_app() 