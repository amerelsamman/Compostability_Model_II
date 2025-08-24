#!/usr/bin/env python3
"""
Streamlit web application for polymer blend prediction.
Uses the new streamlined backend logic with the exact same UI.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: MODEL PREDICTION (using train/modules/ - same as other properties)
# =============================================================================
from train.modules.prediction_engine import predict_blend_property
from train.modules.prediction_utils import PROPERTY_CONFIGS

# =============================================================================
# SECTION 2: CURVE GENERATION (using train/modules_home/ - will become modules_home_curve/)
# =============================================================================
from train.modules_home.curve_generator import generate_compostability_curves

# Streamlit import
import streamlit as st

# For capturing matplotlib plots
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

def load_material_dictionary(dict_path='material-smiles-dictionary.csv'):
    """Load the material-SMILES dictionary."""
    try:
        df = pd.read_csv(dict_path)
        material_dict = {}
        for _, row in df.iterrows():
            key = (row['Material'].strip(), row['Grade'].strip())
            material_dict[key] = row['SMILES'].strip()
        return material_dict
    except Exception as e:
        st.error(f"❌ Error loading material dictionary: {e}")
        return None

# These old functions are no longer needed - we now use the streamlined predict_blend_property from train/modules/

def predict_blend(blend_string, output_prefix="streamlit_prediction", model_dir="train/models/eol/v5/", actual_thickness=None):
    """
    Main prediction function using the EXACT same logic as predict_blend_properties.py.
    The only difference is that this displays curves in Streamlit instead of saving them as files.
    """
    if actual_thickness is None:
        actual_thickness = 0.050  # Default 50 μm
    
    try:
        # Step 1: Load material dictionary (same as predict_blend_properties.py)
        material_dict = load_material_dictionary()
        if material_dict is None:
            return None, None, None, None, None
        
        # Step 2: Parse blend string (same as predict_blend_properties.py)
        parts = [part.strip() for part in blend_string.split(',')]
        polymers = []
        
        i = 0
        while i < len(parts):
            if i + 2 < len(parts):
                material = parts[i]
                grade = parts[i + 1]
                try:
                    fraction = float(parts[i + 2])
                    polymers.append((material, grade, fraction))
                    i += 3
                except ValueError:
                    i += 3
            else:
                break
        
        if not polymers:
            st.error("❌ Failed to parse blend string")
            return None, None, None, None, None
        
        # Step 3: Get environmental parameters (same as predict_blend_properties.py)
        available_env_params = {}
        if actual_thickness is not None:
            available_env_params['Thickness (um)'] = actual_thickness * 1000  # Convert mm to um
        
        # Step 4: Use the EXACT same prediction engine as predict_blend_properties.py
        result = predict_blend_property('compost', polymers, available_env_params, material_dict)
        
        if result is None:
            st.error("❌ Prediction failed")
            return None, None, None, None, None
        
        # Step 5: Generate curves using the EXACT same logic as predict_blend_properties.py
        # Extract data from the result (same structure)
        max_L_pred = result['max_L_pred']
        t0_pred = result['t0_pred']
        thickness = result['thickness']
        
        # Generate all curves using the dedicated function
        curve_results = generate_compostability_curves(
            max_L_pred, t0_pred, thickness,
            output_dir=output_prefix,
            save_csv=False, save_plot=True
        )
        
        if curve_results is None:
            st.error("❌ Curve generation failed")
            return None, None, None, None, None
        
        # Extract results
        disintegration_df = curve_results['disintegration_df']
        biodegradation_df = curve_results['biodegradation_df']
        k0_disintegration = curve_results['k0_disintegration']
        k0_biodegradation = curve_results['k0_biodegradation']
        
        # Capture matplotlib figures for Streamlit display
        disintegration_fig = plt.gcf() if plt.get_fignums() else None
        plt.close()
        
        # For biodegradation, we need to generate the plot separately since it's not saved
        if biodegradation_df is not None and not biodegradation_df.empty:
            plt.figure(figsize=(10, 6))
            plt.plot(biodegradation_df['day'], biodegradation_df['biodegradation'], 'b-', linewidth=2)
            plt.xlabel('Time (days)')
            plt.ylabel('Biodegradation (%)')
            plt.title('Quintic Biodegradation Curve')
            plt.grid(True, alpha=0.3)
            biodegradation_fig = plt.gcf()
            plt.close()
        else:
            biodegradation_fig = None
        
        # Return results in the same format as predict_blend_properties.py
        results = {
            'Max_L_Predicted': max_L_pred,
            't0_Predicted': t0_pred,
            'k0_Disintegration': k0_disintegration,
            'k0_Biodegradation': k0_biodegradation
        }
        
        return results, disintegration_df, biodegradation_df, disintegration_fig, biodegradation_fig
        
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None, None

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
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stMetric, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stAlert, .stSuccess, .stError, .stWarning, .stInfo, .stRadio, .stCheckbox, .stSlider, .stTextInput, .stTextArea, .stDateInput, .stTimeInput, .stColorPicker, .stFileUploader, .stImage, .stAudio, .stVideo, .stJson, .stCode, .stException, .stHelp, .stExpander, .stTabs, .stTab, .stSidebar, .stSidebarNav, .stSidebarNavItem, .stSidebarNavLink, .stSidebarNavLinkActive, .stSidebarNavLinkInactive, .stSidebarNavLinkSelected, .stSidebarNavLinkUnselected, .stSidebarNavLinkDisabled, .stSidebarNavLinkIcon, .stSidebarNavLinkLabel, .stSidebarNavLinkLabelText, .stSidebarNavLinkLabelIcon, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled {
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
    st.markdown('<p class="sub-header">Advanced XGBoost-based modeling for polymer blend biodegradation prediction</p>', unsafe_allow_html=True)
    
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
            options=["train/models/eol/v5/", "train/models/eol/v4/"],
            index=0,  # Default to v5 (latest)
            help="Select the directory containing the trained XGBoost models"
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
                    
                    # Run prediction using the new working logic
                    with st.spinner("Generating prediction..."):
                        # Run the new prediction logic
                        results, dis_df, bio_df, dis_fig, bio_fig = predict_blend(blend_string, temp_output_dir, model_dir, actual_thickness_mm)
                        
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
            
            # Display the captured matplotlib figures
            if dis_fig is not None:
                st.markdown("#### Disintegration Curve")
                st.pyplot(dis_fig)
                
            if bio_fig is not None:
                st.markdown("#### Biodegradation Curve")
                st.pyplot(bio_fig)
            
            # Display quintic biodegradation curve parameters
            st.markdown("#### Quintic Biodegradation Curve Parameters")
            
            # Extract parameters from the generated quintic curve data
            if bio_df is not None and not bio_df.empty:
                # Extract constraint points from the data
                day_0 = bio_df[bio_df['day'] == 0]['biodegradation'].iloc[0] if 0 in bio_df['day'].values else 0
                
                # Find t0 point
                t0_day = results['t0_Predicted']
                day_values = bio_df['day'].values
                t0_idx = np.argmin(np.abs(day_values - t0_day))
                t0_biodegradation = bio_df.iloc[t0_idx]['biodegradation']
                
                # Find closest day value for t0+10 since it might be a float
                t0_plus_10 = results['t0_Predicted'] + 10
                day_values = bio_df['day'].values
                closest_day_idx = np.argmin(np.abs(day_values - t0_plus_10))
                closest_day = day_values[closest_day_idx]
                day_t0_plus_10 = bio_df.iloc[closest_day_idx]['biodegradation']
                
                # Find maximum point
                max_biodegradation = bio_df['biodegradation'].max()
                max_day = bio_df.loc[bio_df['biodegradation'].idxmax(), 'day']
                
                day_400 = bio_df[bio_df['day'] == 400]['biodegradation'].iloc[0] if 400 in bio_df['day'].values else bio_df['biodegradation'].max()
                
                # Display all constraint points in columns
                st.markdown("**Constraint Points:**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Point 1", f"(0, {day_0:.2f}%)")
                with col2:
                    st.metric("Point 2", f"({t0_day:.0f}, {t0_biodegradation:.2f}%)")
                with col3:
                    st.metric("Point 3", f"({closest_day:.0f}, {day_t0_plus_10:.2f}%)")
                with col4:
                    st.metric("Point 4", f"({max_day:.0f}, {max_biodegradation:.2f}%)")
                with col5:
                    st.metric("Point 5", f"(400, {day_400:.2f}%)")
                
                # Display the quintic function form
                st.info(f"**Quintic Function:** y = ax⁵ + bx⁴ + cx³ + dx² + ex")
            else:
                st.warning("⚠️ Quintic biodegradation curve data not available")
            
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
                                label="📥 Download Disintegration Data",
                                data=file.read(),
                                file_name=f"{output_prefix}_disintegration_curves.csv",
                                mime="text/csv"
                            )
            
            if os.path.exists(biodegradation_png):
                st.image(biodegradation_png, use_container_width=True, caption="Quintic Biodegradation Curves")
                
                # Download buttons
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(biodegradation_png, "rb") as file:
                        st.download_button(
                            label="📥 Download Biodegradation Plot",
                            data=file.read(),
                            file_name=f"{output_prefix}_quintic_biodegradation_curves.png",
                            mime="image/png"
                        )
                
                with col2:
                    biodegradation_csv = os.path.join(output_dir, "quintic_biodegradation_curves.csv")
                    if os.path.exists(biodegradation_csv):
                        with open(biodegradation_csv, "rb") as file:
                            st.download_button(
                                label="📥 Download Biodegradation Data",
                                data=file.read(),
                                file_name=f"{output_prefix}_quintic_biodegradation_curves.csv",
                                mime="text/csv"
                            )
            
            # Display comparison plot if it exists
            comparison_png = os.path.join(output_dir, "quintic_vs_sigmoid_comparison.png")
            if os.path.exists(comparison_png):
                st.markdown("#### Comparison Plot")
                st.image(comparison_png, use_container_width=True, caption="Quintic vs Sigmoid Comparison")
                
                # Download button for comparison plot
                with open(comparison_png, "rb") as file:
                    st.download_button(
                        label="📥 Download Comparison Plot",
                        data=file.read(),
                        file_name=f"{output_prefix}_comparison.png",
                        mime="image/png"
                    )
        else:
            st.info("👈 Configure your blend on the left and click 'Generate Prediction' to see results here.")

if __name__ == "__main__":
    run_streamlit_app()
