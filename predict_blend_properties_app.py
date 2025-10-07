 #!/usr/bin/env python3
"""
Clean, Simple Polymer Blend Prediction Script with Streamlit Web Interface
Two clear sections:
1. Model Prediction (using train/modules/ - same as other properties)
2. Curve Generation (using train/modules_home/ - will become modules_home_curve/)

Usage:
  Command Line:
    python predict_blend_properties_app.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
    python predict_blend_properties_app.py compost "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
    python predict_blend_properties_app.py wvtr "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  
  Streamlit Web App:
    streamlit run predict_blend_properties_app.py
"""

import sys
import logging
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SECTION 1: MODEL PREDICTION (using train/modules/ - same as other properties)
# =============================================================================
from train.modules.input_parser import validate_input, load_and_validate_material_dictionary, parse_polymer_input
from train.modules.output_formatter import print_clean_summary
from train.modules.prediction_engine import predict_blend_property
from train.modules.prediction_utils import PROPERTY_CONFIGS

# =============================================================================
# SECTION 2: CURVE GENERATION (using train/modules_home/ - will become modules_home_curve/)
# =============================================================================
try:
    from train.modules_home.curve_generator import generate_compostability_curves
    CURVE_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Compostability curve generation not available: {e}")
    CURVE_GENERATION_AVAILABLE = False

try:
    from train.modules_sealing.curve_generator import generate_sealing_profile
    SEALING_CURVE_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sealing curve generation not available: {e}")
    SEALING_CURVE_GENERATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# STREAMLIT IMPORTS AND SETUP
# =============================================================================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("⚠️ Streamlit not available. Running in command-line mode only.")

# This function is no longer needed - compostability now uses predict_single_property from prediction_engine.py

def generate_compostability_curves(compost_result):
    """
    SECTION 2: Curve generation using train/modules_home/ (will become modules_home_curve/).
    This is the ADDITIONAL functionality on top of the standard prediction.
    """
    if not CURVE_GENERATION_AVAILABLE:
        logger.warning("⚠️ Curve generation not available, returning basic prediction")
        return compost_result
    
    try:
        # Extract data from the basic prediction result
        max_L_pred = compost_result['max_L_pred']
        t0_pred = compost_result['t0_pred']
        thickness = compost_result['thickness']
        
        # Call the dedicated curve generation function
        from train.modules_home.curve_generator import generate_compostability_curves as generate_curves
        
        curve_results = generate_curves(
            max_L_pred, t0_pred, thickness,
            output_dir="test_results/predict_blend_properties",
            save_csv=True, save_plot=True
        )
        
        if curve_results is None:
            return compost_result
        
        # Enhance the result with curve data
        enhanced_result = compost_result.copy()
        enhanced_result.update(curve_results)
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"❌ Curve generation failed: {e}")
        return compost_result

def generate_sealing_profile_curves(seal_result, polymers, compositions, material_dict):
    """
    SECTION 2: Sealing profile generation using train/modules_sealing/.
    This is the ADDITIONAL functionality on top of the standard seal prediction.
    """
    if not SEALING_CURVE_GENERATION_AVAILABLE:
        logger.warning("⚠️ Sealing curve generation not available, returning basic prediction")
        return seal_result
    
    try:
        # Extract predicted seal strength from the basic prediction result
        predicted_seal_strength = seal_result['prediction']
        
        # Load seal masterdata to get real polymer properties
        import pandas as pd
        masterdata_path = 'train/data/seal/masterdata.csv'
        
        # Try to load masterdata, but don't fail if it's not available
        try:
            if not os.path.exists(masterdata_path):
                logger.warning("⚠️ Seal masterdata.csv not found, using placeholder values")
                masterdata_df = None
            else:
                masterdata_df = pd.read_csv(masterdata_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not load masterdata.csv: {e}, using placeholder values")
            masterdata_df = None
        
        # Convert polymers from tuples to dictionaries for sealing profile generation
        # polymers is list of tuples: [(Material, Grade, vol_fraction), ...]
        polymer_dicts = []
        for i, (material, grade, vol_fraction) in enumerate(polymers):
            # Look up the polymer in masterdata if available
            if masterdata_df is not None:
                polymer_data = masterdata_df[
                    (masterdata_df['Materials'] == material) & 
                    (masterdata_df['Polymer Grade 1'] == grade)
                ]
                
                if not polymer_data.empty:
                    # Use actual data from masterdata
                    row = polymer_data.iloc[0]
                    polymer_dict = {
                        'material': material,
                        'grade': grade,
                        'vol_fraction': vol_fraction,
                        'melt temperature': row['melt temperature'],
                        'property': row['property'],
                        'degradation temperature': row['degradation temperature']
                    }
                else:
                    logger.warning(f"⚠️ Polymer {material} {grade} not found in masterdata, using placeholder values")
                    polymer_dict = {
                        'material': material,
                        'grade': grade,
                        'vol_fraction': vol_fraction,
                        'melt temperature': 150.0,  # Placeholder
                        'property': 10.0,  # Placeholder
                        'degradation temperature': 250.0  # Placeholder
                    }
            else:
                # Use placeholder values when masterdata is not available
                polymer_dict = {
                    'material': material,
                    'grade': grade,
                    'vol_fraction': vol_fraction,
                    'melt temperature': 150.0,  # Placeholder
                    'property': 10.0,  # Placeholder
                    'degradation temperature': 250.0  # Placeholder
                }
            
            polymer_dicts.append(polymer_dict)
        
        # Create blend name from polymer grades
        blend_name = "_".join([p[1] for p in polymers])  # p[1] is the grade
        
        # Call the dedicated sealing profile generation function
        # Try with file operations first, fall back to in-memory only if that fails
        try:
            curve_results = generate_sealing_profile(
                polymers=polymer_dicts,
                compositions=compositions,
                predicted_seal_strength=predicted_seal_strength,
                temperature_range=(0, 300),
                num_points=100,
                save_csv=True,
                save_plot=True,
                save_dir="test_results/predict_blend_properties",
                blend_name=blend_name
            )
        except Exception as file_error:
            logger.warning(f"⚠️ File operations failed ({file_error}), trying in-memory generation")
            # Fallback: try without file operations
            try:
                curve_results = generate_sealing_profile(
                    polymers=polymer_dicts,
                    compositions=compositions,
                    predicted_seal_strength=predicted_seal_strength,
                    temperature_range=(0, 300),
                    num_points=100,
                    save_csv=False,  # Disable file operations
                    save_plot=False,  # Disable file operations
                    save_dir=".",  # Not used when save_plot=False
                    blend_name=blend_name
                )
            except Exception as memory_error:
                logger.error(f"❌ Both file and in-memory generation failed: {memory_error}")
                return seal_result
        
        if curve_results is None or not curve_results.get('is_valid', False):
            logger.warning("⚠️ Sealing profile generation failed or invalid curve")
            return seal_result
        
        # Enhance the result with curve data
        enhanced_result = seal_result.copy()
        enhanced_result.update({
            'sealing_profile': curve_results,
            'curve_data': curve_results['curve_data'],
            'boundary_points': curve_results['boundary_points'],
            'temperatures': curve_results['temperatures'],
            'strengths': curve_results['strengths'],
            'is_valid': curve_results['is_valid']
        })
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"❌ Sealing profile generation failed: {e}")
        return seal_result

# =============================================================================
# STREAMLIT FUNCTIONS
# =============================================================================

def load_material_dictionary_streamlit(dict_path='material-smiles-dictionary.csv'):
    """Load the material-SMILES dictionary for Streamlit."""
    try:
        df = pd.read_csv(dict_path)
        material_dict = {}
        for _, row in df.iterrows():
            key = (row['Material'].strip(), row['Grade'].strip())
            material_dict[key] = row['SMILES'].strip()
        return material_dict
    except Exception as e:
        if STREAMLIT_AVAILABLE:
            st.error(f"❌ Error loading material dictionary: {e}")
        return None

def setup_streamlit_ui():
    """Setup Streamlit UI with dark theme and green buttons."""
    if not STREAMLIT_AVAILABLE:
        return
    
    # THIS MUST BE FIRST!
    st.set_page_config(
        page_title="Polymer Blend Property Prediction",
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
    .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader, .stCaption, .stDataFrame, .stTable, .stMetric, .stButton, .stDownloadButton, .stSelectbox, .stNumberInput, .stAlert, .stSuccess, .stError, .stWarning, .stInfo, .stRadio, .stCheckbox, .stSlider, .stTextInput, .stTextArea, .stDateInput, .stTimeInput, .stColorPicker, .stFileUploader, .stImage, .stAudio, .stVideo, .stJson, .stCode, .stException, .stHelp, .stExpander, .stTabs, .stTab, .stSidebar, .stSidebarNav, .stSidebarNavItem, .stSidebarNavLink, .stSidebarNavLinkActive, .stSidebarNavLinkInactive, .stSidebarNavLinkSelected, .stSidebarNavLinkUnselected, .stSidebarNavLinkDisabled, .stSidebarNavLinkIcon, .stSidebarNavLinkLabel, .stSidebarNavLinkLabelText, .stSidebarNavLinkLabelIcon, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled, .stSidebarNavLinkLabelIconActive, .stSidebarNavLinkLabelIconInactive, .stSidebarNavLinkLabelIconSelected, .stSidebarNavLinkLabelIconUnselected, .stSidebarNavLinkLabelIconDisabled {
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

def run_streamlit_app():
    """Run the Streamlit web application."""
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit is not available. Please install it with: pip install streamlit")
        return
    
    setup_streamlit_ui()
    
    # Initialize session state
    if 'prediction_generated' not in st.session_state:
        st.session_state['prediction_generated'] = False
    
    # Professional header
    st.markdown('<h1 class="main-header">🧬 Polymer Blend Property Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced XGBoost-based modeling for polymer blend property prediction</p>', unsafe_allow_html=True)
    
    # Load material dictionary
    material_dict = load_material_dictionary_streamlit()
    if material_dict is None:
        return
    
    # Get unique materials and grades for selection
    polymer_options = []
    for (material, grade), smiles in material_dict.items():
        polymer_options.append({
            'display': f"{material} {grade}",
            'material': material,
            'grade': grade,
            'smiles': smiles
        })
    
    # Sort by material name
    polymer_options.sort(key=lambda x: x['material'])
    
    # Main interface - 35% left for selection, 65% right for results
    col1, col2 = st.columns([35, 65])
    
    with col1:
        st.markdown("### Blend Configuration")
        
        # Property selection
        property_mode = st.selectbox(
            "Prediction Mode",
            options=["all", "wvtr", "ts", "eab", "cobb", "otr", "seal", "compost"],
            index=0,
            help="Select 'all' to predict all properties, or choose a specific property"
        )
        
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
                    step=0.000001,
                    format="%.6f",
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
                st.success(f"✅ Total: {total_vol_frac:.6f}")
            else:
                st.error(f"❌ Total: {total_vol_frac:.6f} (should be 1.0)")
        
        # Environmental parameters based on property mode
        available_env_params = {}
        
        # Show temperature and RH for WVTR/OTR (including when 'all' is selected)
        if property_mode in ['wvtr', 'otr'] or property_mode == 'all':
            st.markdown("#### Environmental Parameters")
            temp = st.number_input(
                "Temperature (°C)",
                min_value=-50.0,
                max_value=200.0,
                value=23.0,
                step=1.0,
                help="Temperature for WVTR/OTR prediction"
            )
            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                help="Relative humidity for WVTR/OTR prediction"
            )
            available_env_params['Temperature (C)'] = temp
            available_env_params['RH (%)'] = rh
        
        # Show thickness for properties that need it (including when 'all' is selected)
        if property_mode in ['wvtr', 'ts', 'eab', 'otr', 'seal', 'compost'] or property_mode == 'all':
            thickness_um = st.number_input(
                "Material Thickness (μm)",
                min_value=1.0,
                max_value=1000.0,
                value=50.0,
                step=1.0,
                help="Enter the actual thickness of the material in micrometers (μm)"
            )
            available_env_params['Thickness (um)'] = thickness_um
        
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
                # Create polymers list in the format expected by the backend
                polymers = []
                for polymer, vol_frac in zip(selected_polymers, volume_fractions):
                    polymers.append((polymer['material'], polymer['grade'], vol_frac))
                
                try:
                    # Create temporary output directory
                    temp_output_dir = f"test_results/{output_prefix}"
                    os.makedirs(temp_output_dir, exist_ok=True)
                    
                    # Run prediction using the original logic
                    with st.spinner("Generating prediction..."):
                        results = run_prediction_logic(property_mode, polymers, available_env_params, material_dict, temp_output_dir)
                        
                        if results:
                            # Store results in session state
                            st.session_state['prediction_generated'] = True
                            st.session_state['prediction_results'] = results
                            st.session_state['prediction_output_dir'] = temp_output_dir
                            st.session_state['property_mode'] = property_mode
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
            property_mode = st.session_state.get('property_mode', 'all')
            
            # Display results using the original logic
            display_streamlit_results(results, property_mode, output_dir)
        else:
            st.info("👈 Configure your blend on the left and click 'Generate Prediction' to see results here.")

def run_prediction_logic(mode, polymers, available_env_params, material_dict, output_dir):
    """Run the original prediction logic."""
    include_errors = True
    
    if mode == 'all':
        # Predict all properties
        results = []
        
        # All properties (including compostability) now use the same predict_blend_property function
        for prop_type in ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']:
            result = predict_blend_property(prop_type, polymers, available_env_params, material_dict, include_errors=include_errors)
            if result:
                # Add curve generation for special properties
                if prop_type == 'compost':
                    enhanced_result = generate_compostability_curves(result)
                    results.append(enhanced_result)
                elif prop_type == 'seal':
                    # Add sealing profile generation for seal
                    compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                    enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                    results.append(enhanced_result)
                else:
                    results.append(result)
        
        return results
        
    else:
        # Single property mode (using train/modules/ - same as other properties)
        result = predict_blend_property(mode, polymers, available_env_params, material_dict, include_errors=include_errors)
        
        if result:
            # Add curve generation for compostability if that's what was requested
            if mode == 'compost':
                enhanced_result = generate_compostability_curves(result)
                return [enhanced_result]
            elif mode == 'seal':
                # Add sealing profile generation for seal
                compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                return [enhanced_result]
            else:
                return [result]
        else:
            return []

def display_streamlit_results(results, property_mode, output_dir):
    """Display results in Streamlit format."""
    if not results:
        st.error("No results to display")
        return
    
    # Display results based on property mode
    if property_mode == 'all':
        # Display all properties
        st.markdown("#### Property Predictions")
        
        # Create columns for metrics
        num_properties = len(results)
        cols_per_row = 3
        for i in range(0, num_properties, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < num_properties:
                    result = results[i + j]
                    with col:
                        # Handle WVTR/OTR dictionary format
                        if result['property_type'] in ['wvtr', 'otr'] and isinstance(result['prediction'], dict):
                            if 'unnormalized_prediction' in result['prediction']:
                                pred_dict = result['prediction']
                                st.metric(
                                    result['name'],
                                    f"{pred_dict['unnormalized_prediction']:.6f}",
                                    help=f"Unit: {result['unit']} (at {pred_dict['thickness_um']:.1f}μm)"
                                )
                            else:
                                st.metric(
                                    result['name'],
                                    f"{result['prediction']['prediction']:.6f}",
                                    help=f"Unit: {result['unit']}"
                                )
                        else:
                            st.metric(
                                result['name'],
                                f"{result['prediction']:.6f}",
                                help=f"Unit: {result['unit']}"
                            )
        
        # Display special results for compostability and seal
        for result in results:
            if result['property_type'] == 'compost' and 'max_biodegradation' in result:
                st.markdown("#### Compostability Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Max Disintegration", f"{result['prediction']:.1f}%")
                with col2:
                    st.metric("Max Biodegradation", f"{result['max_biodegradation']:.1f}%")
                with col3:
                    st.metric("Time to 50% (t0)", f"{result['t0_pred']:.1f} days")
                with col4:
                    st.metric("k0 (Disintegration)", f"{result['k0_disintegration']:.4f}")
                
                # Display compostability curves
                display_compostability_curves(result, output_dir)
            
            elif result['property_type'] == 'seal' and 'sealing_profile' in result:
                st.markdown("#### Sealing Profile Details")
                boundary_points = result['boundary_points']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Initial Sealing", f"{boundary_points['initial_sealing'][0]:.0f}°C")
                with col2:
                    st.metric("First Polymer Max", f"{boundary_points['first_polymer_max'][0]:.0f}°C")
                with col3:
                    st.metric("Blend Predicted", f"{boundary_points['blend_predicted'][0]:.0f}°C")
                with col4:
                    st.metric("Degradation", f"{boundary_points['degradation'][0]:.0f}°C")
                
                # Display sealing profile curves
                display_sealing_profile_curves(result, output_dir)
    
    else:
        # Single property mode
        if results:
            result = results[0]
            st.markdown(f"#### {result['name']} Prediction")
            
            # Handle WVTR/OTR dictionary format
            if result['property_type'] in ['wvtr', 'otr'] and isinstance(result['prediction'], dict):
                if 'unnormalized_prediction' in result['prediction']:
                    pred_dict = result['prediction']
                    st.metric(
                        result['name'],
                        f"{pred_dict['unnormalized_prediction']:.6f} {result['unit']}",
                        help=f"At {pred_dict['thickness_um']:.1f}μm thickness"
                    )
                else:
                    st.metric(
                        result['name'],
                        f"{result['prediction']['prediction']:.6f} {result['unit']}"
                    )
            else:
                st.metric(
                    result['name'],
                    f"{result['prediction']:.6f} {result['unit']}"
                )
            
            # Special handling for compostability
            if result['property_type'] == 'compost' and 'max_biodegradation' in result:
                st.markdown("#### Compostability Details")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Max Disintegration", f"{result['prediction']:.1f}%")
                with col2:
                    st.metric("Max Biodegradation", f"{result['max_biodegradation']:.1f}%")
                with col3:
                    st.metric("Time to 50% (t0)", f"{result['t0_pred']:.1f} days")
                with col4:
                    st.metric("k0 (Disintegration)", f"{result['k0_disintegration']:.4f}")
                
                # Display compostability curves
                display_compostability_curves(result, output_dir)
            
            # Special handling for seal
            elif result['property_type'] == 'seal' and 'sealing_profile' in result:
                st.markdown("#### Sealing Profile Details")
                boundary_points = result['boundary_points']
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Initial Sealing", f"{boundary_points['initial_sealing'][0]:.0f}°C")
                with col2:
                    st.metric("First Polymer Max", f"{boundary_points['first_polymer_max'][0]:.0f}°C")
                with col3:
                    st.metric("Blend Predicted", f"{boundary_points['blend_predicted'][0]:.0f}°C")
                with col4:
                    st.metric("Degradation", f"{boundary_points['degradation'][0]:.0f}°C")
                
                # Display sealing profile curves
                display_sealing_profile_curves(result, output_dir)
    
    # Display other generated plots
    display_other_plots(output_dir)

def display_compostability_curves(result, output_dir):
    """Display compostability curves like in predict_compostability_streamlit.py"""
    st.markdown("#### Disintegration and Biodegradation Curves")
    
    # Look for compostability plot files in the standard location
    plot_dir = "test_results/predict_blend_properties"
    disintegration_png = os.path.join(plot_dir, "sigmoid_disintegration_curves.png")
    biodegradation_png = os.path.join(plot_dir, "quintic_biodegradation_curves.png")
    comparison_png = os.path.join(plot_dir, "quintic_vs_sigmoid_comparison.png")
    
    if os.path.exists(disintegration_png):
        st.image(disintegration_png, use_container_width=True, caption="Disintegration Curves")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            with open(disintegration_png, "rb") as file:
                st.download_button(
                    label="📥 Download Disintegration Plot",
                    data=file.read(),
                    file_name="disintegration_curves.png",
                    mime="image/png"
                )
        
        with col2:
            disintegration_csv = os.path.join(plot_dir, "sigmoid_disintegration_curves.csv")
            if os.path.exists(disintegration_csv):
                with open(disintegration_csv, "rb") as file:
                    st.download_button(
                        label="📥 Download Disintegration Data",
                        data=file.read(),
                        file_name="disintegration_curves.csv",
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
                    file_name="quintic_biodegradation_curves.png",
                    mime="image/png"
                )
        
        with col2:
            biodegradation_csv = os.path.join(plot_dir, "quintic_biodegradation_curves.csv")
            if os.path.exists(biodegradation_csv):
                with open(biodegradation_csv, "rb") as file:
                    st.download_button(
                        label="📥 Download Biodegradation Data",
                        data=file.read(),
                        file_name="quintic_biodegradation_curves.csv",
                        mime="text/csv"
                    )
    
    # Display comparison plot if it exists
    if os.path.exists(comparison_png):
        st.markdown("#### Comparison Plot")
        st.image(comparison_png, use_container_width=True, caption="Quintic vs Sigmoid Comparison")
        
        # Download button for comparison plot
        with open(comparison_png, "rb") as file:
            st.download_button(
                label="📥 Download Comparison Plot",
                data=file.read(),
                file_name="comparison.png",
                mime="image/png"
            )

def display_sealing_profile_curves(result, output_dir):
    """Display sealing profile curves matching compostability style"""
    st.markdown("#### Sealing Profile Curves")
    
    # First try to find existing plot files
    plot_dir = "test_results/predict_blend_properties"
    sealing_files = []
    if os.path.exists(plot_dir):
        for file in os.listdir(plot_dir):
            if 'sealing_profile' in file and file.endswith('.png'):
                sealing_files.append(file)
    
    # If plot files exist, display them
    if sealing_files:
        sealing_files.sort(key=lambda x: os.path.getmtime(os.path.join(plot_dir, x)), reverse=True)
        sealing_file = sealing_files[0]
        sealing_path = os.path.join(plot_dir, sealing_file)
        st.image(sealing_path, use_container_width=True, caption="Sealing Profile Curves")
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            with open(sealing_path, "rb") as file:
                st.download_button(
                    label="📥 Download Sealing Profile Plot",
                    data=file.read(),
                    file_name="sealing_profile_curves.png",
                    mime="image/png"
                )
        
        with col2:
            csv_file = sealing_file.replace('.png', '.csv')
            csv_path = os.path.join(plot_dir, csv_file)
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Sealing Profile Data",
                        data=file.read(),
                        file_name="sealing_profile_curves.csv",
                        mime="text/csv"
                    )
    
    # If no plot files exist, try to generate plot from embedded data
    elif 'temperatures' in result and 'strengths' in result and 'boundary_points' in result:
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Generate plot from embedded data
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Set dark theme
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            
            # Plot main curve
            temperatures = result['temperatures']
            strengths = result['strengths']
            ax.plot(temperatures, strengths, color='#2E8B57', linewidth=3, label='Sealing Profile', alpha=0.9)
            
            # Plot boundary points
            boundary_points = result['boundary_points']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            markers = ['o', 's', '^', 'D']
            
            for i, (name, (temp, strength)) in enumerate(boundary_points.items()):
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                ax.scatter(temp, strength, c=color, s=120, marker=marker, 
                          edgecolors='white', linewidth=2, zorder=5,
                          label=f'{name.replace("_", " ").title()}: ({temp:.0f}°C, {strength:.1f} N/15mm)')
            
            ax.set_xlabel('Temperature (°C)', fontweight='bold', fontsize=14)
            ax.set_ylabel('Sealing Strength (N/15mm)', fontweight='bold', fontsize=14)
            ax.set_title('Sealing Profile Curves', fontweight='bold', fontsize=16, pad=20)
            ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=11, framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Display the plot
            st.pyplot(fig)
            plt.close(fig)
            
            # Create download button for CSV data
            if 'curve_data' in result:
                csv_data = result['curve_data'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Sealing Profile Data",
                    data=csv_data,
                    file_name="sealing_profile_curves.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.warning(f"Could not generate plot from data: {e}")
            st.info("No sealing profile plots found.")
    
    else:
        st.info("No sealing profile plots found.")

def display_other_plots(output_dir):
    """Display other generated plots (excluding compostability and sealing profile plots)"""
    # Look for other plot files in both output directory and standard location
    plot_files = []
    
    # Check output directory first
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if (file.endswith('.png') and 
                'sealing_profile' not in file and 
                'disintegration' not in file and 
                'biodegradation' not in file and 
                'comparison' not in file):
                plot_files.append((file, output_dir))
    
    # Check standard location for any other plots
    plot_dir = "test_results/predict_blend_properties"
    if os.path.exists(plot_dir):
        for file in os.listdir(plot_dir):
            if (file.endswith('.png') and 
                'sealing_profile' not in file and 
                'disintegration' not in file and 
                'biodegradation' not in file and 
                'comparison' not in file and
                not any(f[0] == file for f in plot_files)):  # Avoid duplicates
                plot_files.append((file, plot_dir))
    
    if plot_files:
        st.markdown("#### Other Generated Plots")
        for plot_file, plot_dir in plot_files:
            plot_path = os.path.join(plot_dir, plot_file)
            st.image(plot_path, use_container_width=True, caption=plot_file.replace('.png', '').replace('_', ' ').title())
            
            # Download button
            with open(plot_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {plot_file.replace('.png', '').replace('_', ' ').title()}",
                    data=file.read(),
                    file_name=f"blend_prediction_{plot_file}",
                    mime="image/png"
                )
    else:
        # Only show this message if there are no other plots at all
        st.info("No other plots generated for this prediction.")
    
    # Display CSV files if available (excluding compostability and sealing profile CSVs)
    csv_files = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if (file.endswith('.csv') and 
                'sealing_profile' not in file and 
                'disintegration' not in file and 
                'biodegradation' not in file):
                csv_files.append((file, output_dir))
    
    if csv_files:
        st.markdown("#### Generated Data Files")
        for csv_file, csv_dir in csv_files:
            csv_path = os.path.join(csv_dir, csv_file)
            with open(csv_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {csv_file.replace('.csv', '').replace('_', ' ').title()} Data",
                    data=file.read(),
                    file_name=f"blend_prediction_{csv_file}",
                    mime="text/csv"
                )

def main():
    """Main function to run the clean blend prediction."""
    # Check for --no-errors flag
    include_errors = True
    if '--no-errors' in sys.argv:
        include_errors = False
        sys.argv.remove('--no-errors')
    
    # Validate input
    mode, polymer_input, available_env_params = validate_input()
    if mode is None:
        sys.exit(1)
    
    # Load material dictionary
    material_dict = load_and_validate_material_dictionary()
    if material_dict is None:
        return
    
    # Parse polymer input
    polymers, parsed_env_params = parse_polymer_input(polymer_input, mode)
    if polymers is None:
        return
    
    # Merge environmental parameters (command line takes precedence)
    if parsed_env_params:
        available_env_params.update(parsed_env_params)
    
    if mode == 'all':
        # Predict all properties
        results = []
        
        # All properties (including compostability) now use the same predict_blend_property function
        for prop_type in ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']:
            result = predict_blend_property(prop_type, polymers, available_env_params, material_dict, include_errors=include_errors)
            if result:
                # Add curve generation for special properties
                if prop_type == 'compost':
                    enhanced_result = generate_compostability_curves(result)
                    results.append(enhanced_result)
                elif prop_type == 'seal':
                    # Add sealing profile generation for seal
                    compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                    enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                    results.append(enhanced_result)
                else:
                    results.append(result)
        
        # Print clean summary
        print_clean_summary(results)
        
        return results
        
    else:
        # Single property mode (using train/modules/ - same as other properties)
        result = predict_blend_property(mode, polymers, available_env_params, material_dict, include_errors=include_errors)
        
        if result:
            # Add curve generation for compostability if that's what was requested
            if mode == 'compost':
                enhanced_result = generate_compostability_curves(result)
                
                if 'max_biodegradation' in enhanced_result:
                    # Enhanced results with curves
                    print(f"• Max Disintegration - {enhanced_result['prediction']:.1f}%")
                    print(f"• Max Biodegradation - {enhanced_result['max_biodegradation']:.1f}%")
                    print(f"• Time to 50% (t0) - {enhanced_result['t0_pred']:.1f} days")
                    print(f"• k0 (Disintegration) - {enhanced_result['k0_disintegration']:.4f}")
                    print(f"• k0 (Biodegradation) - {enhanced_result['k0_biodegradation']:.4f}")
                else:
                    # Basic results without curves
                    print(f"• Max Disintegration - {enhanced_result['prediction']:.1f}%")
                
                return enhanced_result
            elif mode == 'seal':
                # Add sealing profile generation for seal
                compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                
                # Print basic seal result
                config = PROPERTY_CONFIGS[enhanced_result['property_type']]
                print(f"• {enhanced_result['name']} - {enhanced_result['prediction']:.6f} {enhanced_result['unit']}")
                
                # Print sealing profile information
                if 'sealing_profile' in enhanced_result:
                    boundary_points = enhanced_result['boundary_points']
                    print(f"\n📊 Sealing Profile Generated:")
                    print(f"  • Initial sealing: {boundary_points['initial_sealing'][0]:.0f}°C, {boundary_points['initial_sealing'][1]:.1f} N/15mm")
                    print(f"  • First polymer max: {boundary_points['first_polymer_max'][0]:.0f}°C, {boundary_points['first_polymer_max'][1]:.1f} N/15mm")
                    print(f"  • Blend predicted: {boundary_points['blend_predicted'][0]:.0f}°C, {boundary_points['blend_predicted'][1]:.1f} N/15mm")
                    print(f"  • Degradation: {boundary_points['degradation'][0]:.0f}°C, {boundary_points['degradation'][1]:.1f} N/15mm")
                    print(f"  • Curve data saved to: test_results/predict_blend_properties/")
                
                return enhanced_result
            else:
                # Standard property results
                config = PROPERTY_CONFIGS[result['property_type']]
                print(f"• {config['name']} - {result['prediction']:.6f} {config['unit']}")
                return result['prediction']
        else:
            return None

if __name__ == "__main__":
    # Check if running as Streamlit app
    if STREAMLIT_AVAILABLE and len(sys.argv) == 1:
        # No command line arguments, assume Streamlit mode
        run_streamlit_app()
    else:
        # Command line mode
        main()
