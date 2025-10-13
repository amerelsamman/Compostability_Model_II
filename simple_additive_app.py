#!/usr/bin/env python3
"""
Simple Additive Explorer - ONE ADDITIVE (Glycerol)
Start from 0 to 100 - build solid foundation
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the train directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from train.modules.prediction_engine import predict_blend_property
from train.modules.prediction_utils import load_material_dictionary

# Page config
st.set_page_config(
    page_title="Simple Additive Explorer",
    page_icon="🧪",
    layout="wide"
)

# Load material dictionary from both sources
@st.cache_data
def load_materials():
    # Load main polymer dictionary
    polymer_dict = load_material_dictionary('material-smiles-dictionary.csv')
    
    # Load additives/fillers dictionary
    additives_dict = load_material_dictionary('additives-fillers-dictionary.csv')
    
    # Combine both dictionaries
    combined_dict = {**polymer_dict, **additives_dict}
    
    return combined_dict

# Load training data for grade statistics
@st.cache_data
def load_training_data():
    """Load training data for grade statistics analysis"""
    data_files = {
        'wvtr': 'train/data/wvtr/polymerblends_for_ml.csv',
        'ts': 'train/data/ts/polymerblends_for_ml.csv',
        'eab': 'train/data/eab/polymerblends_for_ml.csv',
        'cobb': 'train/data/cobb/polymerblends_for_ml.csv',
        'otr': 'train/data/otr/polymerblends_for_ml.csv',
        'seal': 'train/data/seal/polymerblends_for_ml.csv',
        'compost': 'train/data/eol/polymerblends_for_ml.csv'
    }
    
    training_data = {}
    for prop, file_path in data_files.items():
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                training_data[prop] = df
        except Exception as e:
            st.warning(f"Could not load {prop} data: {e}")
    
    return training_data

def unnormalize_to_thickness(normalized_value, thickness):
    """Unnormalize from 1μm reference to actual film thickness - just divide by thickness"""
    return normalized_value / thickness

def get_representative_polymer_families(material_dict):
    """Get representative polymer families with their first grade for comparison testing"""
    families = {}
    
    # Define polymer families and their representative grades (matching material-smiles-dictionary.csv)
    family_mapping = {
        'PLA': ['Ingeo 4032D', 'Ingeo 4043D', 'Luminy® L175', 'PT101'],
        'PBAT': ['Ecoworld', 'ecoflex® F Blend C1200', 'Ecovance rf-PBAT'],
        'PHB': ['BIOCYCLE® 1000', 'ENMAT Y3000'],
        'PHAa': ['PHACT A1000P'],
        'PHAs': ['PHACT S1000P'],
        'PHA': ['BP330-05', 'BP350-05', 'PB3000G', 'PB3430G'],
        'PHBV': ['ENMAT™ Y1000P'],
        'PHBH': ['Green Planet™ PHBH'],
        'PBS': ['BioPBS™ FZ91', 'PBS TH803S'],
        'PBSA': ['BioPBS™ FD92'],
        'PCL': ['Capa™ 6500', 'Capa™ 6800'],
        'PGA': ['Kuredux® PGA', 'Polylactide'],
        'Bio-PE': ['I\'m green™ STN7006'],
        'LDPE': ['LDPE LD 150'],
        'PP': ['Total PPH 3270'],
        'PET': ['Mylar 48-F-OC Clear PET PET'],
        'PVDC': ['Barrialon CX C6'],
        'PA': ['Aegis PCR-H135ZP Nylon 6'],
        'EVOH': ['EVAL F171B']
    }
    
    # Find first available grade for each family
    for family, possible_grades in family_mapping.items():
        for grade in possible_grades:
            if (family, grade) in material_dict:
                families[family] = grade
                break
    
    return families

def analyze_grade_statistics(grade, training_data):
    """Analyze statistics for a specific grade across all properties, separating blends with/without additives"""
    results = {}
    
    for prop, df in training_data.items():
        if df is None or df.empty:
            continue
            
        # Find blends containing this grade
        grade_columns = [col for col in df.columns if 'Polymer Grade' in col]
        grade_matches = []
        
        for col in grade_columns:
            grade_matches.extend(df[df[col] == grade].index.tolist())
        
        if not grade_matches:
            continue
            
        grade_data = df.loc[grade_matches]
        
        # Handle different property column structures
        if prop in ['ts', 'eab']:
            # TS and EAB have property1 and property2 (MD and TD)
            if 'property1' in df.columns and 'property2' in df.columns:
                # Combine both properties for analysis
                prop1_values = grade_data['property1'].dropna()
                prop2_values = grade_data['property2'].dropna()
                prop_values = pd.concat([prop1_values, prop2_values])
            else:
                continue
        elif prop == 'compost':
            # EoL/compost has property1 (disintegration_max) and property2 (t0)
            if 'property1' in df.columns:
                # Use property1 for disintegration_max
                prop_values = grade_data['property1'].dropna()
            else:
                continue
        elif prop in ['wvtr', 'otr'] and 'property' in df.columns:
            # WVTR and OTR have single property column
            if 'Thickness (um)' in grade_data.columns:
                # Unnormalize from 1μm to actual thickness - just divide by thickness
                thickness_values = grade_data['Thickness (um)'].values
                normalized_values = grade_data['property'].values
                
                # Apply unnormalization - simple division
                unnormalized_values = [unnormalize_to_thickness(norm_val, thick_val) 
                                     for norm_val, thick_val in zip(normalized_values, thickness_values)]
                
                # Create new series with unnormalized values
                prop_values = pd.Series(unnormalized_values, index=grade_data.index)
            else:
                prop_values = grade_data['property'].dropna()
        elif 'property' in df.columns:
            # Other properties with single property column
            prop_values = grade_data['property'].dropna()
        else:
            continue
        
        if len(prop_values) == 0:
            continue
        
        # Separate blends with and without additives
        # Check if any of the polymer grades contain additives (like Glycerol)
        has_additive = grade_data[grade_columns].apply(
            lambda row: any('Glycerol' in str(val) for val in row if pd.notna(val)), axis=1
        )
        
        # Get values for each group - need to handle the case where prop_values might be from concatenated data
        if prop in ['ts', 'eab']:
            # For TS and EAB, we need to check additive status for each individual value
            # Since we concatenated property1 and property2, we need to duplicate the additive status
            has_additive_extended = pd.concat([has_additive, has_additive])
            values_with_additive = prop_values[has_additive_extended].dropna().tolist()
            values_without_additive = prop_values[~has_additive_extended].dropna().tolist()
        else:
            # For other properties, use the original logic
            values_with_additive = prop_values[has_additive].dropna().tolist()
            values_without_additive = prop_values[~has_additive].dropna().tolist()
        
        # Calculate statistics for both groups
        stats = {
            'all': {
                'mean': prop_values.mean(),
                'median': prop_values.median(),
                'std': prop_values.std(),
                'min': prop_values.min(),
                'max': prop_values.max(),
                'count': len(prop_values),
                'values': prop_values.tolist()
            },
            'with_additive': {
                'mean': np.mean(values_with_additive) if values_with_additive else None,
                'median': np.median(values_with_additive) if values_with_additive else None,
                'std': np.std(values_with_additive) if values_with_additive else None,
                'min': min(values_with_additive) if values_with_additive else None,
                'max': max(values_with_additive) if values_with_additive else None,
                'count': len(values_with_additive),
                'values': values_with_additive
            },
            'without_additive': {
                'mean': np.mean(values_without_additive) if values_without_additive else None,
                'median': np.median(values_without_additive) if values_without_additive else None,
                'std': np.std(values_without_additive) if values_without_additive else None,
                'min': min(values_without_additive) if values_without_additive else None,
                'max': max(values_without_additive) if values_without_additive else None,
                'count': len(values_without_additive),
                'values': values_without_additive
            }
        }
        
        results[prop] = stats
    
    return results

def create_property_histogram(prop, values, title):
    """Create a histogram for property values"""
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Convert values to numpy array for easier manipulation
    values_array = np.array(values)
    
    # Handle scientific notation and large values
    if prop in ['wvtr', 'otr', 'eab']:  # These properties often have very large values
        # Use log scale for better visualization with logarithmic binning
        # Create logarithmically spaced bins
        positive_values = values_array[values_array > 0]  # Avoid log(0)
        if len(positive_values) > 0:
            log_min = np.log10(np.min(positive_values))
            log_max = np.log10(np.max(values_array))
            log_bins = np.logspace(log_min, log_max, 20)  # 20 bins in log space
        else:
            # Fallback to regular bins if no positive values
            log_bins = 20
        
        ax.hist(values_array, bins=log_bins, alpha=0.7, edgecolor='black', density=True)
        ax.set_xscale('log')
        ax.set_xlabel(f'{prop.upper()} Value (log scale)')
        # For log scale, we don't need scientific notation formatting
    else:
        ax.hist(values_array, bins=20, alpha=0.7, edgecolor='black', density=True)
        ax.set_xlabel(f'{prop.upper()} Value')
        # Format x-axis to show readable numbers for linear scale
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    ax.set_ylabel('Density')
    ax.set_title(f'{title} - {prop.upper()} Distribution')
    ax.grid(True, alpha=0.3)
    
    return fig

def create_comparison_histogram(prop, values_with, values_without, title):
    """Create a comparison histogram showing blends with and without additives"""
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Convert values to numpy arrays
    values_with_array = np.array(values_with) if values_with else np.array([])
    values_without_array = np.array(values_without) if values_without else np.array([])
    
    # Handle scientific notation and large values
    if prop in ['wvtr', 'otr', 'eab']:  # These properties often have very large values
        # Use log scale for better visualization with logarithmic binning
        all_values = np.concatenate([values_with_array, values_without_array])
        positive_values = all_values[all_values > 0]  # Avoid log(0)
        
        if len(positive_values) > 0:
            log_min = np.log10(np.min(positive_values))
            log_max = np.log10(np.max(all_values))
            log_bins = np.logspace(log_min, log_max, 20)  # 20 bins in log space
        else:
            log_bins = 20
        
        # Plot both histograms
        if len(values_with_array) > 0:
            ax.hist(values_with_array, bins=log_bins, alpha=0.6, label='With Additives', color='red', edgecolor='black', density=True)
        if len(values_without_array) > 0:
            ax.hist(values_without_array, bins=log_bins, alpha=0.6, label='Without Additives', color='blue', edgecolor='black', density=True)
        
        ax.set_xscale('log')
        ax.set_xlabel(f'{prop.upper()} Value (log scale)')
    else:
        # Linear scale for other properties
        if len(values_with_array) > 0:
            ax.hist(values_with_array, bins=20, alpha=0.6, label='With Additives', color='red', edgecolor='black', density=True)
        if len(values_without_array) > 0:
            ax.hist(values_without_array, bins=20, alpha=0.6, label='Without Additives', color='blue', edgecolor='black', density=True)
        
        ax.set_xlabel(f'{prop.upper()} Value')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    ax.set_ylabel('Density')
    ax.set_title(f'{title} - {prop.upper()} Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    st.title("🧪 Simple Additive Explorer")
    st.markdown("**ONE ADDITIVE: Glycerol** - Building from 0 to 100")
    
    # Load materials
    material_dict = load_materials()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔬 Property Prediction", "📊 Grade Statistics", "🧪 Grade Comparison"])
    
    with tab1:
        # Original prediction interface
        # Sidebar for blend selection
        st.sidebar.header("Blend Configuration")
        
        # Select polymers - material_dict has (Material, Grade) tuples as keys
        polymer_options = []
        for (material, grade), smiles in material_dict.items():
            polymer_options.append(f"{material} - {grade}")
        
        polymer1 = st.sidebar.selectbox("Polymer 1", polymer_options, index=0)
        fraction1 = st.sidebar.slider("Fraction 1", 0.1, 0.8, 0.4, 0.1)
        
        polymer2 = st.sidebar.selectbox("Polymer 2", polymer_options, index=1)
        fraction2 = st.sidebar.slider("Fraction 2", 0.1, 0.8, 0.4, 0.1)
        
        polymer3 = st.sidebar.selectbox("Polymer 3", polymer_options, index=2)
        max_fraction3 = 1.0 - fraction1 - fraction2
        if max_fraction3 > 0.0:  # Only show slider if there's space for Polymer 3
            fraction3 = st.sidebar.slider("Fraction 3", 0.0, max_fraction3, 0.0, 0.1)
        else:
            st.sidebar.info("ℹ️ No space for Polymer 3 - reduce other fractions first")
            fraction3 = 0.0
        
        # Display current total
        current_total = fraction1 + fraction2 + fraction3
        st.sidebar.write(f"**Current Total**: {current_total:.1f}")
        
        if current_total > 1.0:
            st.sidebar.warning("⚠️ Total fractions exceed 1.0! Please adjust.")
        elif current_total < 1.0:
            st.sidebar.info(f"ℹ️ Remaining: {1.0 - current_total:.1f} available for Glycerol")
        
        # Environmental parameters
        st.sidebar.header("Environmental Parameters")
        temperature = st.sidebar.slider("Temperature (°C)", 20, 40, 25)
        humidity = st.sidebar.slider("Humidity (%)", 30, 100, 60)
        thickness = st.sidebar.slider("Thickness (μm)", 10, 250, 100)
        
        # WVTR/OTR specific parameters
        st.sidebar.header("WVTR/OTR Specific Parameters")
        wvtr_temp = st.sidebar.slider("WVTR Temperature (°C)", 20, 50, 39)
        wvtr_humidity = st.sidebar.slider("WVTR Humidity (%)", 30, 100, 90)
        otr_temp = st.sidebar.slider("OTR Temperature (°C)", 20, 50, 25)
        otr_humidity = st.sidebar.slider("OTR Humidity (%)", 30, 100, 50)
        
        # Glycerol additive
        st.sidebar.header("Glycerol Additive")
        add_glycerol = st.sidebar.checkbox("Add Glycerol", value=False)
        glycerol_fraction = 0.0
        
        if add_glycerol:
            max_glycerol = 1.0 - current_total
            if max_glycerol > 0.01:  # Only show slider if there's space for Glycerol
                glycerol_fraction = st.sidebar.slider("Glycerol Fraction", 0.01, max_glycerol, min(0.05, max_glycerol), 0.01)
                # Adjust polymer fractions proportionally
                total_polymer = fraction1 + fraction2 + fraction3
                if total_polymer > 0:
                    scale_factor = (1 - glycerol_fraction) / total_polymer
                    fraction1 = fraction1 * scale_factor
                    fraction2 = fraction2 * scale_factor
                    fraction3 = fraction3 * scale_factor
            else:
                st.sidebar.warning("⚠️ No space for Glycerol! Reduce polymer fractions first.")
                glycerol_fraction = 0.0
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Property Predictions")
            
            # Create blend
            polymers = []
            if polymer1:
                material1, grade1 = polymer1.split(" - ")
                polymers.append((material1, grade1, fraction1))
            
            if polymer2:
                material2, grade2 = polymer2.split(" - ")
                polymers.append((material2, grade2, fraction2))
            
            if polymer3 and fraction3 > 0:
                material3, grade3 = polymer3.split(" - ")
                polymers.append((material3, grade3, fraction3))
            
            if add_glycerol:
                polymers.append(("Glycerol", "Glycerol", glycerol_fraction))
            
            # Environmental parameters - use specific values for WVTR/OTR
            env_params = {
                'Temperature (C)': temperature,
                'RH (%)': humidity,
                'Thickness (um)': thickness
            }
            
            # WVTR-specific environmental parameters
            wvtr_env_params = {
                'Temperature (C)': wvtr_temp,
                'RH (%)': wvtr_humidity,
                'Thickness (um)': thickness
            }
            
            # OTR-specific environmental parameters
            otr_env_params = {
                'Temperature (C)': otr_temp,
                'RH (%)': otr_humidity,
                'Thickness (um)': thickness
            }
            
            # Predict properties
            properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal']
            results = {}
            
            for prop in properties:
                try:
                    # Use specific environmental parameters for WVTR and OTR
                    if prop == 'wvtr':
                        env_params_to_use = wvtr_env_params
                    elif prop == 'otr':
                        env_params_to_use = otr_env_params
                    else:
                        env_params_to_use = env_params
                    
                    result = predict_blend_property(
                        property_type=prop,
                        polymers=polymers,
                        available_env_params=env_params_to_use,
                        material_dict=material_dict,
                        include_errors=True
                    )
                    if 'prediction' in result:
                        results[prop] = result
                    else:
                        results[prop] = {'prediction': 0, 'unit': 'N/A', 'error': 'No prediction'}
                except Exception as e:
                    results[prop] = {'prediction': 0, 'unit': 'N/A', 'error': str(e)}
            
            # Display results
            for prop, result in results.items():
                if 'prediction' in result and 'error' not in result:
                    # Extract the actual prediction value
                    pred_value = result['prediction']
                    if isinstance(pred_value, dict):
                        # For WVTR/OTR, use unnormalized prediction (actual thickness)
                        if 'unnormalized_prediction' in pred_value:
                            pred_value = pred_value['unnormalized_prediction']
                        elif 'prediction' in pred_value:
                            pred_value = pred_value['prediction']
                    
                    st.metric(
                        label=f"{prop.upper()}",
                        value=f"{pred_value:.2f}",
                        help=f"Unit: {result['unit']}"
                    )
                else:
                    st.metric(
                        label=f"{prop.upper()}",
                        value="Error",
                        help=f"Error: {result.get('error', 'Unknown error')}"
                    )
        
        with col2:
            st.header("Blend Composition")
            if polymer1:
                st.write(f"**Polymer 1**: {polymer1}")
                st.write(f"Fraction: {fraction1:.1%}")
            if polymer2:
                st.write(f"**Polymer 2**: {polymer2}")
                st.write(f"Fraction: {fraction2:.1%}")
            if polymer3 and fraction3 > 0:
                st.write(f"**Polymer 3**: {polymer3}")
                st.write(f"Fraction: {fraction3:.1%}")
            if add_glycerol:
                st.write(f"**Additive**: Glycerol")
                st.write(f"Fraction: {glycerol_fraction:.1%}")
            
            # Show total
            total = fraction1 + fraction2 + fraction3 + glycerol_fraction
            st.write(f"**Total**: {total:.1%}")
    
    with tab2:
        # Grade Statistics Tab
        st.header("📊 Grade Statistics Analysis")
        st.markdown("Analyze the performance profile of the selected Polymer 1 grade across all training data.")
        st.info("ℹ️ **WVTR and OTR values are unnormalized to actual film thickness** - statistics show real-world performance at recorded thicknesses.")
        
        # Load training data
        with st.spinner("Loading training data..."):
            training_data = load_training_data()
        
        if not training_data:
            st.error("No training data available. Please ensure the training data files exist.")
            return
        
        # Get the selected grade from Polymer 1
        if polymer1:
            selected_grade = polymer1.split(" - ")[1]  # Extract grade name
            st.subheader(f"Statistics for Grade: **{selected_grade}**")
            
            # Analyze statistics
            with st.spinner(f"Analyzing statistics for {selected_grade}..."):
                grade_stats = analyze_grade_statistics(selected_grade, training_data)
            
            if not grade_stats:
                st.warning(f"No data found for grade '{selected_grade}' in the training datasets.")
                return
            
            # Display statistics for each property
            for prop, stats in grade_stats.items():
                st.subheader(f"{prop.upper()} Statistics")
                
                # Overall statistics
                st.markdown("**Overall Statistics (All Blends)**")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Mean", f"{stats['all']['mean']:.3f}")
                with col2:
                    st.metric("Median", f"{stats['all']['median']:.3f}")
                with col3:
                    st.metric("Std Dev", f"{stats['all']['std']:.3f}")
                with col4:
                    st.metric("Min", f"{stats['all']['min']:.3f}")
                with col5:
                    st.metric("Max", f"{stats['all']['max']:.3f}")
                
                st.info(f"**Total Sample Count**: {stats['all']['count']} blends containing {selected_grade}")
                
                # Comparison statistics
                st.markdown("**Comparison: With vs Without Additives**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🔴 With Additives**")
                    if stats['with_additive']['count'] > 0:
                        st.write(f"Count: {stats['with_additive']['count']}")
                        st.write(f"Mean: {stats['with_additive']['mean']:.3f}")
                        st.write(f"Median: {stats['with_additive']['median']:.3f}")
                        st.write(f"Std: {stats['with_additive']['std']:.3f}")
                    else:
                        st.write("No data available")
                
                with col2:
                    st.markdown("**🔵 Without Additives**")
                    if stats['without_additive']['count'] > 0:
                        st.write(f"Count: {stats['without_additive']['count']}")
                        st.write(f"Mean: {stats['without_additive']['mean']:.3f}")
                        st.write(f"Median: {stats['without_additive']['median']:.3f}")
                        st.write(f"Std: {stats['without_additive']['std']:.3f}")
                    else:
                        st.write("No data available")
                
                # Create comparison histogram
                if stats['with_additive']['count'] > 0 or stats['without_additive']['count'] > 0:
                    fig = create_comparison_histogram(
                        prop, 
                        stats['with_additive']['values'], 
                        stats['without_additive']['values'], 
                        selected_grade
                    )
                    st.pyplot(fig)
                else:
                    st.warning(f"Not enough data points for {prop.upper()} comparison histogram")
                
                st.markdown("---")
        else:
            st.info("Please select a Polymer 1 grade in the Property Prediction tab to see statistics.")
    
    with tab3:
        # Grade Comparison Tab
        st.header("🧪 Grade Comparison Analysis")
        st.markdown("Test the selected Polymer 1 grade in 50:50 blends with other polymer families, with and without Glycerol additive.")
        
        # Get the selected grade from Polymer 1
        if polymer1:
            selected_material, selected_grade = polymer1.split(" - ")
            st.subheader(f"Testing Grade: **{selected_grade}** ({selected_material})")
            
            # Get representative polymer families
            families = get_representative_polymer_families(material_dict)
            
            if not families:
                st.error("No polymer families available for comparison.")
                return
            
            # Environmental parameters (use same as tab1)
            env_params = {
                'Temperature (C)': temperature,
                'RH (%)': humidity,
                'Thickness (um)': thickness
            }
            
            # WVTR-specific environmental parameters
            wvtr_env_params = {
                'Temperature (C)': wvtr_temp,
                'RH (%)': wvtr_humidity,
                'Thickness (um)': thickness
            }
            
            # OTR-specific environmental parameters
            otr_env_params = {
                'Temperature (C)': otr_temp,
                'RH (%)': otr_humidity,
                'Thickness (um)': thickness
            }
            
            # Properties to test
            properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']
            
            # Create comparison results
            comparison_results = []
            
            with st.spinner("Running predictions for all polymer family combinations..."):
                for family, family_grade in families.items():
                    # Test ALL families, including the same material family
                    
                    # Test both compositions
                    for composition_name, polymer1_frac, polymer2_frac, glycerol_frac in [
                        ("50:50:0 (No Additive)", 0.5, 0.5, 0.0),
                        ("40:40:20 (With Glycerol)", 0.4, 0.4, 0.2)
                    ]:
                        # Create blend
                        polymers = [
                            (selected_material, selected_grade, polymer1_frac),
                            (family, family_grade, polymer2_frac)
                        ]
                        
                        if glycerol_frac > 0:
                            polymers.append(("Glycerol", "Glycerol", glycerol_frac))
                        
                        # Predict all properties - follow exact same procedure as Property Prediction tab
                        blend_results = {
                            'Polymer1': f"{selected_material} {selected_grade}",
                            'Polymer2': f"{family} {family_grade}",
                            'Composition': composition_name,
                            'P1_Fraction': polymer1_frac,
                            'P2_Fraction': polymer2_frac,
                            'Glycerol_Fraction': glycerol_frac
                        }
                        
                        for prop in properties:
                            try:
                                # Use specific environmental parameters for WVTR and OTR
                                if prop == 'wvtr':
                                    env_params_to_use = wvtr_env_params
                                elif prop == 'otr':
                                    env_params_to_use = otr_env_params
                                else:
                                    env_params_to_use = env_params
                                
                                result = predict_blend_property(
                                    property_type=prop,
                                    polymers=polymers,
                                    available_env_params=env_params_to_use,
                                    material_dict=material_dict,
                                    include_errors=True
                                )
                                
                                if 'prediction' in result:
                                    # Extract the actual prediction value - EXACT same as Property Prediction tab
                                    pred_value = result['prediction']
                                    if isinstance(pred_value, dict):
                                        # For WVTR/OTR, use unnormalized prediction (actual thickness)
                                        if 'unnormalized_prediction' in pred_value:
                                            pred_value = pred_value['unnormalized_prediction']
                                        elif 'prediction' in pred_value:
                                            pred_value = pred_value['prediction']
                                    
                                    blend_results[f'{prop.upper()}_Value'] = pred_value
                                    blend_results[f'{prop.upper()}_Unit'] = result.get('unit', 'N/A')
                                    
                                else:
                                    blend_results[f'{prop.upper()}_Value'] = None
                                    blend_results[f'{prop.upper()}_Unit'] = 'No prediction'
                            except Exception as e:
                                blend_results[f'{prop.upper()}_Value'] = None
                                blend_results[f'{prop.upper()}_Unit'] = f'Error: {str(e)}'
                        
                        comparison_results.append(blend_results)
            
            # Display results
            if comparison_results:
                st.subheader("📊 Comparison Results")
                
                # Create a summary table
                summary_data = []
                for result in comparison_results:
                    summary_data.append({
                        'Blend': f"{result['Polymer1']} + {result['Polymer2']}",
                        'Composition': result['Composition'],
                        'WVTR': f"{result.get('WVTR_Value', 0):.2f}" if result.get('WVTR_Value') is not None else 'N/A',
                        'TS': f"{result.get('TS_Value', 0):.2f}" if result.get('TS_Value') is not None else 'N/A',
                        'EAB': f"{result.get('EAB_Value', 0):.2f}" if result.get('EAB_Value') is not None else 'N/A',
                        'Cobb': f"{result.get('COBB_Value', 0):.2f}" if result.get('COBB_Value') is not None else 'N/A',
                        'OTR': f"{result.get('OTR_Value', 0):.2f}" if result.get('OTR_Value') is not None else 'N/A',
                        'Seal': f"{result.get('SEAL_Value', 0):.2f}" if result.get('SEAL_Value') is not None else 'N/A',
                        'Compost': f"{result.get('COMPOST_Value', 0):.2f}" if result.get('COMPOST_Value') is not None else 'N/A'
                    })
                
                df_summary = pd.DataFrame(summary_data)
                st.dataframe(df_summary, use_container_width=True)
                
                # Create comparison charts for each property
                st.subheader("📈 Property Comparison Charts")
                
                for prop in properties:
                    st.subheader(f"{prop.upper()} Comparison")
                    
                    # Prepare data for plotting
                    blend_names = []
                    no_additive_values = []
                    with_additive_values = []
                    
                    for result in comparison_results:
                        blend_name = f"{result['Polymer2']}"
                        if blend_name not in blend_names:
                            blend_names.append(blend_name)
                            
                            # Find values for this blend
                            no_additive_result = next((r for r in comparison_results 
                                                     if r['Polymer2'] == blend_name and r['Composition'] == "50:50:0 (No Additive)"), None)
                            with_additive_result = next((r for r in comparison_results 
                                                       if r['Polymer2'] == blend_name and r['Composition'] == "40:40:20 (With Glycerol)"), None)
                            
                            no_additive_values.append(no_additive_result.get(f'{prop.upper()}_Value') if no_additive_result and no_additive_result.get(f'{prop.upper()}_Value') else 0)
                            with_additive_values.append(with_additive_result.get(f'{prop.upper()}_Value') if with_additive_result and with_additive_result.get(f'{prop.upper()}_Value') else 0)
                    
                    # Calculate average percentage change for the selected grade
                    # This should be calculated for each individual blend (selected grade + other family)
                    # then averaged across all blends
                    if len(no_additive_values) > 0 and len(with_additive_values) > 0 and len(no_additive_values) == len(with_additive_values):
                        # Calculate percentage changes for each individual blend
                        percentage_changes = []
                        for i in range(len(no_additive_values)):
                            if no_additive_values[i] != 0:  # Avoid division by zero
                                pct_change = ((with_additive_values[i] - no_additive_values[i]) / no_additive_values[i]) * 100
                                percentage_changes.append(pct_change)
                        
                        avg_pct_change = np.mean(percentage_changes) if percentage_changes else 0
                    else:
                        avg_pct_change = 0
                    
                    # Create comparison chart
                    fig, ax = plt.subplots(figsize=(12, 6))
                    x = np.arange(len(blend_names))
                    width = 0.35
                    
                    ax.bar(x - width/2, no_additive_values, width, label='No Additive (50:50:0)', alpha=0.8)
                    ax.bar(x + width/2, with_additive_values, width, label='With Glycerol (40:40:20)', alpha=0.8)
                    
                    ax.set_xlabel('Polymer Family')
                    ax.set_ylabel(f'{prop.upper()} Value')
                    ax.set_title(f'{prop.upper()} Comparison: {selected_grade} vs Other Families\nAvg % Change with Glycerol: {avg_pct_change:+.1f}%')
                    ax.set_xticks(x)
                    ax.set_xticklabels(blend_names, rotation=45, ha='right')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    st.pyplot(fig)
            else:
                st.warning("No comparison results available.")
        else:
            st.info("Please select a Polymer 1 grade in the Property Prediction tab to run comparisons.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple Additive Explorer** - ONE ADDITIVE: Glycerol")

if __name__ == "__main__":
    main()