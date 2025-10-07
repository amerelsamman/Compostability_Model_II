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
            
        # Get property values for this grade
        if 'property' not in df.columns:
            continue
            
        grade_data = df.loc[grade_matches]
        prop_values = grade_data['property'].dropna()
        
        if len(prop_values) == 0:
            continue
        
        # Separate blends with and without additives
        # Check if any of the polymer grades contain additives (like Glycerol)
        has_additive = grade_data[grade_columns].apply(
            lambda row: any('Glycerol' in str(val) for val in row if pd.notna(val)), axis=1
        )
        
        # Get values for each group
        values_with_additive = grade_data[has_additive]['property'].dropna().tolist()
        values_without_additive = grade_data[~has_additive]['property'].dropna().tolist()
        
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
    if prop in ['wvtr', 'otr']:  # These properties often have very large values
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
        
        ax.hist(values_array, bins=log_bins, alpha=0.7, edgecolor='black')
        ax.set_xscale('log')
        ax.set_xlabel(f'{prop.upper()} Value (log scale)')
        # For log scale, we don't need scientific notation formatting
    else:
        ax.hist(values_array, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{prop.upper()} Value')
        # Format x-axis to show readable numbers for linear scale
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    ax.set_ylabel('Frequency')
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
    if prop in ['wvtr', 'otr']:  # These properties often have very large values
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
            ax.hist(values_with_array, bins=log_bins, alpha=0.6, label='With Additives', color='red', edgecolor='black')
        if len(values_without_array) > 0:
            ax.hist(values_without_array, bins=log_bins, alpha=0.6, label='Without Additives', color='blue', edgecolor='black')
        
        ax.set_xscale('log')
        ax.set_xlabel(f'{prop.upper()} Value (log scale)')
    else:
        # Linear scale for other properties
        if len(values_with_array) > 0:
            ax.hist(values_with_array, bins=20, alpha=0.6, label='With Additives', color='red', edgecolor='black')
        if len(values_without_array) > 0:
            ax.hist(values_without_array, bins=20, alpha=0.6, label='Without Additives', color='blue', edgecolor='black')
        
        ax.set_xlabel(f'{prop.upper()} Value')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
    
    ax.set_ylabel('Frequency')
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
    tab1, tab2 = st.tabs(["🔬 Property Prediction", "📊 Grade Statistics"])
    
    with tab1:
        # Original prediction interface
        # Sidebar for blend selection
        st.sidebar.header("Blend Configuration")
        
        # Select polymers - material_dict has (Material, Grade) tuples as keys
        polymer_options = []
        for (material, grade), smiles in material_dict.items():
            if material in ['PLA', 'PBAT', 'PHB', 'PHA', 'PBS', 'PCL', 'Glycerol']:
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
            
            # Environmental parameters
            env_params = {
                'Temperature (C)': temperature,
                'RH (%)': humidity,
                'Thickness (um)': thickness
            }
            
            # Predict properties
            properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal']
            results = {}
            
            for prop in properties:
                try:
                    result = predict_blend_property(
                        property_type=prop,
                        polymers=polymers,
                        available_env_params=env_params,
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
    
    # Footer
    st.markdown("---")
    st.markdown("**Simple Additive Explorer** - ONE ADDITIVE: Glycerol")

if __name__ == "__main__":
    main()