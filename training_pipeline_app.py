#!/usr/bin/env python3
"""
Streamlit Training Pipeline App
Runs the complete training pipeline from simulation to model evaluation with a single click.
"""

import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add the simulation directory to the path
sys.path.append(os.path.join(os.getcwd(), 'train', 'simulation'))

# Import simulation functions
from train.simulation.simulation_common import run_simulation_for_property, run_all_simulations
from train.simulation.simulation_rules import PROPERTY_CONFIGS

# Blending rules configuration for each property
BLENDING_RULES = {
    'adhesion': {
        'combined_rom_thin': {
            'name': 'Combined Rule of Mixtures (thin films < 30μm)',
            'description': 'Uses 50/50 combination of rule of mixtures and inverse rule for thin films',
            'enabled_by_default': True
        },
        'standard_rom_thick': {
            'name': 'Standard Rule of Mixtures (thick films ≥ 30μm)', 
            'description': 'Uses standard rule of mixtures for thicker films',
            'enabled_by_default': True
        }
    },
    'eab': {
        'inverse_rom_brittle_soft': {
            'name': 'Inverse Rule of Mixtures (brittle + soft flex)',
            'description': 'Uses inverse rule when brittle and soft flex polymers are combined',
            'enabled_by_default': True
        },
        'regular_rom': {
            'name': 'Regular Rule of Mixtures',
            'description': 'Uses standard rule of mixtures for other combinations',
            'enabled_by_default': True
        }
    },
    'wvtr': {
        'inverse_rom': {
            'name': 'Inverse Rule of Mixtures (WVTR)',
            'description': 'Uses inverse rule of mixtures for WVTR (barrier properties)',
            'enabled_by_default': True
        }
    },
    'otr': {
        'inverse_rom': {
            'name': 'Inverse Rule of Mixtures (OTR)',
            'description': 'Uses inverse rule of mixtures for OTR (barrier properties)',
            'enabled_by_default': True
        }
    },
    'ts': {
        'regular_rom': {
            'name': 'Regular Rule of Mixtures (TS)',
            'description': 'Uses standard rule of mixtures for tensile strength',
            'enabled_by_default': True
        },
        'inverse_rom_brittle_soft': {
            'name': 'Inverse Rule of Mixtures (brittle + soft flex)',
            'description': 'Uses inverse rule when brittle and soft flex polymers are combined',
            'enabled_by_default': True
        },
        'inverse_rom_hard_soft': {
            'name': 'Inverse Rule of Mixtures (hard + soft flex)',
            'description': 'Uses inverse rule when hard and soft flex polymers are combined',
            'enabled_by_default': True
        },
        'miscibility_rule': {
            'name': 'Miscibility Rule (Phase Separation)',
            'description': 'If ≥30% immiscible components, both TS1 and TS2 become random 5-7 MPa',
            'enabled_by_default': True
        }
    },
    'cobb': {
        'inverse_rom': {
            'name': 'Inverse Rule of Mixtures (Cobb)',
            'description': 'Uses inverse rule of mixtures for Cobb angle',
            'enabled_by_default': True
        }
    },
    'compost': {
        'all_home_compostable': {
            'name': 'All Polymers Home-Compostable',
            'description': 'Random max_L between 90-95 for purely home-compostable blends',
            'enabled_by_default': True
        },
        'mixed_blend_rom': {
            'name': 'Mixed Blend (Rule of Mixtures)',
            'description': 'Uses weighted averages for mixed compostable blends',
            'enabled_by_default': True
        },
        'pla_compostable_rule': {
            'name': 'PLA + Compostable Rule',
            'description': 'PLA excluded from max_L calculation, included in t0 calculation',
            'enabled_by_default': True
        },
        'default_rom': {
            'name': 'Default Rule of Mixtures',
            'description': 'Standard rule of mixtures for both max_L and t0',
            'enabled_by_default': True
        }
    }
}

def setup_streamlit_ui():
    """Setup Streamlit UI with dark theme and green buttons."""
    # THIS MUST BE FIRST!
    st.set_page_config(
        page_title="Training Pipeline Manager",
        page_icon="🚀",
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

def run_command(command, description):
    """Run a command and return success status and output"""
    try:
        st.info(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            st.success(f"✅ {description} completed successfully!")
            # Store output in session state instead of displaying immediately
            if 'pipeline_outputs' not in st.session_state:
                st.session_state['pipeline_outputs'] = []
            st.session_state['pipeline_outputs'].append({
                'step': description,
                'output': result.stdout,
                'success': True
            })
            return True, result.stdout
        else:
            st.error(f"❌ {description} failed!")
            # Store error output in session state
            if 'pipeline_outputs' not in st.session_state:
                st.session_state['pipeline_outputs'] = []
            st.session_state['pipeline_outputs'].append({
                'step': description,
                'output': result.stderr,
                'success': False
            })
            return False, result.stderr
    except Exception as e:
        st.error(f"❌ Error during {description}: {str(e)}")
        # Store exception in session state
        if 'pipeline_outputs' not in st.session_state:
            st.session_state['pipeline_outputs'] = []
        st.session_state['pipeline_outputs'].append({
            'step': description,
            'output': str(e),
            'success': False
        })
        return False, str(e)

def display_rule_usage_stats():
    """Display rule usage statistics from simulation"""
    if 'simulation_rule_usage' not in st.session_state:
        return
    
    rule_usage = st.session_state['simulation_rule_usage']
    
    if not rule_usage or 'rules' not in rule_usage:
        return
    
    st.markdown("#### 📊 Rule Usage Statistics")
    st.markdown("*Shows how many times each blending rule was applied during simulation*")
    
    # Create a nice table for rule usage
    rules_data = rule_usage['rules']
    total_blends = rule_usage['total_blends']
    
    if rules_data:
        # Create DataFrame for better display
        df_rules = pd.DataFrame(rules_data)
        df_rules['percentage'] = df_rules['percentage'].round(1)
        
        # Display as a styled table
        st.dataframe(
            df_rules[['rule_name', 'count', 'percentage']].rename(columns={
                'rule_name': 'Rule Name',
                'count': 'Times Used',
                'percentage': 'Percentage (%)'
            }),
            use_container_width=True,
            hide_index=True
        )
        
        # Summary stats
        st.info(f"📈 **Total blends generated:** {total_blends}")
        st.info(f"🔧 **Different rules used:** {len(rules_data)}")
    else:
        st.warning("No rule usage data available.")

def display_pipeline_outputs():
    """Display all pipeline outputs in a nice dark-themed format"""
    if 'pipeline_outputs' not in st.session_state or not st.session_state['pipeline_outputs']:
        return
    
    st.markdown("#### 📋 Pipeline Output")
    
    # Add custom CSS for dark-themed output display
    st.markdown("""
    <style>
    .pipeline-output {
        background-color: #1e1e1e;
        color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333333;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        line-height: 1.4;
        overflow-x: auto;
        white-space: pre-wrap;
        margin: 10px 0;
        max-height: 400px;
        overflow-y: auto;
    }
    .pipeline-step {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 10px;
        border-radius: 6px;
        margin: 10px 0;
        border-left: 4px solid #4ade80;
    }
    .pipeline-step.error {
        border-left-color: #f87171;
    }
    .pipeline-step h4 {
        color: #4ade80;
        margin: 0 0 10px 0;
        font-size: 14px;
    }
    .pipeline-step.error h4 {
        color: #f87171;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Display each step's output
    for i, output_data in enumerate(st.session_state['pipeline_outputs']):
        step_name = output_data['step']
        output_text = output_data['output']
        success = output_data['success']
        
        if not output_text.strip():
            continue
            
        # Create step header
        status_icon = "✅" if success else "❌"
        status_class = "" if success else "error"
        
        st.markdown(f"""
        <div class="pipeline-step {status_class}">
            <h4>{status_icon} {step_name}</h4>
            <div class="pipeline-output">{output_text}</div>
        </div>
        """, unsafe_allow_html=True)

def run_simulation_step(property_name, sample_count, seed, selected_rules=None):
    """Run the simulation step"""
    try:
        # Import simulation functions
        from train.simulation.simulation_common import run_simulation_for_property
        from train.simulation.simulation_rules import PROPERTY_CONFIGS
        
        # Get property configuration
        if property_name not in PROPERTY_CONFIGS:
            return False, f"Property '{property_name}' not found in configuration"
        
        property_config = PROPERTY_CONFIGS[property_name]
        
        # Run simulation directly with selected rules
        st.info(f"🔄 Simulating {property_name} with {sample_count:,} samples...")
        
        # Set random seed
        import numpy as np
        import random
        np.random.seed(seed)
        random.seed(seed)
        
        # Run simulation
        result = run_simulation_for_property(
            property_name=property_name,
            target_total=sample_count,
            property_config=property_config,
            selected_rules=selected_rules
        )
        
        if result:
            st.success(f"✅ Simulation completed successfully!")
            
            # Extract rule usage data from simulation result
            combined_data, augmented_data, ml_dataset, simulation_summary = result
            rule_usage = simulation_summary.get('rule_usage', {})
            
            # Store output in session state
            if 'pipeline_outputs' not in st.session_state:
                st.session_state['pipeline_outputs'] = []
            
            # Create detailed output with rule usage
            rule_usage_text = ""
            if rule_usage and 'rules' in rule_usage:
                rule_usage_text = f"\n\n📊 Rule Usage Statistics:\n"
                rule_usage_text += f"Total blends generated: {rule_usage['total_blends']}\n"
                rule_usage_text += f"Different rules used: {len(rule_usage['rules'])}\n\n"
                for rule in rule_usage['rules']:
                    rule_usage_text += f"• {rule['rule_name']}: {rule['count']} times ({rule['percentage']:.1f}%)\n"
            
            st.session_state['pipeline_outputs'].append({
                'step': f"Simulating {property_name} with {sample_count:,} samples",
                'output': f"Generated {sample_count:,} samples for {property_name}{rule_usage_text}",
                'success': True,
                'rule_usage': rule_usage
            })
            
            # Store rule usage data in session state for display
            st.session_state['simulation_rule_usage'] = rule_usage
            
            return True, "Simulation completed successfully"
        else:
            st.error(f"❌ Simulation failed!")
            return False, "Simulation failed"
            
    except Exception as e:
        st.error(f"❌ Error during simulation: {str(e)}")
        return False, str(e)

def run_featurization_step(property_name):
    """Run the featurization step"""
    input_file = f"train/data/{property_name}/polymerblends_for_ml.csv"
    output_file = f"train/data/{property_name}/polymerblends_for_ml_featurized.csv"
    
    command = f"python train/run_blend_featurization.py {input_file} {output_file}"
    return run_command(command, f"Featurizing {property_name} data")

def run_training_step(property_name, model_version, last_n_training=None, last_n_testing=None, oversampling_factor=None):
    """Run the training step"""
    input_file = f"train/data/{property_name}/polymerblends_for_ml_featurized.csv"
    output_dir = f"train/models/{property_name}/{model_version}"
    
    command = f"python train/training/train_unified_modular.py --property {property_name} --input {input_file} --output {output_dir}"
    
    # Add parameters (including 0 values to override defaults)
    if last_n_training is not None:
        command += f" --last_n_training {last_n_training}"
    if last_n_testing is not None:
        command += f" --last_n_testing {last_n_testing}"
    if oversampling_factor is not None:
        command += f" --oversampling_factor {oversampling_factor}"
    
    return run_command(command, f"Training {property_name} model (v{model_version})")

def display_training_results(property_name, model_version, last_n_testing=0):
    """Display training results and plots with proper filtering and styling"""
    model_dir = f"train/models/{property_name}/{model_version}"
    
    if not os.path.exists(model_dir):
        st.error(f"Model directory not found: {model_dir}")
        return
    
    st.markdown("#### Training Results")
    
    # Display model files
    model_files = []
    plot_files = []
    csv_files = []
    
    for file in os.listdir(model_dir):
        if file.endswith('.pkl'):
            model_files.append(file)
        elif file.endswith('.png'):
            plot_files.append(file)
        elif file.endswith('.csv'):
            csv_files.append(file)
    
    # Display plots with proper filtering
    if plot_files:
        st.markdown("##### Generated Plots")
        
        # Main results plot
        main_plot = None
        last_n_plot = None
        
        for plot_file in plot_files:
            if 'comprehensive_polymer_model_results' in plot_file:
                main_plot = plot_file
            elif f'last_{last_n_testing}_blends_performance' in plot_file:
                last_n_plot = plot_file
        
        # Display main results plot
        if main_plot:
            plot_path = os.path.join(model_dir, main_plot)
            st.image(plot_path, use_container_width=True, caption="Model Performance Results")
            
            # Download button
            with open(plot_path, "rb") as file:
                st.download_button(
                    label="📥 Download Model Performance Plot",
                    data=file.read(),
                    file_name=main_plot,
                    mime="image/png"
                )
        
        # Display last N blends plot based on last_n_testing parameter
        if last_n_testing > 0:
            if last_n_plot:
                plot_path = os.path.join(model_dir, last_n_plot)
                st.image(plot_path, use_container_width=True, caption=f"Last {last_n_testing} Blends Performance")
                
                # Download button
                with open(plot_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Download Last {last_n_testing} Blends Plot",
                        data=file.read(),
                        file_name=last_n_plot,
                        mime="image/png"
                    )
            else:
                st.warning(f"⚠️ Last {last_n_testing} blends plot not found. This may indicate an issue with the training process.")
                # Debug: show available plot files
                st.write("Available plot files:")
                for plot_file in plot_files:
                    st.write(f"- {plot_file}")
        else:
            st.warning("⚠️ Last N Testing was set to 0, so no last N blends performance plot was generated.")
    
    # Display CSV files
    if csv_files:
        st.markdown("##### Generated Data Files")
        for csv_file in csv_files:
            csv_path = os.path.join(model_dir, csv_file)
            with open(csv_path, "rb") as file:
                st.download_button(
                    label=f"📥 Download {csv_file.replace('.csv', '').replace('_', ' ').title()} Data",
                    data=file.read(),
                    file_name=csv_file,
                    mime="text/csv"
                )
    
    # Display model files info
    if model_files:
        st.markdown("##### Generated Model Files")
        for model_file in model_files:
            st.info(f"📁 {model_file}")

def main():
    """Main Streamlit app function"""
    setup_streamlit_ui()
    
    # Initialize session state
    if 'pipeline_completed' not in st.session_state:
        st.session_state['pipeline_completed'] = False
    if 'current_property' not in st.session_state:
        st.session_state['current_property'] = None
    if 'current_version' not in st.session_state:
        st.session_state['current_version'] = None
    
    # Professional header
    st.markdown('<h1 class="main-header">🚀 Training Pipeline Manager</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete ML pipeline from simulation to model evaluation</p>', unsafe_allow_html=True)
    
    # Main interface - 40% left for configuration, 60% right for results
    col1, col2 = st.columns([40, 60])
    
    with col1:
        st.markdown("### Pipeline Configuration")
        
        # Property selection
        property_options = list(PROPERTY_CONFIGS.keys())
        property_name = st.selectbox(
            "Property to Train",
            options=property_options,
            index=0,
            help="Select a specific property to train"
        )
        
        # Blending rules selection
        st.markdown("#### 🔧 Blending Rules Configuration")
        st.markdown("*Select which polymer blending rules to use in the simulation*")
        
        if property_name in BLENDING_RULES:
            selected_rules = {}
            rules = BLENDING_RULES[property_name]
            
            # Create checkboxes for each rule
            for rule_key, rule_config in rules.items():
                default_value = rule_config['enabled_by_default']
                selected = st.checkbox(
                    f"**{rule_config['name']}**",
                    value=default_value,
                    key=f"rule_{property_name}_{rule_key}",
                    help=rule_config['description']
                )
                selected_rules[rule_key] = selected
            
            # Validate that at least one rule is selected
            if not any(selected_rules.values()):
                st.error("⚠️ **Please select at least one blending rule!**")
                st.stop()
            
            # Show selected rules summary
            selected_count = sum(selected_rules.values())
            total_count = len(rules)
            st.success(f"✅ {selected_count}/{total_count} blending rules selected")
            
            # Store selected rules in session state
            st.session_state['selected_rules'] = selected_rules
        else:
            st.error(f"No blending rules defined for property: {property_name}")
            st.stop()
        
        # Sample count
        sample_count = st.number_input(
            "Number of Samples",
            min_value=1,
            max_value=50000,
            value=5000,
            step=1000,
            help="Number of augmented samples to generate (use 1 for quick rule testing)"
        )
        
        # Model version
        model_version = st.text_input(
            "Model Version",
            value="v1",
            help="Version identifier for the trained model (e.g., v1, v2, etc.)"
        )
        
        # Show existing model versions for the selected property
        existing_versions = []
        model_base_dir = f"train/models/{property_name}"
        if os.path.exists(model_base_dir):
            for item in os.listdir(model_base_dir):
                if os.path.isdir(os.path.join(model_base_dir, item)) and item.startswith('v'):
                    existing_versions.append(item)
        
        if existing_versions:
            existing_versions.sort()
            st.info(f"📁 **Existing versions for {property_name}:** {', '.join(existing_versions)}")
            if model_version in existing_versions:
                st.warning(f"⚠️ Version '{model_version}' already exists and will be overwritten!")
        
        # Random seed
        seed = st.number_input(
            "Random Seed",
            min_value=1,
            max_value=999999,
            value=42,
            help="Random seed for reproducibility"
        )
        
        # Training Parameters - Made more prominent
        st.markdown("#### Training Parameters")
        st.markdown("*These parameters control how the data is split between training and testing sets*")
        
        # Last N Training - More prominent
        last_n_training = st.number_input(
            "Last N Training",
            min_value=0,
            max_value=100,
            value=0,
            help="Number of last blends to put in training set (0 = use property default). These are the most recent/important blends that should always be in training."
        )
        
        # Last N Testing - More prominent
        last_n_testing = st.number_input(
            "Last N Testing",
            min_value=0,
            max_value=100,
            value=0,
            help="Number of last blends to put in testing set (0 = use property default). These blends will be used for final model evaluation."
        )
        
        # Add helpful information about these parameters
        with st.expander("ℹ️ About Last N Parameters"):
            st.markdown("""
            **Last N Training:** The most recent N blends that will always be included in the training set. 
            This ensures the model learns from the latest/most important data.
            
            **Last N Testing:** The most recent N blends that will be used for testing. 
            This can override some of the Last N Training blends (e.g., Last 10 in training, but Last 3 of those in testing).
            
            **Recommended Values:**
            - **Adhesion:** Last N Training: 5-10, Last N Testing: 3-5
            - **TS/EAB:** Last N Training: 4-8, Last N Testing: 2-4  
            - **WVTR/OTR:** Last N Training: 10-20, Last N Testing: 5-10
            - **Cobb/Compost:** Last N Training: 0-5, Last N Testing: 0-3
            """)
        
        # Show property-specific defaults
        if property_name in PROPERTY_CONFIGS:
            config = PROPERTY_CONFIGS[property_name]
            st.info(f"**{property_name.upper()} Defaults:** Last N Training: {config.get('default_last_n_training', 0)}, Last N Testing: {config.get('default_last_n_testing', 0)}")
        
        # Advanced options
        with st.expander("Advanced Training Options"):
            oversampling_factor = st.number_input(
                "Oversampling Factor",
                min_value=0,
                max_value=20,
                value=0,
                help="Factor for oversampling last N blends (0 = use property default)"
            )
            
            # Show property-specific oversampling defaults
            if property_name != "all" and property_name in PROPERTY_CONFIGS:
                config = PROPERTY_CONFIGS[property_name]
                st.info(f"**{property_name.upper()} Oversampling Default:** {config.get('oversampling_factor', 0)}")
        
        # Show current configuration
        st.markdown("#### Current Configuration")
        st.markdown(f"**Property:** {property_name}")
        st.markdown(f"**Samples:** {sample_count:,}")
        st.markdown(f"**Model Version:** {model_version}")
        st.markdown(f"**Last N Training:** {last_n_training if last_n_training > 0 else 'Use default'}")
        st.markdown(f"**Last N Testing:** {last_n_testing if last_n_testing > 0 else 'Use default'}")
        st.markdown(f"**Oversampling Factor:** {oversampling_factor if oversampling_factor > 0 else 'Use default'}")
        st.markdown(f"**Random Seed:** {seed}")
        
        # Run pipeline button
        run_pipeline = st.button("🚀 Run Complete Pipeline", type="primary")
        
        if run_pipeline:
            # Check for existing model directories before starting
            existing_dirs = []
            model_dir = f"train/models/{property_name}/{model_version}"
            if os.path.exists(model_dir):
                existing_dirs.append(f"{property_name}/{model_version}")
            
            if existing_dirs:
                st.error("🚫 **Pipeline cannot start - existing model directories found!**")
                st.error("The following model directories already exist and would be overwritten:")
                for dir_path in existing_dirs:
                    st.error(f"  • `train/models/{dir_path}`")
                st.error("")
                st.error("**Please either:**")
                st.error("1. Choose a different model version (e.g., v2, v3, etc.)")
                st.error("2. Manually delete the existing directories")
                st.error("3. Rename the existing directories")
                st.error("")
                st.error("**This prevents accidental data loss!**")
                return
            
            # Clear previous pipeline outputs
            st.session_state['pipeline_outputs'] = []
            
            # Store current configuration
            st.session_state['current_property'] = property_name
            st.session_state['current_version'] = model_version
            st.session_state['last_n_testing'] = last_n_testing
            
            # Run the complete pipeline
            with st.spinner("Running complete training pipeline..."):
                success = True
                
                # Step 1: Simulation
                sim_success, _ = run_simulation_step(property_name, sample_count, seed, st.session_state.get('selected_rules'))
                if not sim_success:
                    success = False
                    st.error("❌ Pipeline stopped at simulation step")
                    return
                
                # Step 2: Featurization
                feat_success, _ = run_featurization_step(property_name)
                if not feat_success:
                    success = False
                    st.error("❌ Pipeline stopped at featurization step")
                    return
                
                # Step 3: Training
                train_success, _ = run_training_step(
                    property_name, model_version,
                    last_n_training,
                    last_n_testing,
                    oversampling_factor if oversampling_factor > 0 else None
                )
                if not train_success:
                    success = False
                    st.error("❌ Pipeline stopped at training step")
                    return
                
                if success:
                    st.session_state['pipeline_completed'] = True
                    st.success("🎉 Complete pipeline executed successfully!")
                else:
                    st.error("💥 Pipeline execution failed!")

    with col2:
        st.markdown("### Pipeline Results")
        
        if st.session_state.get('pipeline_completed', False):
            current_property = st.session_state.get('current_property')
            current_version = st.session_state.get('current_version')
            
            # Display pipeline outputs first
            display_pipeline_outputs()
            
            # Display rule usage statistics
            display_rule_usage_stats()
            
            st.markdown("---")
            
            # Display results for single property
            last_n_testing = st.session_state.get('last_n_testing', 0)
            display_training_results(current_property, current_version, last_n_testing)
        else:
            st.info("👈 Configure your pipeline on the left and click 'Run Complete Pipeline' to see results here.")
    
    # Pipeline status
    st.markdown("---")
    st.markdown("### Pipeline Status")
    
    if st.session_state.get('pipeline_completed', False):
        st.success("✅ Pipeline completed successfully!")
        st.info("📁 Check the train/models/ directory for all generated models and plots")
    else:
        st.info("⏳ Pipeline not yet executed")

if __name__ == "__main__":
    main()
