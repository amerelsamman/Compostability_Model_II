#!/usr/bin/env python3
"""
Simple test to check if imports work
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
import os
import sys

# Add modules to path
sys.path.append('modules')

try:
    from modules.optimizer import DifferentiableLabelOptimizer
    st.write("✅ DifferentiableLabelOptimizer imported successfully")
except Exception as e:
    st.write(f"❌ Error importing DifferentiableLabelOptimizer: {e}")

try:
    from modules.blend_feature_extractor import process_blend_features
    st.write("✅ process_blend_features imported successfully")
except Exception as e:
    st.write(f"❌ Error importing process_blend_features: {e}")

try:
    from modules.utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves
    st.write("✅ utils functions imported successfully")
except Exception as e:
    st.write(f"❌ Error importing utils functions: {e}")

# Check if required files exist
st.write("## Checking required files:")

files_to_check = [
    'polymer_properties_reference.csv',
    'models/v1/dlo_model.pth',
    'models/v1/dlo_model_scaler.pkl',
    'models/v1/dlo_model_metadata.json'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        st.write(f"✅ {file_path} exists")
    else:
        st.write(f"❌ {file_path} missing")

st.write("## Test complete!") 