#!/usr/bin/env python3
"""
Simple test Streamlit app without RDKit dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

st.set_page_config(
    page_title="Polymer Blend Prediction Test",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Polymer Blend Prediction Test")
st.write("This is a simplified test version without RDKit dependencies.")

# Check if required files exist
st.write("## Checking required files:")

files_to_check = [
    'polymer_properties_reference.csv',
    'models/v1/dlo_model.pth',
    'models/v1/dlo_model_scaler.pkl',
    'models/v1/dlo_model_metadata.json'
]

all_files_exist = True
for file_path in files_to_check:
    if os.path.exists(file_path):
        st.success(f"✅ {file_path} exists")
    else:
        st.error(f"❌ {file_path} missing")
        all_files_exist = False

if all_files_exist:
    st.success("All required files are present!")
else:
    st.error("Some required files are missing. Please check the file structure.")

# Test basic imports
st.write("## Testing basic imports:")

try:
    import pandas as pd
    st.success("✅ pandas imported successfully")
except Exception as e:
    st.error(f"❌ Error importing pandas: {e}")

try:
    import numpy as np
    st.success("✅ numpy imported successfully")
except Exception as e:
    st.error(f"❌ Error importing numpy: {e}")

try:
    import torch
    st.success("✅ torch imported successfully")
except Exception as e:
    st.error(f"❌ Error importing torch: {e}")

try:
    import sklearn
    st.success("✅ scikit-learn imported successfully")
except Exception as e:
    st.error(f"❌ Error importing scikit-learn: {e}")

try:
    import matplotlib
    st.success("✅ matplotlib imported successfully")
except Exception as e:
    st.error(f"❌ Error importing matplotlib: {e}")

# Test RDKit import (this will likely fail)
st.write("## Testing RDKit import:")
try:
    import rdkit
    st.success("✅ RDKit imported successfully")
except Exception as e:
    st.warning(f"⚠️ RDKit import failed: {e}")
    st.info("This is expected on Streamlit Cloud. We'll need to handle this differently.")

st.write("## Test complete!")
st.write("If you see this message, the basic Streamlit setup is working.") 