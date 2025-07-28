#!/usr/bin/env python3
"""
Local test to verify the code works
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

def main():
    st.set_page_config(
        page_title="Local Test",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("🧬 Local Test - Polymer Blend Prediction")
    st.write("This is a local test to verify the code works.")
    
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
        st.error("Some required files are missing.")
    
    st.write("## Test complete!")
    st.write("If you see this message, the basic setup is working.")

if __name__ == "__main__":
    main() 