"""
Data Loading Module
Functions for loading validation data, material families, and existing KI values
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import re


def load_validation_data(property_name: str, last_n_testing: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation blends for a specific property and split into training/testing sets."""
    validation_path = f"train/data/{property_name}/validationblends.csv"
    
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation data not found: {validation_path}")
    
    print(f"Loading validation data from: {validation_path}")
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded {len(validation_df)} validation blends")
    
    if last_n_testing > 0:
        if last_n_testing >= len(validation_df):
            raise ValueError(f"last_n_testing ({last_n_testing}) must be less than total validation blends ({len(validation_df)})")
        
        # Split data: last N for testing, rest for training
        training_df = validation_df.iloc[:-last_n_testing].copy()
        testing_df = validation_df.iloc[-last_n_testing:].copy()
        
        print(f"Data split:")
        print(f"  Training set: {len(training_df)} blends")
        print(f"  Testing set: {len(testing_df)} blends (last {last_n_testing})")
        
        return training_df, testing_df
    else:
        print(f"Using all {len(validation_df)} blends for training")
        return validation_df, pd.DataFrame()  # Empty testing set


def load_material_families() -> Dict[str, str]:
    """Load material family mapping from material-smiles-dictionary.csv."""
    try:
        df = pd.read_csv('material-smiles-dictionary.csv')
        family_mapping = {}
        for _, row in df.iterrows():
            grade = row['Grade']
            family = row['Material']
            family_mapping[grade] = family
        return family_mapping
    except Exception as e:
        print(f"❌ Error loading material families: {e}")
        return {}


def load_existing_ki_values(property_name: str) -> Dict[str, float]:
    """Load existing KI values from the compatibility YAML file."""
    compatibility_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    
    try:
        with open(compatibility_file, 'r') as f:
            content = f.read()
        
        # Extract KI values using regex
        ki_values = {}
        pattern = r'"([^"]+)":\s*\{KI:\s*([0-9.-]+)'
        matches = re.findall(pattern, content)
        
        for pair_name, ki_value in matches:
            ki_values[pair_name] = float(ki_value)
        
        print(f"Loaded {len(ki_values)} existing KI values from {compatibility_file}")
        return ki_values
        
    except Exception as e:
        print(f"Warning: Could not load existing KI values: {e}")
        return {}
