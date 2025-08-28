#!/usr/bin/env python3
"""WVTR blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import scaling functions from common module - using original function names
from simulation_common import scale_with_dynamic_thickness, scale_with_temperature, scale_with_humidity


def load_wvtr_data():
    """Load WVTR data"""
    return pd.read_csv('data/wvtr/masterdata.csv')


def apply_wvtr_blending_rules(polymers: List[Dict], compositions: List[float]) -> float:
    """Apply WVTR blending rules - inverse rule of mixtures"""
    wvtr_values = [p['wvtr'] for p in polymers]
    return 1 / sum(comp / wvtr for comp, wvtr in zip(compositions, wvtr_values) if wvtr > 0)


def create_wvtr_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create WVTR blend row with temp, humidity, thickness scaling and noise - EXACTLY as original"""
    # Generate random environmental parameters - EXACTLY as original
    temp = np.random.uniform(23, 50)  # Temperature between 23-50°C - EXACTLY as original
    rh = np.random.uniform(50, 95)    # RH between 50-95% - EXACTLY as original
    thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm - EXACTLY as original
    
    # Apply blending rules
    blend_wvtr = apply_wvtr_blending_rules(polymers, compositions)
    
    # Scale WVTR based on environmental conditions using dynamic thickness reference - EXACTLY as original
    blend_wvtr = scale_with_dynamic_thickness(blend_wvtr, thickness, polymers, compositions, 0.5, 25)
    blend_wvtr = scale_with_temperature(blend_wvtr, temp, 23)
    blend_wvtr = scale_with_humidity(blend_wvtr, rh, 50)
    
    # Add noise - EXACTLY as original
    noise_level = 0.1  # 10% noise - EXACTLY as original
    blend_wvtr_noisy = blend_wvtr * (1 + np.random.normal(0, noise_level))
    
    # Ensure the result stays positive - EXACTLY as original
    blend_wvtr_noisy = max(blend_wvtr_noisy, 0.01)  # Minimum WVTR of 0.01
    
    # Create complete row with all required columns - EXACTLY as original
    row = {
        'Materials': str(blend_number),  # Use blend number for Materials column - EXACTLY as original
        'Polymer Grade 1': polymers[0]['grade'],
        'Polymer Grade 2': polymers[1]['grade'] if len(polymers) > 1 else 'Unknown',
        'Polymer Grade 3': polymers[2]['grade'] if len(polymers) > 2 else 'Unknown',
        'Polymer Grade 4': polymers[3]['grade'] if len(polymers) > 3 else 'Unknown',
        'Polymer Grade 5': polymers[4]['grade'] if len(polymers) > 4 else 'Unknown',
        'SMILES1': polymers[0]['smiles'],
        'SMILES2': polymers[1]['smiles'] if len(polymers) > 1 else '',
        'SMILES3': polymers[2]['smiles'] if len(polymers) > 2 else '',
        'SMILES4': polymers[3]['smiles'] if len(polymers) > 3 else '',
        'SMILES5': polymers[4]['smiles'] if len(polymers) > 4 else '',
        'vol_fraction1': compositions[0],
        'vol_fraction2': compositions[1] if len(compositions) > 1 else 0.0,
        'vol_fraction3': compositions[2] if len(compositions) > 2 else 0.0,
        'vol_fraction4': compositions[3] if len(compositions) > 3 else 0.0,
        'vol_fraction5': compositions[4] if len(compositions) > 4 else 0.0,
        'Temperature (C)': temp,
        'RH (%)': rh,
        'Thickness (um)': thickness,
        'property': blend_wvtr_noisy,
        'unit': 'g*um/m2*day'
    }
    
    return row
