#!/usr/bin/env python3
"""TS blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


def load_ts_data():
    """Load TS data"""
    return pd.read_csv('data/ts/masterdata.csv')


def apply_ts_blending_rules(polymers: List[Dict], compositions: List[float]) -> Tuple[float, float]:
    """Apply TS-specific blending rules - EXACTLY as original"""
    # Check if there is a coincidence of 'brittle' and 'soft flex' materials in the blend - EXACTLY as original
    has_brittle = any(p['type'] == 'brittle' for p in polymers)
    has_soft_flex = any(p['type'] == 'soft flex' for p in polymers)
    has_hard = any(p['type'] == 'hard' for p in polymers)
    brittle_soft_flex_coincidence = has_brittle and has_soft_flex
    hard_soft_flex_coincidence = has_hard and has_soft_flex
    
    # Choose rule of mixtures based on material types - EXACTLY as original
    if brittle_soft_flex_coincidence or hard_soft_flex_coincidence:
        # Use inverse rule of mixtures if there's a coincidence of brittle/soft flex or hard/soft flex materials
        ts1_values = [p['ts1'] for p in polymers]
        blend_ts1 = 1 / sum(comp / ts for comp, ts in zip(compositions, ts1_values) if ts > 0)
        
        ts2_values = [p['ts2'] for p in polymers]
        blend_ts2 = 1 / sum(comp / ts for comp, ts in zip(compositions, ts2_values) if ts > 0)
    else:
        # Use regular rule of mixtures otherwise
        ts1_values = [p['ts1'] for p in polymers]
        blend_ts1 = sum(comp * p['ts1'] for comp, p in zip(compositions, polymers))
        
        ts2_values = [p['ts2'] for p in polymers]
        blend_ts2 = sum(comp * p['ts2'] for comp, p in zip(compositions, polymers))
    
    return blend_ts1, blend_ts2


def create_ts_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create TS blend row with thickness scaling and noise - EXACTLY as original"""
    # Generate random thickness (only environmental parameter for TS) - EXACTLY as original
    thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm - EXACTLY as original
    
    # Apply blending rules
    blend_ts1, blend_ts2 = apply_ts_blending_rules(polymers, compositions)
    
    # Scale TS1 based on thickness using fixed reference - EXACTLY as original
    # Based on validation data analysis: TS scales as thickness^0.1687
    empirical_exponent = 0.125  # From validation data analysis - EXACTLY as original
    blend_ts1 = blend_ts1 * ((thickness ** empirical_exponent) / (25 ** empirical_exponent))
    
    # Scale TS2 based on thickness using fixed reference - EXACTLY as original
    blend_ts2 = blend_ts2 * ((thickness ** empirical_exponent) / (25 ** empirical_exponent))
    
    # Apply miscibility rule: if 30% or more of blend is immiscible components, 
    # both TS1 and TS2 become random values between 5-7 MPa (phase separation) - EXACTLY as original
    immiscible_volume_fraction = 0
    for i, polymer in enumerate(polymers):
        if polymer['is_immiscible']:
            immiscible_volume_fraction += compositions[i]
    
    if immiscible_volume_fraction >= 0.3:  # 30% threshold - EXACTLY as original
        blend_ts1 = np.random.uniform(5.0, 7.0)  # Random TS1 between 5-7 MPa
        blend_ts2 = np.random.uniform(5.0, 7.0)  # Random TS2 between 5-7 MPa
    
    # Add 5% Gaussian noise to make the data more realistic - EXACTLY as original
    noise_level = 0.05  # 5% noise - EXACTLY as original
    blend_ts1_noisy = blend_ts1 * (1 + np.random.normal(0, noise_level))
    blend_ts2_noisy = blend_ts2 * (1 + np.random.normal(0, noise_level))
    
    # Ensure the results stay positive - EXACTLY as original
    blend_ts1_noisy = max(blend_ts1_noisy, 1.0)  # Minimum TS of 1 MPa
    blend_ts2_noisy = max(blend_ts2_noisy, 1.0)  # Minimum TS of 1 MPa
    
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
        'Thickness (um)': thickness,
        'property1': blend_ts1_noisy,
        'property2': blend_ts2_noisy
    }
    
    return row
