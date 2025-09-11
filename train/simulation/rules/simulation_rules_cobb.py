#!/usr/bin/env python3
"""Cobb blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_cobb_data():
    """Load Cobb data"""
    cobb_data = pd.read_csv('data/cobb/masterdata.csv')
    
    # Add Thickness column if missing
    if 'Thickness (um)' not in cobb_data.columns:
        cobb_data['Thickness (um)'] = 25.0
    
    # Ensure column order
    columns = list(cobb_data.columns)
    if 'Thickness (um)' in columns and 'property' in columns:
        thickness_idx = columns.index('Thickness (um)')
        property_idx = columns.index('property')
        if thickness_idx > property_idx:
            columns.remove('Thickness (um)')
            property_idx = columns.index('property')
            columns.insert(property_idx, 'Thickness (um)')
            cobb_data = cobb_data[columns]
    
    return cobb_data


def apply_cobb_blending_rules(polymers: List[Dict], compositions: List[float]) -> float:
    """Apply Cobb blending rules - inverse rule of mixtures"""
    cobb_values = [p['cobb'] for p in polymers]
    return 1 / sum(comp / cobb for comp, cobb in zip(compositions, cobb_values) if cobb > 0)


def scale_cobb_with_fixed_thickness(base_cobb: float, thickness: float, reference_thickness: float = 25) -> float:
    """Scale Cobb with empirical power law using fixed 25μm reference
    Cobb decreases with thickness, so we use exponent 0.15 (opposite to EAB)"""
    empirical_exponent = 0.15  # Cobb decreases with thickness
    return base_cobb * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))


def create_cobb_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create Cobb blend row with thickness scaling - clean simulation"""
    # Generate random thickness - EXACTLY as original
    thickness = np.random.uniform(10, 300)  # Thickness between 10-300 μm - EXACTLY as original
    
    # Apply blending rules
    blend_cobb = apply_cobb_blending_rules(polymers, compositions)
    
    # Debug: Show which rule was used - EXACTLY as original
    print(f"Blend {blend_number}: Using inverse rule of mixtures for Cobb angle")
    
    # Thickness scaling - EXACTLY as original
    # Cobb decreases with thickness, so we use exponent 0.15 (opposite to EAB)
    blend_cobb = scale_cobb_with_fixed_thickness(blend_cobb, thickness, 25)
    
    # No noise added - clean simulation
    blend_cobb_final = blend_cobb
    
    # DEBUG: Print the property value to ensure it's not NaN
    if pd.isna(blend_cobb_final) or blend_cobb_final <= 0:
        print(f"WARNING: Invalid Cobb value for blend {blend_number}: {blend_cobb_final}")
        blend_cobb_final = 5.0  # Fallback value
    
    # Fill polymer grades - EXACTLY as original
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES - EXACTLY as original
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(compositions))
    
    # Fill volume fractions - EXACTLY as original
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Create complete row with all required columns - EXACTLY as original
    row = {
        'Materials': str(blend_number),  # Use blend number for Materials column - EXACTLY as original
        'Polymer Grade 1': grades[0],
        'Polymer Grade 2': grades[1],
        'Polymer Grade 3': grades[2],
        'Polymer Grade 4': grades[3],
        'Polymer Grade 5': grades[4],
        'SMILES1': smiles[0],
        'SMILES2': smiles[1],
        'SMILES3': smiles[2],
        'SMILES4': smiles[3],
        'SMILES5': smiles[4],
        'vol_fraction1': vol_fractions[0],
        'vol_fraction2': vol_fractions[1],
        'vol_fraction3': vol_fractions[2],
        'vol_fraction4': vol_fractions[3],
        'vol_fraction5': vol_fractions[4],
        'Thickness (um)': thickness,
        'property': blend_cobb_final
    }
    
    return row
