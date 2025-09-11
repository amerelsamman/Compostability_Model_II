#!/usr/bin/env python3
"""EAB blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


def load_eab_data():
    """Load EAB data"""
    return pd.read_csv('train/data/eab/masterdata.csv')


def apply_eab_blending_rules(polymers: List[Dict], compositions: List[float]) -> Tuple[float, float]:
    """Apply EAB blending rules - immiscibility rule removed"""
    brittle_types = ['brittle']
    soft_flex_types = ['soft flex']
    
    has_brittle = any(p['type'] in brittle_types for p in polymers)
    has_soft_flex = any(p['type'] in soft_flex_types for p in polymers)
    
    if has_brittle and has_soft_flex:
        # Inverse rule for brittle + soft flex
        eab1_values = [p['eab'] for p in polymers]
        eab2_values = [p['eab'] for p in polymers]  # EAB uses same value for both directions
        
        blend_eab1 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab1_values) if eab > 0)
        blend_eab2 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab2_values) if eab > 0)
    else:
        # Regular rule of mixtures
        blend_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
        blend_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
    
    return blend_eab1, blend_eab2


def create_eab_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create EAB blend row with thickness scaling - clean simulation"""
    # Generate random thickness - EXACTLY as original
    thickness = np.random.uniform(10, 300)  # Thickness between 10-300 μm - EXACTLY as original
    
    # Apply blending rules
    blend_eab1, blend_eab2 = apply_eab_blending_rules(polymers, compositions)
    
    # Debug: Show which rule was used
    brittle_types = ['brittle']
    soft_flex_types = ['soft flex']
    has_brittle = any(p['type'] in brittle_types for p in polymers)
    has_soft_flex = any(p['type'] in soft_flex_types for p in polymers)
    
    if has_brittle and has_soft_flex:
        print(f"Blend {blend_number}: Using inverse rule of mixtures (brittle + soft flex coincidence)")
    else:
        print(f"Blend {blend_number}: Using regular rule of mixtures")
    
    # Thickness scaling - EXACTLY as original
    # Based on validation data analysis: EAB scales as thickness^0.4 (increased from 0.1687)
    empirical_exponent = 0.4  # Increased from 0.1687 for stronger thickness effect - EXACTLY as original
    reference_thickness = 25  # Fixed 25μm reference - EXACTLY as original
    blend_eab1 = blend_eab1 * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))
    blend_eab2 = blend_eab2 * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))
    
    # No noise added - clean simulation
    blend_eab1_final = blend_eab1
    blend_eab2_final = blend_eab2
    
    # DEBUG: Print the property values to ensure they're not NaN
    if pd.isna(blend_eab1_final) or blend_eab1_final <= 0:
        print(f"WARNING: Invalid EAB1 value for blend {blend_number}: {blend_eab1_final}")
        blend_eab1_final = 5.0  # Fallback value
    
    if pd.isna(blend_eab2_final) or blend_eab2_final <= 0:
        print(f"WARNING: Invalid EAB2 value for blend {blend_number}: {blend_eab2_final}")
        blend_eab2_final = 5.0  # Fallback value
    
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
        'property1': blend_eab1_final,
        'property2': blend_eab2_final
    }
    
    return row
