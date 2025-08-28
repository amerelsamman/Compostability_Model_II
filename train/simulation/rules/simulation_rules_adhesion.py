#!/usr/bin/env python3
"""Adhesion blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any


def load_adhesion_data():
    """Load adhesion data"""
    adhesion_data = pd.read_csv('data/adhesion/masterdata.csv')  # Correct path for current directory structure
    
    return adhesion_data


def rule_of_mixtures(compositions: List[float], adhesion_values: List[float]) -> float:
    """Calculate adhesion using rule of mixtures weighted by volume fraction"""
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate adhesion weighted by volume fraction
    blend_adhesion = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        blend_adhesion += comp * adhesion
    
    return blend_adhesion


def inverse_rule_of_mixtures(compositions: List[float], adhesion_values: List[float]) -> float:
    """Calculate adhesion using inverse rule of mixtures"""
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate inverse adhesion
    inverse_adhesion_sum = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        if adhesion > 0:  # Avoid division by zero
            inverse_adhesion_sum += comp / adhesion
    
    # Return the final adhesion
    if inverse_adhesion_sum > 0:
        return 1 / inverse_adhesion_sum
    else:
        return 0


def combined_rule_of_mixtures(compositions: List[float], adhesion_values: List[float], thickness: float) -> float:
    """Calculate adhesion using combined rule: 50% rule of mixtures + 50% inverse rule of mixtures for thin films"""
    if thickness < 30:
        # Use 50/50 combination for thin films
        rom_adhesion = rule_of_mixtures(compositions, adhesion_values)
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, adhesion_values)
        combined_adhesion = 0.5 * rom_adhesion + 0.5 * inv_rom_adhesion
        return combined_adhesion
    else:
        # Use standard rule of mixtures for thicker films
        return rule_of_mixtures(compositions, adhesion_values)


def apply_adhesion_blending_rules(polymers: List[Dict], compositions: List[float], thickness: float) -> float:
    """Apply adhesion blending rules - combined rule for thin films, standard for thicker films"""
    adhesion_values = [p['adhesion'] for p in polymers]
    return combined_rule_of_mixtures(compositions, adhesion_values, thickness)


def scale_adhesion_with_thickness(base_adhesion: float, thickness: float, reference_thickness: float = 20) -> float:
    """Scale adhesion with thickness scaling using fixed 20 μm reference - EXACTLY as original"""
    empirical_exponent = 0.5  # Moderate scaling for balanced thickness sensitivity
    return base_adhesion * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))


def create_adhesion_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create adhesion blend row with thickness scaling and sealing temperature - EXACTLY as original"""
    # Generate random thickness - EXACTLY as original
    thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm - EXACTLY as original
    
    # Determine max sealing temperature for the blend (lowest among polymers) - EXACTLY as original
    sealing_temps = [p.get('sealing_temp', 23.0) for p in polymers]  # Default to 23°C if missing
    blend_max_sealing_temp = min(sealing_temps)  # Lowest sealing temperature determines blend capability
    
    # Use the blend's max sealing temperature directly (no random temperature) - EXACTLY as original
    blend_temperature = blend_max_sealing_temp
    
    # Use combined rule of mixtures for thin films (< 30 μm), standard rule for thicker films - EXACTLY as original
    blend_adhesion = apply_adhesion_blending_rules(polymers, compositions, thickness)
    
    # Debug: Show which rule was used - EXACTLY as original
    if thickness < 30:
        rom_adhesion = rule_of_mixtures(compositions, [p['adhesion'] for p in polymers])
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, [p['adhesion'] for p in polymers])
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm < 30 μm - Using 50/50 combined rule")
        print(f"  Rule of Mixtures: {rom_adhesion:.3f}, Inverse Rule: {inv_rom_adhesion:.3f}, Combined: {blend_adhesion:.3f}")
    else:
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm ≥ 30 μm - Using standard rule of mixtures: {blend_adhesion:.3f}")
    
    # Scale adhesion based on thickness using fixed 20 μm reference - EXACTLY as original
    blend_adhesion = scale_adhesion_with_thickness(blend_adhesion, thickness, reference_thickness=20)
    
    # No temperature scaling needed - we're at the optimal sealing temperature - EXACTLY as original
    
    # Add noise - EXACTLY as original
    noise_level = 0.05  # 5% noise - EXACTLY as original
    blend_adhesion_noisy = blend_adhesion * (1 + np.random.normal(0, noise_level))
    
    # Ensure the results stay positive - EXACTLY as original
    blend_adhesion_noisy = max(blend_adhesion_noisy, 0.1)  # Minimum adhesion of 0.1
    
    # DEBUG: Print the property value to ensure it's not NaN - EXACTLY as original
    if pd.isna(blend_adhesion_noisy) or blend_adhesion_noisy <= 0:
        print(f"WARNING: Invalid property value for blend {blend_number}: {blend_adhesion_noisy}")
        blend_adhesion_noisy = 0.5  # Fallback value
    
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
        'Sealing Temperature (C)': blend_temperature,  # Blend's max sealing temperature - EXACTLY as original
        'property': blend_adhesion_noisy,
        'unit': 'N/15mm'  # Default unit for adhesion - EXACTLY as original
    }
    
    return row
