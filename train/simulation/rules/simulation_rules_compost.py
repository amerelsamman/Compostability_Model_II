#!/usr/bin/env python3
"""Compostability blending rules only - everything else is common"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Any, Tuple


def load_compost_data():
    """Load compostability data"""
    return pd.read_csv('data/eol/masterdata.csv')


def rule_of_mixtures(compositions: List[float], values: List[float]) -> float:
    """
    Calculate property using rule of mixtures weighted by volume fraction
    Property_blend = Σ(vol_fraction_i * Property_i)
    """
    if len(compositions) != len(values):
        raise ValueError("Compositions and values must have same length")
    
    # Calculate weighted average
    blend_property = 0
    for comp, prop in zip(compositions, values):
        if pd.notna(prop):  # Handle NaN values
            blend_property += comp * prop
    
    return blend_property


def apply_compost_blending_rules(polymers: List[Dict], compositions: List[float]) -> Tuple[float, float]:
    """Apply compostability-specific rules to determine max_L and t0 - EXACTLY as original"""
    max_L_values = []
    t0_values = []
    pla_fraction = 0.0
    compostable_fraction = 0.0
    non_compostable_fraction = 0.0
    
    # Collect polymer properties and identify PLA - EXACTLY as original
    for polymer, fraction in zip(polymers, compositions):
        max_L = polymer['property1']  # Disintegration level (0-100, >90 = home-compostable)
        t0 = polymer['property2']     # Time to 50% disintegration (days)
        
        max_L_values.append(max_L)
        t0_values.append(t0)
        
        # Check if it's PLA - EXACTLY as original
        if 'PLA' in polymer['material'].upper():
            pla_fraction += fraction
        
        # Check if it's compostable (max_L > 90)
        if max_L > 90:
            compostable_fraction += fraction
        else:
            non_compostable_fraction += fraction
    
    # Apply Rule 1: If all polymers have known values and total fraction = 1.0 (but NOT PLA blends) - EXACTLY as original
    if (len(max_L_values) == len(polymers) and 
        all(pd.notna(max_L) for max_L in max_L_values) and
        all(pd.notna(t0) for t0 in t0_values) and
        abs(sum(compositions) - 1.0) < 0.01 and
        pla_fraction == 0):  # Exclude PLA blends from Rule 1
        
        # If all polymers are home-compostable (max_L > 90)
        if all(max_L > 90 for max_L in max_L_values):
            # Use random max_L between 90-95 for purely home-compostable blends
            max_L_pred = random.uniform(90.0, 95.0)
            # Calculate weighted average t0
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  All polymers home-compostable: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
        else:
            # Use weighted averages for mixed blends
            max_L_pred = rule_of_mixtures(compositions, max_L_values)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  Mixed blend: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
    
    # Apply Rule 2: PLA + Compostable polymer rule - EXACTLY as original
    if (pla_fraction > 0 and 
        compostable_fraction >= 0.15):
        
        # For max_L: Exclude PLA from calculation, use only non-PLA polymers
        non_pla_max_L_values = []
        non_pla_compositions = []
        non_pla_t0_values = []
        
        for i, (polymer, fraction) in enumerate(zip(polymers, compositions)):
            if 'PLA' not in polymer['material'].upper():
                non_pla_max_L_values.append(max_L_values[i])
                non_pla_compositions.append(fraction)
                non_pla_t0_values.append(t0_values[i])
        
        # Normalize non-PLA compositions to sum to 1
        if sum(non_pla_compositions) > 0:
            normalized_compositions = [f / sum(non_pla_compositions) for f in non_pla_compositions]
            
            # Calculate max_L excluding PLA (rule of mixtures on non-PLA polymers only)
            max_L_pred = rule_of_mixtures(normalized_compositions, non_pla_max_L_values)
            
            # For t0: PLA still contributes normally (rule of mixtures on all polymers)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            
            print(f"  PLA rule applies: max_L = {max_L_pred:.2f} (PLA excluded), t0 = {t0_pred:.2f} (PLA included)")
            return max_L_pred, t0_pred
        else:
            # Fallback if somehow no non-PLA polymers
            max_L_pred = rule_of_mixtures(compositions, max_L_values)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  PLA rule fallback: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
    
    # Default: Use rule of mixtures for both properties
    max_L_pred = rule_of_mixtures(compositions, max_L_values)
    t0_pred = rule_of_mixtures(compositions, t0_values)
    print(f"  Default rule of mixtures: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
    return max_L_pred, t0_pred


def create_compost_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int) -> Dict[str, Any]:
    """Create compostability blend row with complex rules - EXACTLY as original"""
    # Apply compostability rules to determine property1 (max_L) and property2 (t0)
    max_L_pred, t0_pred = apply_compost_blending_rules(polymers, compositions)
    
    # No noise added - clean simulation
    max_L_final = max_L_pred
    t0_final = t0_pred
    
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
        'property1': max_L_final,  # max_L - disintegration level
        'property2': t0_final      # t0 - time to 50% disintegration
    }
    
    return row
