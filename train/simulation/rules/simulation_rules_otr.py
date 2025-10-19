#!/usr/bin/env python3
"""OTR blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Import scaling functions from common module
from simulation_common import scale_with_dynamic_thickness, scale_with_temperature, scale_with_humidity


def load_otr_data():
    """Load OTR data"""
    otr_data = pd.read_csv('train/data/otr/masterdata.csv')
    
    # Add Thickness column if missing
    if 'Thickness (um)' not in otr_data.columns:
        otr_data['Thickness (um)'] = 25.0
    
    return otr_data


def apply_otr_blending_rules(polymers: List[Dict], compositions: List[float], selected_rules: Dict[str, bool] = None) -> float:
    """Apply OTR blending rules based on selected rules configuration"""
    otr_values = [p['otr'] for p in polymers]
    
    # If no rules specified, use default behavior (all rules enabled)
    if selected_rules is None:
        return 1 / sum(comp / otr for comp, otr in zip(compositions, otr_values) if otr > 0)
    
    # Check which rules are enabled
    use_inverse_rom = selected_rules.get('inverse_rom', True)
    
    # Apply rules based on enabled rules
    if use_inverse_rom:
        return 1 / sum(comp / otr for comp, otr in zip(compositions, otr_values) if otr > 0)
    else:
        # Fallback to regular rule of mixtures if inverse rule is disabled
        return sum(comp * otr for comp, otr in zip(compositions, otr_values))


def create_otr_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int, rule_tracker=None, selected_rules: Dict[str, bool] = None, environmental_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create OTR blend row with temp, humidity, thickness scaling - clean simulation"""
    # Use environmental config if provided, otherwise generate random parameters
    if environmental_config and 'otr' in environmental_config:
        env_params = environmental_config['otr']
        # Use deterministic values when min == max, otherwise random
        if env_params['temperature']['min'] == env_params['temperature']['max']:
            temp = env_params['temperature']['min']
        else:
            temp = np.random.uniform(env_params['temperature']['min'], env_params['temperature']['max'])
        
        if env_params['humidity']['min'] == env_params['humidity']['max']:
            rh = env_params['humidity']['min']
        else:
            rh = np.random.uniform(env_params['humidity']['min'], env_params['humidity']['max'])
        
        if env_params['thickness']['min'] == env_params['thickness']['max']:
            thickness = env_params['thickness']['min']
        else:
            thickness = np.random.uniform(env_params['thickness']['min'], env_params['thickness']['max'])
    else:
        # Fallback to original values
        temp = np.random.uniform(23, 50)  # Temperature between 23-50°C
        rh = np.random.uniform(50, 95)    # RH between 50-95%
        thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm
    
    # Apply blending rules with selected rules
    blend_otr = apply_otr_blending_rules(polymers, compositions, selected_rules)
    
    # Track rule usage based on selected rules
    if rule_tracker is not None:
        if selected_rules is None:
            # Default behavior - track inverse rule
            rule_tracker.record_rule_usage("Inverse Rule of Mixtures (OTR)")
        else:
            # Track based on which rules are actually enabled
            if selected_rules.get('inverse_rom', True):
                rule_tracker.record_rule_usage("Inverse Rule of Mixtures (OTR)")
            else:
                rule_tracker.record_rule_usage("Regular Rule of Mixtures (OTR)")
    
    # Scale OTR based on environmental conditions using config parameters
    if environmental_config and 'otr' in environmental_config:
        env_params = environmental_config['otr']
        thickness_config = env_params['thickness']
        temp_config = env_params['temperature']
        humidity_config = env_params['humidity']
        
        # Thickness scaling
        blend_otr = scale_with_dynamic_thickness(blend_otr, thickness, polymers, compositions, 
                                               thickness_config['power_law'], thickness_config['reference'])
        
        # Temperature scaling
        blend_otr = scale_with_temperature(blend_otr, temp, temp_config['reference'], temp_config['max_scale'], temp_config.get('divisor', 10))
        
        # Humidity scaling
        blend_otr = scale_with_humidity(blend_otr, rh, humidity_config['reference'], humidity_config['max_scale'], humidity_config.get('divisor', 20))
    else:
        # Fallback to original scaling
        blend_otr = scale_with_dynamic_thickness(blend_otr, thickness, polymers, compositions, 0.1, 25)
        blend_otr = scale_with_temperature(blend_otr, temp, 23, 5)
        blend_otr = scale_with_humidity(blend_otr, rh, 50, 3)
    
    # No noise added - clean simulation
    blend_otr_final = blend_otr
    
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
        'property': blend_otr_final,
        'unit': 'cc*um/m2/day'
    }
    
    return row
