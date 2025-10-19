#!/usr/bin/env python3
"""EAB blending rules only - everything else is common"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


def load_eab_data():
    """Load EAB data"""
    return pd.read_csv('train/data/eab/masterdata.csv')


def apply_eab_blending_rules(polymers: List[Dict], compositions: List[float], selected_rules: Dict[str, bool] = None) -> Tuple[float, float]:
    """Apply EAB blending rules based on selected rules configuration"""
    brittle_types = ['brittle']
    soft_flex_types = ['soft flex']
    
    has_brittle = any(p['type'] in brittle_types for p in polymers)
    has_soft_flex = any(p['type'] in soft_flex_types for p in polymers)
    
    # If no rules specified, use default behavior (all rules enabled)
    if selected_rules is None:
        if has_brittle and has_soft_flex:
            # Calculate brittle fraction
            brittle_fraction = sum(comp for comp, p in zip(compositions, polymers) if p['type'] == 'brittle')
            
            if brittle_fraction > 0.4:
                # >40% brittle: Use inverse rule of mixtures
                eab1_values = [p['eab'] for p in polymers]
                eab2_values = [p['eab'] for p in polymers]  # EAB uses same value for both directions
                
                blend_eab1 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab1_values) if eab > 0)
                blend_eab2 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab2_values) if eab > 0)
            elif brittle_fraction > 0.2:
                # 20% < brittle < 40%: Use 0.5 regular + 0.5 inverse rule of mixtures
                eab1_values = [p['eab'] for p in polymers]
                eab2_values = [p['eab'] for p in polymers]  # EAB uses same value for both directions
                
                # Regular rule of mixtures
                regular_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
                regular_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
                
                # Inverse rule of mixtures
                inverse_eab1 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab1_values) if eab > 0)
                inverse_eab2 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab2_values) if eab > 0)
                
                # Blend: 0.5 regular + 0.5 inverse
                blend_eab1 = 0.5 * regular_eab1 + 0.5 * inverse_eab1
                blend_eab2 = 0.5 * regular_eab2 + 0.5 * inverse_eab2
            else:
                # <20% brittle: Use regular rule of mixtures
                blend_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
                blend_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
        else:
            # Regular rule of mixtures
            blend_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
            blend_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
    else:
        # Check which rules are enabled
        use_inverse_rom_brittle_soft = selected_rules.get('inverse_rom_brittle_soft', True)
        use_regular_rom = selected_rules.get('regular_rom', True)
        
        # Apply rules based on material types and enabled rules
        if has_brittle and has_soft_flex and use_inverse_rom_brittle_soft:
            # Inverse rule for brittle + soft flex
            eab1_values = [p['eab'] for p in polymers]
            eab2_values = [p['eab'] for p in polymers]  # EAB uses same value for both directions
            
            blend_eab1 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab1_values) if eab > 0)
            blend_eab2 = 1 / sum(comp / eab for comp, eab in zip(compositions, eab2_values) if eab > 0)
        elif use_regular_rom:
            # Regular rule of mixtures
            blend_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
            blend_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
        else:
            # Fallback to regular rule if no specific rule is enabled
            blend_eab1 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
            blend_eab2 = sum(comp * p['eab'] for comp, p in zip(compositions, polymers))
    
    return blend_eab1, blend_eab2


def create_eab_blend_row(polymers: List[Dict], compositions: List[float], blend_number: int, rule_tracker=None, selected_rules: Dict[str, bool] = None, environmental_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create EAB blend row with thickness scaling - clean simulation"""
    
    # Apply blending rules with selected rules
    blend_eab1, blend_eab2 = apply_eab_blending_rules(polymers, compositions, selected_rules)
    
    # Track rule usage based on selected rules and material types
    if rule_tracker is not None:
        brittle_types = ['brittle']
        soft_flex_types = ['soft flex']
        has_brittle = any(p['type'] in brittle_types for p in polymers)
        has_soft_flex = any(p['type'] in soft_flex_types for p in polymers)
        
        if selected_rules is None:
            # Default behavior - track based on material types and brittle fraction
            if has_brittle and has_soft_flex:
                # Calculate brittle fraction for tracking
                brittle_fraction = sum(comp for comp, p in zip(compositions, polymers) if p['type'] == 'brittle')
                if brittle_fraction > 0.4:
                    rule_tracker.record_rule_usage("Inverse Rule of Mixtures (brittle + soft flex, >40% brittle)")
                elif brittle_fraction > 0.2:
                    rule_tracker.record_rule_usage("Mixed Rule of Mixtures (brittle + soft flex, 20-40% brittle)")
                else:
                    rule_tracker.record_rule_usage("Regular Rule of Mixtures (brittle + soft flex, <20% brittle)")
            else:
                rule_tracker.record_rule_usage("Regular Rule of Mixtures")
        else:
            # Track based on which rules are actually enabled and used
            if has_brittle and has_soft_flex and selected_rules.get('inverse_rom_brittle_soft', True):
                rule_tracker.record_rule_usage("Inverse Rule of Mixtures (brittle + soft flex)")
            elif selected_rules.get('regular_rom', True):
                rule_tracker.record_rule_usage("Regular Rule of Mixtures")
            else:
                # Fallback rule
                rule_tracker.record_rule_usage("Regular Rule of Mixtures (fallback)")
    
    # Thickness scaling using environmental controls
    if environmental_config and 'eab' in environmental_config:
        env_params = environmental_config['eab']
        thickness_config = env_params['thickness']
        
        # Use deterministic thickness when min == max, otherwise random
        if thickness_config['min'] == thickness_config['max']:
            thickness = thickness_config['min']
        else:
            thickness = np.random.uniform(thickness_config['min'], thickness_config['max'])
        
        # Apply thickness scaling using config parameters
        power_law = thickness_config['power_law']
        reference_thickness = thickness_config['reference']
        
        blend_eab1 = blend_eab1 * ((thickness ** power_law) / (reference_thickness ** power_law))
        blend_eab2 = blend_eab2 * ((thickness ** power_law) / (reference_thickness ** power_law))
    else:
        # Fallback to original hardcoded values
        thickness = np.random.uniform(10, 300)
        empirical_exponent = 0.4
        reference_thickness = 25
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
