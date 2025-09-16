"""
Simulation rules and configurations for all properties
This file contains the property-specific configurations and material mapping functions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

# Import property-specific functions
from rules.simulation_rules_ts import load_ts_data, apply_ts_blending_rules, create_ts_blend_row
from rules.simulation_rules_wvtr import load_wvtr_data, apply_wvtr_blending_rules, create_wvtr_blend_row
from rules.simulation_rules_otr import load_otr_data, apply_otr_blending_rules, create_otr_blend_row
from rules.simulation_rules_adhesion import load_adhesion_data, apply_adhesion_blending_rules, create_adhesion_blend_row
from rules.simulation_rules_eab import load_eab_data, apply_eab_blending_rules, create_eab_blend_row
from rules.simulation_rules_cobb import load_cobb_data, apply_cobb_blending_rules, create_cobb_blend_row
from rules.simulation_rules_compost import load_compost_data, apply_compost_blending_rules, create_compost_blend_row

# Import common functions
from simulation_common import (
    load_material_smiles_dict, get_random_polymer_combination, 
    generate_random_composition, run_augmentation_loop, 
    combine_with_original_data, save_augmented_data, create_ml_dataset,
    generate_simple_report
)

def create_material_mapping(property_name: str):
    """Unified material mapping function for all properties"""
    # Load data based on property
    if property_name == 'ts':
        data = load_ts_data()
    elif property_name == 'wvtr':
        data = load_wvtr_data()
    elif property_name == 'otr':
        data = load_otr_data()
    elif property_name == 'adhesion':
        data = load_adhesion_data()
    elif property_name == 'eab':
        data = load_eab_data()
    elif property_name == 'cobb':
        data = load_cobb_data()
    elif property_name == 'compost':
        data = load_compost_data()
    else:
        raise ValueError(f"Unknown property: {property_name}")
    
    smiles_dict = load_material_smiles_dict()
    mapping = {}
    
    # Define immiscible materials (consistent across all properties)
    # Note: EAB no longer uses immiscibility rules
    immiscible_materials = {
        'Bio-PE': ['all'],  # All Bio-PE grades
        'PP': ['all'],       # All PP grades  
        'PET': ['all'],      # All PET grades
        'PA': ['all'],       # All PA grades
        'EVOH': ['all']      # All EVOH grades
    }
    
    for _, row in data.iterrows():
        material = row['Materials']
        grade = row['Polymer Grade 1']
        
        # Determine if material is immiscible
        is_immiscible = False
        if material in immiscible_materials:
            if immiscible_materials[material] == ['all'] or grade in immiscible_materials[material]:
                is_immiscible = True
        
        # Find corresponding SMILES
        mask = (smiles_dict['Material'] == material) & (smiles_dict['Grade'] == grade)
        if mask.any():
            smiles = smiles_dict[mask]['SMILES'].iloc[0]
            
            # Create base polymer data
            polymer_data = {
                'material': material,
                'grade': grade,
                'smiles': smiles,
                'is_immiscible': is_immiscible
            }
            
            # Add property-specific fields
            if property_name == 'ts':
                # TS has two properties: property1 and property2
                polymer_data.update({
                    'ts1': row['property1'],
                    'ts2': row['property2'],
                    'type': row['type']  # Material type for blending rules
                })
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'cobb':
                # Cobb has single property
                polymer_data['cobb'] = row['property']
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'wvtr':
                # WVTR has single property
                polymer_data['wvtr'] = row['property']
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'otr':
                # OTR has single property
                polymer_data['otr'] = row['property']
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'adhesion':
                # Adhesion has single property: property (sealing strength/adhesion strength)
                polymer_data.update({
                    'adhesion': row['property']  # Sealing strength (adhesion strength)
                })
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'eab':
                # EAB has single property
                polymer_data['eab'] = row['property1']  # EAB uses property1 for both directions
                polymer_data['type'] = row['type']  # Add type field for EAB blending rules
                if 'Thickness (um)' in row:
                    polymer_data['thickness'] = row['Thickness (um)']
                    
            elif property_name == 'compost':
                # EOL has two properties: max_L and t0
                polymer_data.update({
                    'max_L': row['max_L'],
                    't0': row['t0']
                })
            
            mapping[f"{material}_{grade}"] = polymer_data
    
    return mapping

# Property configurations with all required functions
PROPERTY_CONFIGS = {
    'ts': {
        'name': 'Tensile Strength',
        'load_data_func': load_ts_data,
        'create_material_mapping': lambda: create_material_mapping('ts'),
        'create_blend_row_func': create_ts_blend_row
    },
    'wvtr': {
        'name': 'Water Vapor Transmission Rate',
        'load_data_func': load_wvtr_data,
        'create_material_mapping': lambda: create_material_mapping('wvtr'),
        'create_blend_row_func': create_wvtr_blend_row
    },
    'otr': {
        'name': 'Oxygen Transmission Rate',
        'load_data_func': load_otr_data,
        'create_material_mapping': lambda: create_material_mapping('otr'),
        'create_blend_row_func': create_otr_blend_row
    },
    'adhesion': {
        'name': 'Adhesion',
        'load_data_func': load_adhesion_data,
        'create_material_mapping': lambda: create_material_mapping('adhesion'),
        'create_blend_row_func': create_adhesion_blend_row
    },
    'eab': {
        'name': 'Elongation at Break',
        'load_data_func': load_eab_data,
        'create_material_mapping': lambda: create_material_mapping('eab'),
        'create_blend_row_func': create_eab_blend_row
    },
    'cobb': {
        'name': 'Cobb Angle',
        'load_data_func': load_cobb_data,
        'create_material_mapping': lambda: create_material_mapping('cobb'),
        'create_blend_row_func': create_cobb_blend_row
    },
    'compost': {
        'name': 'Compostability (EOL)',
        'load_data_func': load_compost_data,
        'create_material_mapping': lambda: create_material_mapping('compost'),
        'create_blend_row_func': create_compost_blend_row
    }
}
