"""
Simulation Engine Module
Functions for blend simulation and gradient calculation
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# Add the train/simulation directory to the path
sys.path.append('train/simulation')

from simulation_common import set_random_seeds, load_environmental_controls_config, get_environmental_parameters, load_polymer_corrections_config, apply_umm3_corrections
from simulation_rules import PROPERTY_CONFIGS
from umm3_correction import UMM3Correction, load_family_compatibility_config

from .family_manager import extract_polymer_families_from_blend, get_ki_overrides_from_vector


def simulate_validation_blend(blend_row: pd.Series, property_name: str, family_mapping: Dict[str, str],
                            ki_overrides: Dict[str, float]) -> Optional[float]:
    """Simulate a single validation blend with KI overrides."""
    try:
        # Extract polymer families from this blend
        families = extract_polymer_families_from_blend(blend_row, family_mapping)
        
        if len(families) < 2:
            return None
        
        # Get the first grade from each family for simulation
        property_config = PROPERTY_CONFIGS[property_name]
        material_mapping = property_config['create_material_mapping'](enable_additives=False)
        
        # Find representative grades for each family
        polymer_data_list = []
        compositions = []
        
        for i in range(1, 6):  # Polymer Grade 1-5
            grade_col = f'Polymer Grade {i}'
            vol_frac_col = f'vol_fraction{i}'
            
            if pd.notna(blend_row[grade_col]) and pd.notna(blend_row[vol_frac_col]):
                grade = blend_row[grade_col]
                vol_frac = blend_row[vol_frac_col]
                
                if grade in family_mapping:
                    family = family_mapping[grade]
                    # Find a representative grade for this family
                    family_grades = [k for k in material_mapping.keys() if k.startswith(f"{family}_")]
                    
                    if family_grades:
                        # Use the first available grade for this family
                        grade_key = family_grades[0]
                        polymer_data = material_mapping[grade_key]
                        polymer_data_list.append(polymer_data)
                        compositions.append(float(vol_frac))
        
        if len(polymer_data_list) < 2:
            return None
        
        # Normalize compositions
        total_comp = sum(compositions)
        compositions = [c / total_comp for c in compositions]
        
        # Use validation data for environmental parameters - let internal simulation system handle scaling
        validation_thickness = blend_row['Thickness (um)']
        
        # Load environmental controls from internal simulation system
        try:
            environmental_controls_config = load_environmental_controls_config()
            base_env_config = environmental_controls_config[property_name]
        except Exception as e:
            print(f"Warning: Could not load environmental controls config: {e}")
            # Fallback to basic config
            base_env_config = {}
        
        # Create environmental config with validation data, using internal system parameters
        env_config = {property_name: {}}
        
        # Handle thickness - always use validation thickness
        if 'thickness' in base_env_config:
            thickness_config = base_env_config['thickness'].copy()
            thickness_config['min'] = validation_thickness
            thickness_config['max'] = validation_thickness
            env_config[property_name]['thickness'] = thickness_config
        else:
            # Fallback if no thickness config found - use same structure as optimized config
            env_config[property_name]['thickness'] = {
                'min': validation_thickness, 
                'max': validation_thickness, 
                'power_law': 0.4, 
                'reference': 25.0, 
                'scaling_type': 'piecewise',
                'piecewise': {
                    'thin_regime': {
                        'max_thickness': 10.0,
                        'power_law': -1.3698586602730032,
                        'reference': 9.667362897949074
                    },
                    'thick_regime': {
                        'min_thickness': 10.0,
                        'power_law': -9.84332914939351e-05,
                        'reference': 10.006408994765495
                    }
                }
            }
        
        # Handle temperature and humidity for WVTR/OTR properties
        if property_name in ['wvtr', 'otr']:
            validation_temp = blend_row['Temperature (C)']
            validation_rh = blend_row['RH (%)']
            
            # Use internal system temperature config
            if 'temperature' in base_env_config:
                temp_config = base_env_config['temperature'].copy()
                temp_config['min'] = validation_temp
                temp_config['max'] = validation_temp
                env_config[property_name]['temperature'] = temp_config
            
            # Use internal system humidity config
            if 'humidity' in base_env_config:
                humidity_config = base_env_config['humidity'].copy()
                humidity_config['min'] = validation_rh
                humidity_config['max'] = validation_rh
                env_config[property_name]['humidity'] = humidity_config
        
        
        # Create blend row
        create_blend_row_func = property_config['create_blend_row_func']
        blend_row_data = create_blend_row_func(
            polymers=polymer_data_list,
            compositions=compositions,
            blend_number=1,
            rule_tracker=None,
            selected_rules=None,
            environmental_config=env_config
        )
        
        # Apply polymer corrections using internal simulation system
        try:
            # Load polymer corrections config
            polymer_corrections_config = load_polymer_corrections_config()
            
            # Load UMM3 correction system
            umm3_correction = UMM3Correction.from_config_files("train/simulation/config")
            
            # Extract property values for correction
            property_values = {}
            for key, value in blend_row_data.items():
                if key.startswith('property') and isinstance(value, (int, float)):
                    property_values[key] = value
            
            # Apply UMM3 corrections to all materials
            if property_values:
                corrected_property_values = apply_umm3_corrections(
                    property_values, property_name, polymer_data_list, compositions, 
                    umm3_correction, None, polymer_corrections_config, None
                )
                
                # Update the blend_row_data with corrected values
                for key, value in corrected_property_values.items():
                    if key != 'corrections_applied':  # Handle corrections_applied separately
                        blend_row_data[key] = value
                        
        except Exception as e:
            raise RuntimeError(f"Failed to apply polymer corrections: {e}")
        
        
        
        # Apply UMM3 correction if KI overrides are provided
        if ki_overrides:
            umm3 = UMM3Correction.from_config_files()
            umm3.ki_overrides = ki_overrides
            
            family_config = load_family_compatibility_config(
                config_dir="train/simulation/config/compatibility", 
                property_name=property_name
            )
            
            # Get property values - handle different property key formats
            property_values = {}
            if property_name in ['ts', 'eab']:
                # TS and EAB use property1, property2 keys
                for key in ['property1', 'property2']:
                    if key in blend_row_data and isinstance(blend_row_data[key], (int, float)):
                        property_values[key] = blend_row_data[key]
            else:
                # Other properties use 'property' key
                if 'property' in blend_row_data and isinstance(blend_row_data['property'], (int, float)):
                    property_values[property_name] = blend_row_data['property']
            
            if property_values:
                corrected_property_values = umm3.apply_pairwise_compatibility_corrections(
                    property_values, polymer_data_list, compositions, 
                    family_config, property_name
                )
                
                if property_name == 'ts':
                    return corrected_property_values.get('property1', blend_row_data['property1'])
                elif property_name == 'eab':
                    return corrected_property_values.get('property1', blend_row_data['property1'])
                else:
                    return corrected_property_values.get(property_name, blend_row_data['property'])
            else:
                if property_name == 'ts':
                    return blend_row_data['property1']
                elif property_name == 'eab':
                    return blend_row_data['property1']
                else:
                    return blend_row_data['property']
        else:
            if property_name == 'ts':
                return blend_row_data['property1']
            elif property_name == 'eab':
                return blend_row_data['property1']
            else:
                return blend_row_data['property']
                
    except Exception as e:
        print(f"    Error simulating blend: {e}")
        return None


def calculate_blend_specific_gradients(validation_df: pd.DataFrame, property_name: str, family_mapping: Dict[str, str],
                                     ki_vector: np.ndarray, pair_mapping: Dict[str, int], 
                                     epsilon: float = 1e-6) -> Tuple[np.ndarray, List[List[float]], List[float]]:
    """Calculate gradients for each KI parameter based on individual blend errors."""
    n_pairs = len(ki_vector)
    n_blends = len(validation_df)
    
    # Calculate current errors for all blends
    current_errors = []
    for idx, row in validation_df.iterrows():
        families = extract_polymer_families_from_blend(row, family_mapping)
        if len(families) < 2:
            current_errors.append(0.0)
            continue
            
        ki_overrides = get_ki_overrides_from_vector(ki_vector, pair_mapping, families)
        simulated_value = simulate_validation_blend(row, property_name, family_mapping, ki_overrides)
        
        if simulated_value is not None:
            if property_name in ['ts', 'eab']:
                exp_value = row['property1']
            else:
                exp_value = row['property']
            
            if pd.notna(exp_value):
                error = abs(simulated_value - exp_value)
                current_errors.append(error)
            else:
                current_errors.append(0.0)
        else:
            current_errors.append(0.0)
    
    # Calculate gradients for each KI parameter
    gradients = np.zeros(n_pairs)
    blend_gradients = []
    
    for i in range(n_pairs):
        # Forward difference for this parameter
        ki_vector_plus = ki_vector.copy()
        ki_vector_plus[i] += epsilon
        
        # Calculate errors with perturbed KI
        plus_errors = []
        for idx, row in validation_df.iterrows():
            families = extract_polymer_families_from_blend(row, family_mapping)
            if len(families) < 2:
                plus_errors.append(0.0)
                continue
                
            ki_overrides = get_ki_overrides_from_vector(ki_vector_plus, pair_mapping, families)
            simulated_value = simulate_validation_blend(row, property_name, family_mapping, ki_overrides)
            
            if simulated_value is not None:
                if property_name in ['ts', 'eab']:
                    exp_value = row['property1']
                else:
                    exp_value = row['property']
                
                if pd.notna(exp_value):
                    error = abs(simulated_value - exp_value)
                    plus_errors.append(error)
                else:
                    plus_errors.append(0.0)
            else:
                plus_errors.append(0.0)
        
        # Calculate gradient as average of individual blend gradients
        blend_grads = [(plus_errors[j] - current_errors[j]) / epsilon for j in range(n_blends)]
        blend_gradients.append(blend_grads)
        gradients[i] = np.mean(blend_grads)
    
    return gradients, blend_gradients, current_errors


def calculate_validation_mae(validation_df: pd.DataFrame, property_name: str, family_mapping: Dict[str, str],
                           ki_vector: np.ndarray, pair_mapping: Dict[str, int]) -> Tuple[float, List[float], List[float]]:
    """Calculate MAE across all validation blends."""
    errors = []
    simulated_values = []
    
    for idx, row in validation_df.iterrows():
        # Get families in this blend
        families = extract_polymer_families_from_blend(row, family_mapping)
        
        if len(families) < 2:
            continue
        
        # Get KI overrides for this blend
        ki_overrides = get_ki_overrides_from_vector(ki_vector, pair_mapping, families)
        
        # Simulate the blend
        simulated_value = simulate_validation_blend(row, property_name, family_mapping, ki_overrides)
        
        if simulated_value is not None:
            # Get experimental value
            if property_name in ['ts', 'eab']:
                exp_value = row['property1']
            else:
                exp_value = row['property']
            
            if pd.notna(exp_value):
                error = abs(simulated_value - exp_value)
                errors.append(error)
                simulated_values.append(simulated_value)
                
    
    if not errors:
        return float('inf'), [], []
    
    mae = np.mean(errors)
    return mae, errors, simulated_values
