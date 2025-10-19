#!/usr/bin/env python3
"""
Blend Optimization Script
Optimizes KI values for polymer-polymer compatibility using validation blend data.
Uses gradient descent to minimize MAE across all validation blends simultaneously.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import warnings
import re
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple

# Add the train/simulation directory to the path
sys.path.append('train/simulation')

warnings.filterwarnings('ignore')

from simulation_common import set_random_seeds, load_environmental_controls_config, get_environmental_parameters, load_polymer_corrections_config, apply_umm3_corrections
from simulation_rules import PROPERTY_CONFIGS
from umm3_correction import UMM3Correction, load_family_compatibility_config


def load_validation_data(property_name: str, last_n_testing: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load validation blends for a specific property and split into training/testing sets."""
    validation_path = f"train/data/{property_name}/validationblends.csv"
    
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation data not found: {validation_path}")
    
    print(f"Loading validation data from: {validation_path}")
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded {len(validation_df)} validation blends")
    
    if last_n_testing > 0:
        if last_n_testing >= len(validation_df):
            raise ValueError(f"last_n_testing ({last_n_testing}) must be less than total validation blends ({len(validation_df)})")
        
        # Split data: last N for testing, rest for training
        training_df = validation_df.iloc[:-last_n_testing].copy()
        testing_df = validation_df.iloc[-last_n_testing:].copy()
        
        print(f"Data split:")
        print(f"  Training set: {len(training_df)} blends")
        print(f"  Testing set: {len(testing_df)} blends (last {last_n_testing})")
        
        return training_df, testing_df
    else:
        print(f"Using all {len(validation_df)} blends for training")
        return validation_df, pd.DataFrame()  # Empty testing set


def load_material_families() -> Dict[str, str]:
    """Load material family mapping from material-smiles-dictionary.csv."""
    try:
        df = pd.read_csv('material-smiles-dictionary.csv')
        family_mapping = {}
        for _, row in df.iterrows():
            grade = row['Grade']
            family = row['Material']
            family_mapping[grade] = family
        return family_mapping
    except Exception as e:
        print(f"❌ Error loading material families: {e}")
        return {}


def extract_polymer_families_from_blend(blend_row: pd.Series, family_mapping: Dict[str, str]) -> List[str]:
    """Extract polymer families from a blend row."""
    families = []
    missing_grades = []
    
    for i in range(1, 6):  # Polymer Grade 1-5
        grade_col = f'Polymer Grade {i}'
        vol_frac_col = f'vol_fraction{i}'
        
        # Only process if grade exists AND has a volume fraction > 0
        if pd.notna(blend_row[grade_col]) and pd.notna(blend_row[vol_frac_col]) and blend_row[vol_frac_col] > 0:
            grade = blend_row[grade_col]
            
            # Skip 'Unknown' grades
            if grade == 'Unknown':
                continue
                
            if grade in family_mapping:
                family = family_mapping[grade]
                if family not in families:
                    families.append(family)
            else:
                missing_grades.append(grade)
    
    # THROW ERROR if any grades are missing from family mapping
    if missing_grades:
        blend_name = blend_row.get('Materials', 'Unknown')
        raise ValueError(f"❌ BLEND {blend_name}: Missing grades in family mapping: {missing_grades}")
    
    return families


def get_polymer_family_groups() -> Dict[str, List[str]]:
    """Define polymer family groups for optimization."""
    return {
        "rigids": ["PLA", "PGA", "PHAs"],
        "brittles": ["PHB", "PHA", "PHBV"], 
        "soft_flex": ["PHBH", "PHAs","PHAa"],
        "good_flex": ["PBAT", "PCL", "PBS","PBSA"],
        "bio_pe": ["Bio-PE"],
        "traditional": ["LDPE", "PP", "PET", "PVDC", "PA", "EVOH"]
    }


def get_family_group_for_polymer(polymer_family: str, family_groups: Dict[str, List[str]]) -> str:
    """Get the family group for a given polymer family."""
    for group_name, families in family_groups.items():
        if polymer_family in families:
            return group_name
    return polymer_family


def load_existing_ki_values(property_name: str) -> Dict[str, float]:
    """Load existing KI values from the compatibility YAML file."""
    import yaml
    import re
    
    compatibility_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    
    try:
        with open(compatibility_file, 'r') as f:
            content = f.read()
        
        # Extract KI values using regex
        ki_values = {}
        pattern = r'"([^"]+)":\s*\{KI:\s*([0-9.-]+)'
        matches = re.findall(pattern, content)
        
        for pair_name, ki_value in matches:
            ki_values[pair_name] = float(ki_value)
        
        print(f"Loaded {len(ki_values)} existing KI values from {compatibility_file}")
        return ki_values
        
    except Exception as e:
        print(f"Warning: Could not load existing KI values: {e}")
        return {}


def build_ki_vector(validation_df: pd.DataFrame, family_mapping: Dict[str, str], property_name: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build KI vector and mapping for polymer pairs found in validation data."""
    print("Building KI vector from validation data...")
    
    # Load existing KI values from YAML file
    existing_ki_values = load_existing_ki_values(property_name)
    
    # Collect all unique polymer families from validation blends
    all_families = set()
    for _, row in validation_df.iterrows():
        families = extract_polymer_families_from_blend(row, family_mapping)
        all_families.update(families)
    
    all_families = sorted(list(all_families))
    print(f"Found {len(all_families)} unique polymer families: {all_families}")
    
    # Find all polymer-polymer pairs that exist in validation blends
    existing_pairs = set()
    blend_pairs = {}  # Track which pairs come from which blends
    
    for _, row in validation_df.iterrows():
        families = extract_polymer_families_from_blend(row, family_mapping)
        blend_name = row.get('Materials', 'Unknown')
        
        # Create all possible pairs from families in this blend
        blend_pairs[blend_name] = []
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                pair = tuple(sorted([families[i], families[j]]))
                existing_pairs.add(pair)
                blend_pairs[blend_name].append(f"{pair[0]}-{pair[1]}")
    
    existing_pairs = sorted(list(existing_pairs))
    print(f"Found {len(existing_pairs)} unique polymer pairs in validation data")
    
    
    # Create KI vector and mapping, loading existing values
    ki_vector = np.zeros(len(existing_pairs))
    pair_mapping = {}
    
    for i, (family1, family2) in enumerate(existing_pairs):
        pair_name = f"{family1}-{family2}"
        # Handle multi-part family names like "Bio-PE" correctly
        if 'Bio-PE' in pair_name:
            if pair_name.startswith('Bio-PE-'):
                reverse_name = pair_name.replace('Bio-PE-', '') + '-Bio-PE'
            else:
                reverse_name = 'Bio-PE-' + pair_name.replace('-Bio-PE', '')
        else:
            reverse_name = f"{family2}-{family1}"
        pair_mapping[pair_name] = i
        
        # Try to load existing KI value (check both directions)
        existing_value = None
        if pair_name in existing_ki_values:
            existing_value = existing_ki_values[pair_name]
        elif reverse_name in existing_ki_values:
            existing_value = existing_ki_values[reverse_name]
        else:
            raise ValueError(f"❌ Pair '{pair_name}' not found in {property_name}_compatibility.yaml. "
                           f"Tried: '{pair_name}' and '{reverse_name}'. "
                           f"Please add this pair to the YAML file before running optimization.")
        
        ki_vector[i] = existing_value
    
    return ki_vector, pair_mapping


def get_ki_overrides_from_vector(ki_vector: np.ndarray, pair_mapping: Dict[str, int], 
                                families_in_blend: List[str]) -> Dict[str, float]:
    """Get KI overrides for a specific blend from the KI vector."""
    ki_overrides = {}
    missing_pairs = []
    
    # Create all possible pairs from families in this blend
    for i in range(len(families_in_blend)):
        for j in range(i + 1, len(families_in_blend)):
            family1, family2 = sorted([families_in_blend[i], families_in_blend[j]])
            pair_name = f"{family1}-{family2}"
            
            if pair_name in pair_mapping:
                ki_value = ki_vector[pair_mapping[pair_name]]
                ki_overrides[pair_name] = ki_value
            else:
                missing_pairs.append(pair_name)
    
    if missing_pairs:
        raise ValueError(f"❌ Missing pairs in KI vector: {missing_pairs}. "
                        f"Available pairs: {list(pair_mapping.keys())}")
    
    return ki_overrides


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
            # Fallback if no thickness config found
            env_config[property_name]['thickness'] = {
                'min': validation_thickness, 'max': validation_thickness, 'power_law': 0.4, 'reference': 25.0, 'scaling_type': 'fixed'
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


def calculate_blend_learning_rates(current_errors: List[float], base_lr: float = 0.1) -> List[float]:
    """Calculate learning rates for each blend based on how close their MAE is to 0."""
    blend_lrs = []
    
    for error in current_errors:
        if error > 1000:
            # Far from target - fast learning
            lr = base_lr
        elif error > 100:
            # Getting closer - medium learning
            lr = base_lr * 0.5
        elif error > 10:
            # Very close - slow learning
            lr = base_lr * 0.1
        elif error > 1:
            # Almost there - very slow learning
            lr = base_lr * 0.01
        else:
            # Extremely close - tiny steps
            lr = base_lr * 0.001
        
        blend_lrs.append(lr)
    
    return blend_lrs


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


def optimize_blend_ki_values(property_name: str, max_iterations: int, learning_rate: float, seed: int, 
                           use_adaptive_lr: bool = True, gradient_threshold: float = 2.0,
                           max_step_size: float = 0.5, training_df: pd.DataFrame = None) -> Tuple[bool, np.ndarray, Dict[str, int]]:
    """Optimize KI values for polymer-polymer compatibility."""
    print(f"🚀 Starting blend optimization for {property_name}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Random seed: {seed}")
    
    # Set random seed
    set_random_seeds(seed)
    
    # Use provided training data or load it
    if training_df is None:
        validation_df, _ = load_validation_data(property_name)
    else:
        validation_df = training_df
    family_mapping = load_material_families()
    
    if not family_mapping:
        print("❌ Failed to load material families")
        return False
    
    # Build KI vector
    ki_vector, pair_mapping = build_ki_vector(validation_df, family_mapping, property_name)
    n_pairs = len(ki_vector)
    
    if n_pairs == 0:
        print("❌ No polymer pairs found in validation data")
        return False
    
    print(f"🎯 Optimizing {n_pairs} polymer pairs")
    
    # Calculate initial MAE and capture initial predictions
    initial_mae, initial_errors, initial_simulated = calculate_validation_mae(validation_df, property_name, family_mapping, ki_vector, pair_mapping)
    print(f"   Initial MAE: {initial_mae:.4f}")
    
    # Initialize optimization
    best_ki_vector = ki_vector.copy()
    best_mae = initial_mae
    prev_mae = initial_mae
    prev_ki_vector = ki_vector.copy()
    
    # Initialize adaptive learning rates for each KI parameter
    if use_adaptive_lr:
        n_blends = len(validation_df)
        print(f"   Using blend-specific learning rates (base: {learning_rate})")
        print(f"   Learning rate automatically adjusts based on MAE proximity to 0")
    else:
        print(f"   Using fixed learning rate: {learning_rate}")
    
    # Store results
    results = []
    
    print(f"\n{'='*80}")
    print(f"ITERATION |    MAE    |  IMPROVEMENT |  STATUS")
    print(f"{'='*80}")
    
    for iteration in range(max_iterations):
        # Calculate current MAE
        current_mae, errors, simulated_values = calculate_validation_mae(
            validation_df, property_name, family_mapping, ki_vector, pair_mapping
        )
        
        # Calculate improvement
        improvement = prev_mae - current_mae if iteration > 0 else 0
        
        # Update best if improved
        if current_mae < best_mae:
            best_mae = current_mae
            best_ki_vector = ki_vector.copy()
        
        # Status
        status = "✅ NEW BEST" if current_mae < best_mae else "🔄"
        
        # Print progress with adaptive learning rate info
        if use_adaptive_lr and iteration > 0 and 'blend_lrs' in locals():
            lr_info = f"LR: {np.mean(blend_lrs):.4f}±{np.std(blend_lrs):.4f}"
            print(f"   {iteration+1:8d} | {current_mae:8.4f} | {improvement:10.4f} | {status} | {lr_info}")
        else:
            print(f"   {iteration+1:8d} | {current_mae:8.4f} | {improvement:10.4f} | {status}")
        
        # Store results
        results.append({
            'iteration': iteration + 1,
            'mae': current_mae,
            'improvement': improvement,
            'ki_vector': ki_vector.copy(),
            'blend_lrs': blend_lrs if use_adaptive_lr and 'blend_lrs' in locals() else None
        })
        
        # Check convergence
        if iteration > 0 and abs(improvement) < 1e-6:
            print(f"   🎉 Converged after {iteration + 1} iterations (improvement: {improvement:.8f})")
            break
        
        # Calculate gradient using finite differences
        if iteration < max_iterations - 1:
            if use_adaptive_lr:
                # Use blend-specific gradients and learning rates
                gradient, blend_gradients, current_errors = calculate_blend_specific_gradients(
                    validation_df, property_name, family_mapping, ki_vector, pair_mapping
                )
                
                # Calculate blend-specific learning rates based on MAE proximity to 0
                blend_lrs = calculate_blend_learning_rates(current_errors, learning_rate)
                
                # Calculate weighted gradient for each KI parameter based on blend learning rates
                weighted_gradient = np.zeros(n_pairs)
                for i in range(n_pairs):
                    # Weight each blend's gradient by its learning rate
                    weighted_grad = 0.0
                    total_weight = 0.0
                    for j in range(n_blends):
                        if blend_gradients[i][j] != 0:  # Only consider blends that contribute
                            weight = blend_lrs[j]
                            weighted_grad += blend_gradients[i][j] * weight
                            total_weight += weight
                    
                    if total_weight > 0:
                        weighted_gradient[i] = weighted_grad / total_weight
                    else:
                        weighted_gradient[i] = gradient[i]  # Fallback to average gradient
                
                # Normalize the weighted gradient to prevent explosion
                gradient_norm = np.linalg.norm(weighted_gradient)
                if gradient_norm > 0:
                    # Scale down very large gradients but allow reasonable steps
                    if gradient_norm > gradient_threshold:
                        weighted_gradient = weighted_gradient / gradient_norm * max_step_size
                
                # Show gradient and learning rate info for debugging
                if iteration % 5 == 0 or iteration < 3:  # Show first few iterations and every 5
                    avg_lr = np.mean(blend_lrs)
                    min_lr = np.min(blend_lrs)
                    max_lr = np.max(blend_lrs)
                    grad_norm = np.linalg.norm(weighted_gradient)
                    print(f"      Gradient norm: {grad_norm:.8f}, Blend LR: avg={avg_lr:.4f}, min={min_lr:.4f}, max={max_lr:.4f}")
                    print(f"      KI changes: {np.linalg.norm(weighted_gradient):.8f}")
            else:
                # Use simple global gradient calculation
                gradient = np.zeros_like(ki_vector)
                epsilon = 1e-6
                
                for i in range(n_pairs):
                    ki_vector_plus = ki_vector.copy()
                    ki_vector_plus[i] += epsilon
                    
                    mae_plus, _, _ = calculate_validation_mae(
                        validation_df, property_name, family_mapping, ki_vector_plus, pair_mapping
                    )
                    
                    gradient[i] = (mae_plus - current_mae) / epsilon
            
            # No gradient normalization or clipping needed for blend-specific approach
            
            # Update KI vector with adaptive learning rates
            if use_adaptive_lr:
                ki_vector = ki_vector - weighted_gradient
            else:
                ki_vector = ki_vector - learning_rate * gradient
            
            # Light clipping to prevent extreme values while allowing large changes
            ki_vector = np.clip(ki_vector, -50, 50)
            
            # Store for next iteration
            prev_mae = current_mae
            prev_ki_vector = ki_vector.copy()
    
    # Final results
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Improvement: {initial_mae - best_mae:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'blend_optimization_results_{property_name}.csv', index=False)
    print(f"📊 Results saved to: blend_optimization_results_{property_name}.csv")
    
    # Create optimization plots
    create_optimization_plots(results_df, property_name)
    
    # Calculate final predictions for validation performance plots
    final_mae, final_errors, final_simulated = calculate_validation_mae(validation_df, property_name, family_mapping, best_ki_vector, pair_mapping)
    
    # Create validation performance plots
    create_validation_performance_plots(validation_df, property_name, initial_mae, final_mae, 
                                      initial_simulated, final_simulated)
    
    # Update compatibility file
    update_compatibility_file(best_ki_vector, pair_mapping, property_name)
    
    return True, best_ki_vector, pair_mapping


def update_compatibility_file(ki_vector: np.ndarray, pair_mapping: Dict[str, int], property_name: str):
    """Create a backup copy of the compatibility file with optimized KI values."""
    compatibility_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    backup_file = f"train/simulation/config/compatibility/{property_name}_compatibility_optimized.yaml"
    
    print(f"📝 Creating optimized copy: {backup_file}")
    
    # Read current file
    with open(compatibility_file, 'r') as f:
        content = f.read()
    
    # Update KI values (check both directions)
    for pair_name, ki_value in pair_mapping.items():
        # Handle multi-part family names like "Bio-PE" correctly
        if 'Bio-PE' in pair_name:
            if pair_name.startswith('Bio-PE-'):
                reverse_name = pair_name.replace('Bio-PE-', '') + '-Bio-PE'
            else:
                reverse_name = 'Bio-PE-' + pair_name.replace('-Bio-PE', '')
        else:
            reverse_name = '-'.join(reversed(pair_name.split('-')))
        
        # Try to update the pair in its current direction
        old_pattern = f'"{pair_name}": {{KI: [0-9.-]+'
        new_pattern = f'"{pair_name}": {{KI: {ki_vector[ki_value]:.3f}'
        updated_content = re.sub(old_pattern, new_pattern, content)
        
        # If no change was made, try the reverse direction
        if updated_content == content:
            old_pattern = f'"{reverse_name}": {{KI: [0-9.-]+'
            new_pattern = f'"{reverse_name}": {{KI: {ki_vector[ki_value]:.3f}'
            updated_content = re.sub(old_pattern, new_pattern, content)
        
        content = updated_content
    
    # Write to backup file (not overwriting original)
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created optimized copy: {backup_file}")
    print(f"   Original file preserved: {compatibility_file}")
    print(f"   To use optimized values, rename: mv {backup_file} {compatibility_file}")


def optimize_family_group_ki_values(property_name: str, max_iterations: int, learning_rate: float, seed: int, 
                                  use_adaptive_lr: bool = True, gradient_threshold: float = 2.0,
                                  max_step_size: float = 0.5, training_df: pd.DataFrame = None) -> Tuple[bool, np.ndarray, Dict[str, int]]:
    """Optimize KI values for validation pairs and copy to all family members."""
    print(f"🚀 Starting family group optimization for {property_name}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Random seed: {seed}")
    
    # Set random seed
    set_random_seeds(seed)
    
    # Use provided training data or load it
    if training_df is None:
        validation_df, _ = load_validation_data(property_name)
    else:
        validation_df = training_df
    family_mapping = load_material_families()
    
    if not family_mapping:
        print("❌ Failed to load material families")
        return False
    
    # Step 1: Run individual pair optimization to get optimized values
    print("Step 1: Optimizing validation pairs...")
    success, best_ki_vector, pair_mapping = optimize_blend_ki_values(property_name, max_iterations, learning_rate, seed, use_adaptive_lr, 
                                     gradient_threshold, max_step_size, training_df)
    
    if not success:
        print("❌ Validation pair optimization failed")
        return False, np.array([]), {}
    
    # Step 2: Load the optimized values and copy to family members
    print("\nStep 2: Copying optimized values to all family members...")
    
    # Load the optimized KI values from the results file
    results_df = pd.read_csv(f'blend_optimization_results_{property_name}.csv')
    best_mae = results_df['mae'].min()
    best_iteration = results_df[results_df['mae'] == best_mae]['iteration'].iloc[0]
    
    # Load the optimized KI values from the optimized YAML file
    # The file is named ts_compatibility_optimized.yaml, so we need to load it directly
    import yaml
    import re
    
    optimized_file = f"train/simulation/config/compatibility/{property_name}_compatibility_optimized.yaml"
    optimized_ki_values = {}
    
    try:
        with open(optimized_file, 'r') as f:
            content = f.read()
        
        # Extract KI values using regex
        pattern = r'"([^"]+)":\s*\{KI:\s*([0-9.-]+)'
        matches = re.findall(pattern, content)
        
        for pair_name, ki_value in matches:
            optimized_ki_values[pair_name] = float(ki_value)
        
        print(f"Loaded {len(optimized_ki_values)} optimized KI values from {optimized_file}")
        
    except Exception as e:
        print(f"Warning: Could not load optimized KI values: {e}")
        optimized_ki_values = {}
    
    # Build pair mapping from validation data
    ki_vector, pair_mapping = build_ki_vector(validation_df, family_mapping, property_name)
    
    # Get family groups
    family_groups = get_polymer_family_groups()
    
    # Create mapping from validation pairs to their optimized KI values
    validation_ki_values = {}
    for pair_name, ki_index in pair_mapping.items():
        # Get optimized KI value from the optimized YAML file
        optimized_ki = optimized_ki_values.get(pair_name, 0.0)
        # Try reverse name if not found
        if optimized_ki == 0.0:
            # Handle Bio-PE splitting correctly
            if 'Bio-PE' in pair_name:
                if pair_name.startswith('Bio-PE-'):
                    reverse_name = pair_name.replace('Bio-PE-', '') + '-Bio-PE'
                else:
                    reverse_name = 'Bio-PE-' + pair_name.replace('-Bio-PE', '')
            else:
                reverse_name = f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"
            optimized_ki = optimized_ki_values.get(reverse_name, 0.0)
        
        validation_ki_values[pair_name] = optimized_ki
        print(f"  {pair_name}: KI = {optimized_ki:.3f}")
    
    # Create all family member pairs and copy optimized values
    all_family_pairs = {}
    
    for validation_pair, optimized_ki in validation_ki_values.items():
        # Handle Bio-PE splitting correctly
        if 'Bio-PE' in validation_pair:
            if validation_pair.startswith('Bio-PE-'):
                family1 = 'Bio-PE'
                family2 = validation_pair.replace('Bio-PE-', '')
            else:
                family1 = validation_pair.replace('-Bio-PE', '')
                family2 = 'Bio-PE'
        else:
            family1, family2 = validation_pair.split('-')
        
        # Get family groups
        group1 = get_family_group_for_polymer(family1, family_groups)
        group2 = get_family_group_for_polymer(family2, family_groups)
        
        # Get all families in each group
        families1 = family_groups.get(group1, [group1])
        families2 = family_groups.get(group2, [group2])
        
        # Copy optimized KI value to all family member pairs
        for f1 in families1:
            for f2 in families2:
                pair_name = f"{f1}-{f2}"
                reverse_name = f"{f2}-{f1}"
                all_family_pairs[pair_name] = optimized_ki
                all_family_pairs[reverse_name] = optimized_ki
    
    print(f"Copied optimized values to {len(all_family_pairs)} family member pairs")
    
    # Update compatibility file with all family member values
    update_family_group_compatibility_file(all_family_pairs, property_name)
    
    return True, best_ki_vector, pair_mapping


def update_family_group_compatibility_file(family_pairs: Dict[str, float], property_name: str):
    """Update compatibility file with family group values."""
    compatibility_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    backup_file = f"train/simulation/config/compatibility/{property_name}_compatibility_family_optimized.yaml"
    
    print(f"📝 Creating family group optimized copy: {backup_file}")
    
    # Read current file
    with open(compatibility_file, 'r') as f:
        content = f.read()
    
    # Update KI values for all family member pairs
    for pair_name, ki_value in family_pairs.items():
        # Try both directions
        old_pattern = f'"{pair_name}": {{KI: [0-9.-]+'
        new_pattern = f'"{pair_name}": {{KI: {ki_value:.3f}'
        updated_content = re.sub(old_pattern, new_pattern, content)
        
        # If no change was made, try the reverse direction
        if updated_content == content:
            reverse_name = '-'.join(reversed(pair_name.split('-')))
            old_pattern = f'"{reverse_name}": {{KI: [0-9.-]+'
            new_pattern = f'"{reverse_name}": {{KI: {ki_value:.3f}'
            updated_content = re.sub(old_pattern, new_pattern, content)
        
        content = updated_content
    
    # Write to backup file
    with open(backup_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Created family group optimized copy: {backup_file}")
    print(f"   Original file preserved: {compatibility_file}")
    print(f"   To use optimized values, rename: mv {backup_file} {compatibility_file}")


def create_optimization_plots(results_df: pd.DataFrame, property_name: str, output_dir: str = "."):
    """Create comprehensive optimization visualization plots."""
    print("📈 Creating optimization visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': True,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Blend Optimization Results - {property_name.upper()}', fontsize=16, fontweight='bold')
    
    # 1. MAE vs Iterations
    axes[0, 0].plot(results_df['iteration'], results_df['mae'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add improvement annotation
    initial_mae = results_df['mae'].iloc[0]
    final_mae = results_df['mae'].iloc[-1]
    improvement = initial_mae - final_mae
    axes[0, 0].annotate(f'Improvement: {improvement:.3f}', 
                       xy=(len(results_df), final_mae), 
                       xytext=(len(results_df)*0.7, initial_mae*0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')
    
    # 2. Improvement per Iteration
    axes[0, 1].plot(results_df['iteration'][1:], results_df['improvement'][1:], 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('MAE Improvement')
    axes[0, 1].set_title('Improvement per Iteration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate Distribution (if available)
    if 'blend_lrs' in results_df.columns:
        # Extract learning rates from the last iteration
        last_lrs = results_df['blend_lrs'].iloc[-1]
        if isinstance(last_lrs, str):
            # Parse the string representation of the array
            lr_values = eval(last_lrs)
        else:
            lr_values = last_lrs
        
        # Ensure lr_values is not None and is iterable
        if lr_values is None or len(lr_values) == 0:
            lr_values = [0.001]  # Default fallback
        
        axes[0, 2].hist(lr_values, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Learning Rate')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Learning Rate Distribution (Final)')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        # Fallback: show iteration count
        axes[0, 2].bar(['Total Iterations'], [len(results_df)], alpha=0.7, color='blue')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Optimization Summary')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. MAE Distribution
    axes[1, 0].hist(results_df['mae'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(initial_mae, color='red', linestyle='--', linewidth=2, label=f'Initial: {initial_mae:.3f}')
    axes[1, 0].axvline(final_mae, color='green', linestyle='--', linewidth=2, label=f'Final: {final_mae:.3f}')
    axes[1, 0].set_xlabel('MAE')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MAE Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Improvement Distribution
    improvements = results_df['improvement'][results_df['improvement'] != 0]
    if len(improvements) > 0:
        axes[1, 1].hist(improvements, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('MAE Improvement')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Improvement Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No improvements\nrecorded', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Improvement Distribution')
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
OPTIMIZATION SUMMARY

Property: {property_name.upper()}
Total Iterations: {len(results_df)}
Initial MAE: {initial_mae:.4f}
Final MAE: {final_mae:.4f}
Total Improvement: {improvement:.4f}
Improvement %: {(improvement/initial_mae)*100:.2f}%

Convergence:
• Total Iterations: {len(results_df)}
• Improved Iterations: {len(results_df[results_df['improvement'] > 0])}
• No Change Iterations: {len(results_df[results_df['improvement'] == 0])}
• Worsened Iterations: {len(results_df[results_df['improvement'] < 0])}

Best Performance:
• Best MAE: {results_df['mae'].min():.4f}
• Best Iteration: {results_df.loc[results_df['mae'].idxmin(), 'iteration']}
    """
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'blend_optimization_results_{property_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Optimization plots saved to: {plot_path}")


def create_validation_performance_plots(validation_df: pd.DataFrame, property_name: str, 
                                      initial_mae: float, final_mae: float, 
                                      initial_predictions: List[float], final_predictions: List[float],
                                      output_dir: str = ".", plot_suffix: str = ""):
    """Create validation blend performance plots EXACTLY matching last_11_blends_performance.png format."""
    print("📊 Creating validation performance plots...")
    
    # Set up the dark theme EXACTLY like XGBoost training plots
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'font.size': 12,
        'font.weight': 'normal',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': 'black',
        'legend.framealpha': 0.8
    })
    
    # Property abbreviation mapping with units (EXACTLY like XGBoost)
    property_abbreviations = {
        'tensile': 'TS (MPa)',
        'tensile strength': 'TS (MPa)',
        'wvtr': 'WVTR (g/m²/day)',
        'water vapor transmission rate': 'WVTR (g/m²/day)',
        'eab': 'EaB (%)',
        'elongation at break': 'EaB (%)',
        'cobb': 'Cobb (g/m²)',
        'cobb angle': 'Cobb (g/m²)',
        'seal': 'Max Seal Strength (N/15mm)',
        'sealing': 'Max Seal Strength (N/15mm)',
        'adhesion': 'Max Seal Strength (N/15mm)',
        'compost': 'Compost (%)',
        'compostability': 'Compost (%)',
        'otr': 'OTR (cc/m²/day)',
        'oxygen transmission rate': 'OTR (cc/m²/day)'
    }
    
    # Get property abbreviation
    prop_abbrev = property_abbreviations.get(property_name.lower(), property_name.upper())
    
    # Get actual values (assuming they're in the first property column)
    property_cols = [col for col in validation_df.columns if col.startswith('property')]
    if property_cols:
        actual_values = validation_df[property_cols[0]].values
    else:
        actual_values = np.zeros(len(validation_df))
    
    # Check if this is a dual property (TS or EAB)
    is_dual_property = len(property_cols) == 2 and property_name.lower() in ['ts', 'eab']
    
    # Create plot with EXACT same layout as XGBoost
    if is_dual_property:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    
    if not is_dual_property:
        axes = axes.reshape(2, 1)
    
    # Get blend labels from Materials column
    if 'Materials' in validation_df.columns:
        blend_labels = validation_df['Materials'].astype(str).values
    else:
        blend_labels = [f'Blend {i+1}' for i in range(len(validation_df))]
    
    # Calculate metrics
    initial_r2 = 1 - (np.sum((actual_values - initial_predictions)**2) / np.sum((actual_values - np.mean(actual_values))**2))
    final_r2 = 1 - (np.sum((actual_values - final_predictions)**2) / np.sum((actual_values - np.mean(actual_values))**2))
    
    # Plot for each property (single or dual)
    properties_to_plot = property_cols if is_dual_property else [property_cols[0]]
    
    for i, prop_col in enumerate(properties_to_plot):
        if is_dual_property:
            actual_vals = validation_df[prop_col].values
            initial_preds = initial_predictions
            final_preds = final_predictions
            
            # For dual properties, use specific labels
            if property_name.lower() == 'ts':
                current_prop_label = 'TS-MD (MPa)' if i == 0 else 'TS-TD (MPa)'
            elif property_name.lower() == 'eab':
                current_prop_label = 'EaB-MD (%)' if i == 0 else 'EaB-TD (%)'
            else:
                current_prop_label = prop_abbrev
        else:
            actual_vals = actual_values
            initial_preds = initial_predictions
            final_preds = final_predictions
            current_prop_label = prop_abbrev
        
        # Plot 1: Before Optimization (Log Scale)
        axes[0, i].scatter(actual_vals, initial_preds, color='#FF6B6B', s=100, alpha=0.7)
        axes[0, i].plot([actual_vals.min(), actual_vals.max()], 
                       [actual_vals.min(), actual_vals.max()], 'w--', lw=3, alpha=0.8, label='y=x')
        axes[0, i].set_xlabel(f'Actual {current_prop_label}', fontweight='bold')
        axes[0, i].set_ylabel(f'Predicted {current_prop_label}', fontweight='bold')
        axes[0, i].set_title(f'{current_prop_label} - Before Optimization', fontweight='bold')
        axes[0, i].legend(loc='best', framealpha=0.8)
        axes[0, i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add blend labels
        for j, (actual, pred) in enumerate(zip(actual_vals, initial_preds)):
            blend_label = str(blend_labels[j])
            axes[0, i].annotate(blend_label, (actual, pred), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='#DDA0DD')
        
        # Add metrics text box
        metrics_text = f'MAE: {initial_mae:.3f}\nR²: {initial_r2:.3f}'
        axes[0, i].text(0.05, 0.95, metrics_text, transform=axes[0, i].transAxes,
                        fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: After Optimization (Log Scale)
        axes[1, i].scatter(actual_vals, final_preds, color='#4ECDC4', s=100, alpha=0.7)
        axes[1, i].plot([actual_vals.min(), actual_vals.max()], 
                       [actual_vals.min(), actual_vals.max()], 'w--', lw=3, alpha=0.8, label='y=x')
        axes[1, i].set_xlabel(f'Actual {current_prop_label}', fontweight='bold')
        axes[1, i].set_ylabel(f'Predicted {current_prop_label}', fontweight='bold')
        axes[1, i].set_title(f'{current_prop_label} - After Optimization', fontweight='bold')
        axes[1, i].legend(loc='best', framealpha=0.8)
        axes[1, i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add blend labels
        for j, (actual, pred) in enumerate(zip(actual_vals, final_preds)):
            blend_label = str(blend_labels[j])
            axes[1, i].annotate(blend_label, (actual, pred), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='#DDA0DD')
        
        # Add metrics text box
        metrics_text = f'MAE: {final_mae:.3f}\nR²: {final_r2:.3f}'
        axes[1, i].text(0.05, 0.95, metrics_text, transform=axes[1, i].transAxes,
                        fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot with EXACT same naming convention as XGBoost
    plot_path = os.path.join(output_dir, f'last_{len(validation_df)}_blends_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    
    print(f"✅ Validation performance plots saved to: {plot_path}")


def create_testing_performance_plot(testing_df: pd.DataFrame, property_name: str, 
                                  final_mae: float, actual_values: List[float], 
                                  final_predictions: List[float], output_dir: str = ".", 
                                  plot_suffix: str = ""):
    """Create testing performance plot showing only final predictions vs actual values."""
    print("📊 Creating testing performance plot...")
    
    # Ensure predictions and actual values have the same length
    if len(final_predictions) != len(actual_values):
        print(f"⚠️  Warning: Mismatch in prediction lengths - actual: {len(actual_values)}, final: {len(final_predictions)}")
        min_len = min(len(final_predictions), len(actual_values))
        final_predictions = final_predictions[:min_len]
        actual_values = actual_values[:min_len]
    
    if len(final_predictions) == 0:
        print("⚠️  No valid predictions available for plotting")
        return
    
    # Set up the dark theme EXACTLY like XGBoost training plots
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'font.size': 12,
        'font.weight': 'normal',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': 'black',
        'legend.framealpha': 0.8
    })
    
    # Property abbreviation mapping with units
    property_abbreviations = {
        'tensile': 'TS (MPa)',
        'tensile strength': 'TS (MPa)',
        'wvtr': 'WVTR (g/m²/day)',
        'water vapor transmission rate': 'WVTR (g/m²/day)',
        'eab': 'EaB (%)',
        'elongation at break': 'EaB (%)',
        'cobb': 'Cobb (g/m²)',
        'cobb angle': 'Cobb (g/m²)',
        'seal': 'Max Seal Strength (N/15mm)',
        'sealing': 'Max Seal Strength (N/15mm)',
        'adhesion': 'Max Seal Strength (N/15mm)',
        'compost': 'Compost (%)',
        'compostability': 'Compost (%)',
        'otr': 'OTR (cc/m²/day)',
        'oxygen transmission rate': 'OTR (cc/m²/day)'
    }
    
    # Get property abbreviation
    prop_abbrev = property_abbreviations.get(property_name.lower(), property_name.upper())
    
    # Check if this is a dual property (TS or EAB)
    property_cols = [col for col in testing_df.columns if col.startswith('property')]
    is_dual_property = len(property_cols) == 2 and property_name.lower() in ['ts', 'eab']
    
    # Create plot
    if is_dual_property:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]
    
    # Get blend labels from Materials column (filtered to match predictions length)
    if 'Materials' in testing_df.columns:
        blend_labels = testing_df['Materials'].astype(str).values[:len(final_predictions)]
    else:
        blend_labels = [f'Blend {i+1}' for i in range(len(final_predictions))]
    
    # Calculate R2
    actual_array = np.array(actual_values)
    pred_array = np.array(final_predictions)
    r2 = 1 - (np.sum((actual_array - pred_array)**2) / np.sum((actual_array - np.mean(actual_array))**2))
    
    # Plot for each property (single or dual)
    properties_to_plot = property_cols if is_dual_property else [property_cols[0]]
    
    for i, prop_col in enumerate(properties_to_plot):
        if is_dual_property:
            actual_vals = testing_df[prop_col].values[:len(final_predictions)]
            
            # For dual properties, use specific labels
            if property_name.lower() == 'ts':
                current_prop_label = 'TS-MD (MPa)' if i == 0 else 'TS-TD (MPa)'
            elif property_name.lower() == 'eab':
                current_prop_label = 'EaB-MD (%)' if i == 0 else 'EaB-TD (%)'
            else:
                current_prop_label = prop_abbrev
        else:
            actual_vals = actual_values
            current_prop_label = prop_abbrev
        
        # Plot: Final Predictions vs Actual (Log Scale)
        axes[i].scatter(actual_vals, final_predictions, color='#6BFF6B', s=100, alpha=0.7)
        axes[i].plot([actual_vals.min(), actual_vals.max()], 
                    [actual_vals.min(), actual_vals.max()], 'w--', lw=3, alpha=0.8, label='y=x')
        axes[i].set_xlabel(f'Actual {current_prop_label}', fontweight='bold')
        axes[i].set_ylabel(f'Predicted {current_prop_label}', fontweight='bold')
        axes[i].set_title(f'{current_prop_label} - Testing Set (Optimized Predictions)', fontweight='bold')
        axes[i].legend(loc='best', framealpha=0.8)
        axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add blend labels
        for j, (actual, pred) in enumerate(zip(actual_vals, final_predictions)):
            blend_label = str(blend_labels[j])
            axes[i].annotate(blend_label, (actual, pred), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color='white')
        
        # Add MAE and R2 metrics
        metrics_text = f'MAE: {final_mae:.3f}\nR²: {r2:.3f}'
        axes[i].text(0.05, 0.95, metrics_text, transform=axes[i].transAxes,
                    fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plot_filename = f'last_{len(testing_df)}_blends_performance{plot_suffix}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
    plt.close()
    
    print(f"✅ Testing performance plot saved to: {plot_path}")


def evaluate_performance_on_dataset(validation_df: pd.DataFrame, property_name: str, family_mapping: Dict[str, str],
                                  ki_vector: np.ndarray, pair_mapping: Dict[str, int], 
                                  use_optimized_ki: bool = True, use_family_groups: bool = False) -> Tuple[float, List[float], List[float]]:
    """Evaluate performance on a specific dataset (training or testing)."""
    if len(validation_df) == 0:
        return 0.0, [], []
    
    errors = []
    predictions = []
    actual_values = []
    
    # Load full compatibility file for fallback values
    full_compatibility = load_existing_ki_values(property_name)
    
    for _, blend_row in validation_df.iterrows():
        # Get polymer families for this blend
        families_in_blend = extract_polymer_families_from_blend(blend_row, family_mapping)
        
        # Try to get KI overrides from optimized values first
        ki_overrides = {}
        missing_pairs = []
        
        # Generate all possible pairs from families in blend
        for i, family1 in enumerate(families_in_blend):
            for j, family2 in enumerate(families_in_blend):
                if i < j:  # Avoid duplicates and self-pairs
                    pair_name = f"{family1}-{family2}"
                    reverse_pair = f"{family2}-{family1}"
                    
                    # Try to get from optimized KI vector first
                    if pair_name in pair_mapping:
                        ki_overrides[pair_name] = ki_vector[pair_mapping[pair_name]]
                    elif reverse_pair in pair_mapping:
                        ki_overrides[reverse_pair] = ki_vector[pair_mapping[reverse_pair]]
                    else:
                        # Not in optimized vector, try family group rules or fallback
                        missing_pairs.append(pair_name)
        
        # For missing pairs, handle based on whether family groups are enabled
        if missing_pairs and use_optimized_ki:
            if use_family_groups:
                # Try to apply family group rules for missing pairs
                family_groups = get_polymer_family_groups()
                for pair_name in missing_pairs:
                    # Handle pair names that might have multiple dashes
                    pair_parts = pair_name.split('-')
                    if len(pair_parts) >= 2:
                        family1 = pair_parts[0]
                        family2 = '-'.join(pair_parts[1:])  # Join remaining parts
                        group1 = get_family_group_for_polymer(family1, family_groups)
                        group2 = get_family_group_for_polymer(family2, family_groups)
                        
                        # Try to find a representative pair from the same family groups
                        found_ki = False
                        for optimized_pair in pair_mapping.keys():
                            ki_value = ki_vector[pair_mapping[optimized_pair]]
                            # Handle pair names that might have multiple dashes
                            opt_pair_parts = optimized_pair.split('-')
                            if len(opt_pair_parts) >= 2:
                                opt_family1 = opt_pair_parts[0]
                                opt_family2 = '-'.join(opt_pair_parts[1:])  # Join remaining parts
                                opt_group1 = get_family_group_for_polymer(opt_family1, family_groups)
                                opt_group2 = get_family_group_for_polymer(opt_family2, family_groups)
                                
                                # Check if groups match (order independent) OR if one family matches
                                if ((opt_group1 == group1 and opt_group2 == group2) or 
                                    (opt_group1 == group2 and opt_group2 == group1) or
                                    (opt_family1 == family1 or opt_family2 == family2) or
                                    (opt_family1 == family2 or opt_family2 == family1)):
                                    ki_overrides[pair_name] = ki_value
                                    found_ki = True
                                    break
                        
                        # If still not found, use original compatibility file value
                        if not found_ki:
                            if pair_name in full_compatibility:
                                ki_overrides[pair_name] = full_compatibility[pair_name]
                            elif f"{family2}-{family1}" in full_compatibility:
                                ki_overrides[pair_name] = full_compatibility[f"{family2}-{family1}"]
                            else:
                                # Use zero as fallback
                                ki_overrides[pair_name] = 0.0
            else:
                # Family groups NOT enabled - use original values or 0.0 for missing pairs
                for pair_name in missing_pairs:
                    if pair_name in full_compatibility:
                        ki_overrides[pair_name] = full_compatibility[pair_name]
                    elif f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}" in full_compatibility:
                        ki_overrides[pair_name] = full_compatibility[f"{pair_name.split('-')[1]}-{pair_name.split('-')[0]}"]
                    else:
                        # Use zero as fallback for pairs not in training set
                        ki_overrides[pair_name] = 0.0
        
        # Simulate the blend
        predicted_value = simulate_validation_blend(blend_row, property_name, family_mapping, ki_overrides)
        
        if predicted_value is not None:
            # Get actual value based on property type
            if property_name in ['ts', 'eab']:
                actual_value = blend_row['property1']  # Use property1 for dual properties
            else:
                actual_value = blend_row['property']
            
            error = abs(predicted_value - actual_value)
            errors.append(error)
            predictions.append(predicted_value)
            actual_values.append(actual_value)
    
    if errors:
        mae = np.mean(errors)
        return mae, predictions, actual_values
    else:
        return 0.0, [], []


def save_detailed_results_csv(validation_df: pd.DataFrame, property_name: str, 
                            actual_values: List[float], predicted_values: List[float], 
                            dataset_type: str, output_prefix: str = ""):
    """Save detailed results to CSV with predicted, actual, error, and accuracy."""
    if len(actual_values) == 0 or len(predicted_values) == 0:
        print(f"⚠️  No data to save for {dataset_type} set")
        return
    
    # Calculate errors and accuracy
    actual_array = np.array(actual_values)
    predicted_array = np.array(predicted_values)
    errors = np.abs(actual_array - predicted_array)
    
    # Calculate accuracy as (1 - relative_error) * 100
    # Use relative error to avoid division by zero
    relative_errors = np.where(actual_array != 0, errors / actual_array, errors)
    accuracy = (1 - relative_errors) * 100
    
    # Create results DataFrame
    results_data = {
        'Blend_Name': validation_df['Materials'].values[:len(actual_values)] if 'Materials' in validation_df.columns else [f'Blend_{i+1}' for i in range(len(actual_values))],
        'Actual': actual_values,
        'Predicted': predicted_values,
        'Error': errors,
        'Relative_Error': relative_errors,
        'Accuracy_%': accuracy
    }
    
    # Add property-specific columns if it's a dual property
    if property_name.lower() in ['ts', 'eab'] and len(validation_df.columns) > 1:
        property_cols = [col for col in validation_df.columns if col.startswith('property')]
        if len(property_cols) >= 2:
            results_data['Actual_Property2'] = validation_df[property_cols[1]].values[:len(actual_values)]
    
    results_df = pd.DataFrame(results_data)
    
    # Save to CSV
    filename = f"{output_prefix}_{dataset_type}_detailed_results.csv"
    results_df.to_csv(filename, index=False)
    
    # Print summary statistics
    mae = np.mean(errors)
    mean_accuracy = np.mean(accuracy)
    print(f"📊 {dataset_type.capitalize()} set detailed results saved to: {filename}")
    print(f"   MAE: {mae:.4f}")
    print(f"   Mean Accuracy: {mean_accuracy:.2f}%")
    print(f"   Accuracy Range: {np.min(accuracy):.2f}% - {np.max(accuracy):.2f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Optimize KI values for polymer-polymer compatibility')
    parser.add_argument('--property', required=True, 
                       choices=['wvtr', 'otr', 'ts', 'eab', 'cobb', 'seal', 'compost'],
                       help='Property to optimize')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum number of iterations (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for gradient descent (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--adaptive-lr', action='store_true', default=True,
                       help='Use adaptive learning rates (default: True)')
    parser.add_argument('--no-adaptive-lr', action='store_true',
                       help='Disable adaptive learning rates')
    parser.add_argument('--family-groups', action='store_true', default=False,
                       help='Use family group optimization instead of individual pair optimization')
    
    # Adaptive learning rate control parameters
    # Blend-specific learning rate parameters (automatically calculated based on MAE)
    parser.add_argument('--base-lr', type=float, default=0.1,
                        help='Base learning rate for blend-specific adaptive learning (default: 0.1)')
    
    # Gradient normalization control parameters
    parser.add_argument('--gradient-threshold', type=float, default=2.0,
                        help='Gradient norm threshold for normalization (default: 2.0)')
    parser.add_argument('--max-step-size', type=float, default=0.5,
                        help='Maximum step size after gradient normalization (default: 0.5)')
    
    # Data splitting parameters
    parser.add_argument('--last-n-testing', type=int, default=0,
                        help='Number of last N blends to use for testing (default: 0 - use all for training)')
    
    args = parser.parse_args()
    
    # Handle adaptive learning rate logic
    use_adaptive_lr = args.adaptive_lr and not args.no_adaptive_lr
    
    # Load and split validation data
    training_df, testing_df = load_validation_data(args.property, args.last_n_testing)
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Load family mapping
    family_mapping = load_material_families()
    
    print(f"Optimizing KI values using {len(training_df)} training blends...")
    
    if args.family_groups:
        success, optimized_ki_vector, optimized_pair_mapping = optimize_family_group_ki_values(
            property_name=args.property,
            max_iterations=args.max_iterations,
            learning_rate=args.base_lr if use_adaptive_lr else args.learning_rate,
            seed=args.seed,
            use_adaptive_lr=use_adaptive_lr,
            gradient_threshold=args.gradient_threshold,
            max_step_size=args.max_step_size,
            training_df=training_df
        )
    else:
        success, optimized_ki_vector, optimized_pair_mapping = optimize_blend_ki_values(
            property_name=args.property,
            max_iterations=args.max_iterations,
            learning_rate=args.base_lr if use_adaptive_lr else args.learning_rate,
            seed=args.seed,
            use_adaptive_lr=use_adaptive_lr,
            gradient_threshold=args.gradient_threshold,
            max_step_size=args.max_step_size,
            training_df=training_df
        )
    
    if success:
        print("🎉 Optimization completed successfully!")
        
        # Evaluate performance on both training and testing sets
        print("\n📊 Evaluating performance...")
        
        # Training set performance
        training_mae, training_predictions, training_actual = evaluate_performance_on_dataset(
            training_df, args.property, family_mapping, optimized_ki_vector, optimized_pair_mapping, 
            use_optimized_ki=True, use_family_groups=args.family_groups
        )
        print(f"Training set MAE: {training_mae:.4f}")
        
        # Calculate and display training set accuracy
        training_accuracy = np.mean([(1 - abs(actual - pred) / actual) * 100 for actual, pred in zip(training_actual, training_predictions) if actual != 0])
        print(f"Training set Average Accuracy: {training_accuracy:.2f}%")
        
        # Create training set performance plot
        if len(training_df) > 0:
            create_validation_performance_plots(
                training_df, args.property, 0.0, training_mae, 
                training_actual, training_predictions, ".", "_training"
            )
            
            # Save training set detailed results to CSV
            save_detailed_results_csv(
                training_df, args.property, training_actual, training_predictions, 
                "training", args.property
            )
        
        # Testing set performance
        if len(testing_df) > 0:
            testing_mae, testing_predictions, testing_actual = evaluate_performance_on_dataset(
                testing_df, args.property, family_mapping, optimized_ki_vector, optimized_pair_mapping, 
                use_optimized_ki=True, use_family_groups=args.family_groups
            )
            print(f"Testing set MAE: {testing_mae:.4f}")
            
            # Calculate and display testing set accuracy
            testing_accuracy = np.mean([(1 - abs(actual - pred) / actual) * 100 for actual, pred in zip(testing_actual, testing_predictions) if actual != 0])
            print(f"Testing set Average Accuracy: {testing_accuracy:.2f}%")
            
            # Create testing set performance plot (only final predictions, no before/after)
            create_testing_performance_plot(
                testing_df, args.property, testing_mae, 
                testing_actual, testing_predictions, ".", "_testing"
            )
            
            # Save testing set detailed results to CSV
            save_detailed_results_csv(
                testing_df, args.property, testing_actual, testing_predictions, 
                "testing", args.property
            )
        else:
            print("No testing set (using all blends for training)")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"🎯 OPTIMIZATION SUMMARY")
        print(f"{'='*80}")
        print(f"Training set MAE: {training_mae:.4f}")
        print(f"Training set Average Accuracy: {training_accuracy:.2f}%")
        if len(testing_df) > 0:
            print(f"Testing set MAE: {testing_mae:.4f}")
            print(f"Testing set Average Accuracy: {testing_accuracy:.2f}%")
        print(f"📊 Detailed results saved to CSV files")
        print(f"{'='*80}")
            
    else:
        print("❌ Optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
