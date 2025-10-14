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
from typing import Dict, List, Any, Optional, Tuple

# Add the train/simulation directory to the path
sys.path.append('train/simulation')

warnings.filterwarnings('ignore')

from simulation_common import set_random_seeds
from simulation_rules import PROPERTY_CONFIGS
from umm3_correction import UMM3Correction, load_family_compatibility_config


def load_validation_data(property_name: str) -> pd.DataFrame:
    """Load validation blends for a specific property."""
    validation_path = f"train/data/{property_name}/validationblends.csv"
    
    if not os.path.exists(validation_path):
        raise FileNotFoundError(f"Validation data not found: {validation_path}")
    
    print(f"Loading validation data from: {validation_path}")
    validation_df = pd.read_csv(validation_path)
    print(f"Loaded {len(validation_df)} validation blends")
    
    return validation_df


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
    for i in range(1, 6):  # Polymer Grade 1-5
        grade_col = f'Polymer Grade {i}'
        if pd.notna(blend_row[grade_col]):
            grade = blend_row[grade_col]
            if grade in family_mapping:
                family = family_mapping[grade]
                if family not in families:
                    families.append(family)
    return families


def build_ki_vector(validation_df: pd.DataFrame, family_mapping: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build KI vector and mapping for polymer pairs found in validation data."""
    print("Building KI vector from validation data...")
    
    # Collect all unique polymer families from validation blends
    all_families = set()
    for _, row in validation_df.iterrows():
        families = extract_polymer_families_from_blend(row, family_mapping)
        all_families.update(families)
    
    all_families = sorted(list(all_families))
    print(f"Found {len(all_families)} unique polymer families: {all_families}")
    
    # Find all polymer-polymer pairs that exist in validation blends
    existing_pairs = set()
    for _, row in validation_df.iterrows():
        families = extract_polymer_families_from_blend(row, family_mapping)
        # Create all possible pairs from families in this blend
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                pair = tuple(sorted([families[i], families[j]]))
                existing_pairs.add(pair)
    
    existing_pairs = sorted(list(existing_pairs))
    print(f"Found {len(existing_pairs)} unique polymer pairs in validation data")
    
    # Create KI vector and mapping
    ki_vector = np.zeros(len(existing_pairs))
    pair_mapping = {}
    
    for i, (family1, family2) in enumerate(existing_pairs):
        pair_name = f"{family1}-{family2}"
        pair_mapping[pair_name] = i
        print(f"  {i:2d}: {pair_name}")
    
    return ki_vector, pair_mapping


def get_ki_overrides_from_vector(ki_vector: np.ndarray, pair_mapping: Dict[str, int], 
                                families_in_blend: List[str]) -> Dict[str, float]:
    """Get KI overrides for a specific blend from the KI vector."""
    ki_overrides = {}
    
    # Create all possible pairs from families in this blend
    for i in range(len(families_in_blend)):
        for j in range(i + 1, len(families_in_blend)):
            family1, family2 = sorted([families_in_blend[i], families_in_blend[j]])
            pair_name = f"{family1}-{family2}"
            
            if pair_name in pair_mapping:
                ki_value = ki_vector[pair_mapping[pair_name]]
                ki_overrides[pair_name] = ki_value
    
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
        
        # Create environmental config
        if property_name in ['wvtr', 'otr']:
            env_config = {
                property_name: {
                    'temperature': {
                        'min': 38, 'max': 38, 'reference': 36, 'power_law': 0.44, 'scaling_type': 'power_law'
                    },
                    'humidity': {
                        'min': 90, 'max': 90, 'reference': 85.0, 'power_law': 0.3, 'scaling_type': 'power_law'
                    },
                    'thickness': {
                        'min': 100, 'max': 100, 'power_law': 0.5, 'reference': 50.0, 'scaling_type': 'fixed'
                    }
                }
            }
        elif property_name == 'otr':
            env_config = {
                property_name: {
                    'temperature': {
                        'min': 38, 'max': 38, 'reference': 23.0, 'max_scale': 5.0, 'divisor': 10.0, 'scaling_type': 'logarithmic'
                    },
                    'humidity': {
                        'min': 90, 'max': 90, 'reference': 50.0, 'max_scale': 3.0, 'divisor': 20.0, 'scaling_type': 'logarithmic'
                    },
                    'thickness': {
                        'min': 100, 'max': 100, 'power_law': 0.1, 'reference': 25.0, 'scaling_type': 'dynamic'
                    }
                }
            }
        else:
            # For TS, EAB, etc. - only thickness
            env_config = {
                property_name: {
                    'thickness': {
                        'min': 100, 'max': 100, 'power_law': 0.5, 'reference': 25.0, 'scaling_type': 'fixed'
                    }
                }
            }
        
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
        
        # Apply UMM3 correction if KI overrides are provided
        if ki_overrides:
            umm3 = UMM3Correction.from_config_files()
            umm3.ki_overrides = ki_overrides
            
            family_config = load_family_compatibility_config(
                config_dir="train/simulation/config/compatibility", 
                property_name=property_name
            )
            
            # Get property values
            property_values = {}
            for key, value in blend_row_data.items():
                if key == 'property' and isinstance(value, (int, float)):
                    property_values[property_name] = value
            
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
                           use_adaptive_lr: bool = True) -> bool:
    """Optimize KI values for polymer-polymer compatibility."""
    print(f"🚀 Starting blend optimization for {property_name}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Random seed: {seed}")
    
    # Set random seed
    set_random_seeds(seed)
    
    # Load data
    validation_df = load_validation_data(property_name)
    family_mapping = load_material_families()
    
    if not family_mapping:
        print("❌ Failed to load material families")
        return False
    
    # Build KI vector
    ki_vector, pair_mapping = build_ki_vector(validation_df, family_mapping)
    n_pairs = len(ki_vector)
    
    if n_pairs == 0:
        print("❌ No polymer pairs found in validation data")
        return False
    
    print(f"🎯 Optimizing {n_pairs} polymer pairs")
    
    # Calculate initial MAE
    initial_mae, _, _ = calculate_validation_mae(validation_df, property_name, family_mapping, ki_vector, pair_mapping)
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
            print(f"   🎉 Converged after {iteration + 1} iterations")
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
                    # Scale down very large gradients but don't completely normalize
                    if gradient_norm > 1.0:
                        weighted_gradient = weighted_gradient / gradient_norm * 0.1
                
                # Show blend learning rate distribution for debugging
                if iteration % 10 == 0:  # Show every 10 iterations
                    avg_lr = np.mean(blend_lrs)
                    min_lr = np.min(blend_lrs)
                    max_lr = np.max(blend_lrs)
                    print(f"      Blend LR: avg={avg_lr:.4f}, min={min_lr:.4f}, max={max_lr:.4f}")
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
    
    # Update compatibility file
    update_compatibility_file(best_ki_vector, pair_mapping, property_name)
    
    return True


def update_compatibility_file(ki_vector: np.ndarray, pair_mapping: Dict[str, int], property_name: str):
    """Update the compatibility file with optimized KI values."""
    compatibility_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    
    print(f"📝 Updating {compatibility_file} with optimized KI values...")
    
    # Read current file
    with open(compatibility_file, 'r') as f:
        content = f.read()
    
    # Update KI values
    for pair_name, ki_value in pair_mapping.items():
        old_pattern = f'"{pair_name}": {{KI: [0-9.-]+'
        new_pattern = f'"{pair_name}": {{KI: {ki_vector[ki_value]:.3f}'
        content = re.sub(old_pattern, new_pattern, content)
    
    # Write updated file
    with open(compatibility_file, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {compatibility_file}")


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
    
    # Adaptive learning rate control parameters
    # Blend-specific learning rate parameters (automatically calculated based on MAE)
    parser.add_argument('--base-lr', type=float, default=0.1,
                        help='Base learning rate for blend-specific adaptive learning (default: 0.1)')
    
    args = parser.parse_args()
    
    # Handle adaptive learning rate logic
    use_adaptive_lr = args.adaptive_lr and not args.no_adaptive_lr
    
    success = optimize_blend_ki_values(
        property_name=args.property,
        max_iterations=args.max_iterations,
        learning_rate=args.base_lr if use_adaptive_lr else args.learning_rate,
        seed=args.seed,
        use_adaptive_lr=use_adaptive_lr
    )
    
    if success:
        print("🎉 Optimization completed successfully!")
    else:
        print("❌ Optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
