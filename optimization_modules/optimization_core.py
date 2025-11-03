"""
Optimization Core Module
Core optimization algorithms and KI vector building
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import yaml
import re

# Add the train/simulation directory to the path
sys.path.append('train/simulation')

from simulation_common import set_random_seeds
from umm3_correction import UMM3Correction, load_family_compatibility_config

from .data_loader import load_existing_ki_values
from .family_manager import extract_polymer_families_from_blend, get_polymer_family_groups, get_family_group_for_polymer
from .simulation_engine import calculate_validation_mae, calculate_blend_specific_gradients


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


def optimize_blend_ki_values(property_name: str, max_iterations: int, learning_rate: float, seed: int, 
                           use_adaptive_lr: bool = True, gradient_threshold: float = 2.0,
                           max_step_size: float = 0.5, training_df: pd.DataFrame = None,
                           hard_freeze: bool = False, freeze_tolerance: float = 0.05) -> Tuple[bool, np.ndarray, Dict[str, int]]:
    """Optimize KI values for polymer-polymer compatibility."""
    print(f"🚀 Starting blend optimization for {property_name}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Random seed: {seed}")
    
    # Set random seed (matches original behavior)
    set_random_seeds(seed)
    
    # Use provided training data or load it
    if training_df is None:
        from .data_loader import load_validation_data
        validation_df, _ = load_validation_data(property_name)
    else:
        validation_df = training_df
    
    from .data_loader import load_material_families
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
        is_new_best = current_mae < best_mae
        if is_new_best:
            best_mae = current_mae
            best_ki_vector = ki_vector.copy()
        
        # Status
        status = "✅ NEW BEST" if is_new_best else "🔄"
        
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
                gradients, blend_gradients, current_errors = calculate_blend_specific_gradients(
                    validation_df, property_name, family_mapping, ki_vector, pair_mapping
                )
                
                # Calculate blend-specific learning rates
                blend_lrs = calculate_blend_learning_rates(current_errors, learning_rate)

                # Optional hard-freeze: zero out LR for blends within tolerance
                if hard_freeze:
                    # We need per-blend relative error; approximate by error/expected
                    rel_errors = []
                    for idx, row in validation_df.iterrows():
                        if property_name in ['ts', 'eab']:
                            exp_value = row['property1'] if 'property1' in row and not pd.isna(row['property1']) else None
                        else:
                            exp_value = row['property'] if 'property' in row and not pd.isna(row['property']) else None
                        err = current_errors[idx] if idx < len(current_errors) else None
                        if exp_value and exp_value != 0 and err is not None:
                            rel = abs(err / exp_value)
                        else:
                            rel = float('inf')
                        rel_errors.append(rel)
                    for j in range(len(blend_lrs)):
                        if rel_errors[j] != float('inf') and rel_errors[j] <= freeze_tolerance:
                            blend_lrs[j] = 0.0
                
                # Calculate weighted gradient (weight by learning rates)
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
                        weighted_gradient[i] = gradients[i]  # Fallback to average gradient
                
                # Normalize the weighted gradient to prevent explosion
                gradient_norm = np.linalg.norm(weighted_gradient)
                if gradient_norm > 0:
                    # Scale down very large gradients but allow reasonable steps
                    if gradient_norm > gradient_threshold:
                        weighted_gradient = weighted_gradient / gradient_norm * max_step_size
                
                # Update KI vector
                ki_vector = ki_vector - weighted_gradient
                
                # Print gradient info
                print(f"      Gradient norm: {gradient_norm:.8f}, Blend LR: avg={np.mean(blend_lrs):.4f}, min={np.min(blend_lrs):.4f}, max={np.max(blend_lrs):.4f}")
                print(f"      KI changes: {np.linalg.norm(weighted_gradient):.8f}")
                
            else:
                # Use simple gradient descent
                gradients, _, _ = calculate_blend_specific_gradients(
                    validation_df, property_name, family_mapping, ki_vector, pair_mapping
                )
                
                # Normalize gradient
                gradient_norm = np.linalg.norm(gradients)
                if gradient_norm > 0:
                    if gradient_norm > gradient_threshold:
                        gradients = gradients / gradient_norm * max_step_size
                
                # Update KI vector
                ki_vector = ki_vector - learning_rate * gradients
                
                print(f"      Gradient norm: {gradient_norm:.8f}")
                print(f"      KI changes: {np.linalg.norm(learning_rate * gradients):.8f}")
        
        # Store previous values
        prev_mae = current_mae
        prev_ki_vector = ki_vector.copy()
    
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Best MAE: {best_mae:.4f}")
    print(f"Improvement: {initial_mae - best_mae:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('blend_optimization_results.csv', index=False)
    print(f"📊 Results saved to: blend_optimization_results.csv")
    
    # Note: Plotting functions will be called from main script
    # to avoid circular imports
    
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
    """Optimize KI values using family group approach."""
    print(f"🚀 Starting family group optimization for {property_name}")
    
    # First run regular optimization
    success, best_ki_vector, pair_mapping = optimize_blend_ki_values(
        property_name, max_iterations, learning_rate, seed, use_adaptive_lr, 
        gradient_threshold, max_step_size, training_df
    )
    
    if not success:
        return False, np.array([]), {}
    
    # Get family groups
    family_groups = get_polymer_family_groups()
    
    # Create family group pairs
    family_pairs = {}
    for group_name, families in family_groups.items():
        for i in range(len(families)):
            for j in range(i + 1, len(families)):
                pair_name = f"{families[i]}-{families[j]}"
                family_pairs[pair_name] = 0.0  # Default KI value
    
    # Copy optimized KI values to family group pairs
    for pair_name, ki_value in zip(pair_mapping.keys(), best_ki_vector):
        if pair_name in family_pairs:
            family_pairs[pair_name] = ki_value
    
    # Update family group compatibility file
    update_family_group_compatibility_file(family_pairs, property_name)
    
    return True, best_ki_vector, pair_mapping


def update_family_group_compatibility_file(family_pairs: Dict[str, float], property_name: str):
    """Update family group compatibility file."""
    print(f"📝 Creating family group optimized copy: train/simulation/config/compatibility/{property_name}_compatibility_family_optimized.yaml")
    
    # Load original file
    original_file = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
    family_file = f"train/simulation/config/compatibility/{property_name}_compatibility_family_optimized.yaml"
    
    try:
        with open(original_file, 'r') as f:
            content = f.read()
        
        # Update KI values for family pairs
        for pair_name, ki_value in family_pairs.items():
            pattern = rf'("{pair_name}"\s*:\s*{{KI:\s*)[0-9.-]+'
            replacement = rf'\g<1>{ki_value:.6f}'
            content = re.sub(pattern, replacement, content)
        
        # Write family group file
        with open(family_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Created family group optimized copy: {family_file}")
        
    except Exception as e:
        print(f"❌ Error updating family group compatibility file: {e}")
