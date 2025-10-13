#!/usr/bin/env python3
"""
Dedicated optimization script for KI values
Creates deterministic representative blends and optimizes KI values to hit target percentage changes
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple

# Add the train/simulation directory to the path
sys.path.append('train/simulation')

# Suppress the specific UMM3 warnings about missing Glycerol-Glycerol pairs
warnings.filterwarnings("ignore", message="Material family pair 'Glycerol-Glycerol' not found in compatibility config")

from simulation_common import set_random_seeds
from simulation_rules import PROPERTY_CONFIGS
from umm3_correction import UMM3Correction, load_family_compatibility_config


def load_targets_from_csv() -> List[Dict[str, Any]]:
    """Load optimization targets from targets.csv"""
    try:
        df = pd.read_csv('train/simulation/targets.csv')
        targets = []
        for _, row in df.iterrows():
            if pd.notna(row['Polymer Family']) and pd.notna(row['Additive']) and pd.notna(row['Target (average % increase on representative blends)']):
                targets.append({
                    'polymer_family': row['Polymer Family'],
                    'additive': row['Additive'],
                    'target_percentage': float(row['Target (average % increase on representative blends)'])
                })
        return targets
    except Exception as e:
        print(f"❌ Error loading targets.csv: {e}")
        return []


def load_polymer_families_from_csv() -> Dict[str, List[Dict[str, str]]]:
    """Load polymer families from material-smiles-dictionary.csv"""
    try:
        df = pd.read_csv('material-smiles-dictionary.csv')
        families = {}
        for _, row in df.iterrows():
            material = row['Material']
            grade = row['Grade']
            smiles = row['SMILES']
            
            if material not in families:
                families[material] = []
            families[material].append({'grade': grade, 'smiles': smiles})
        
        return families
    except Exception as e:
        print(f"❌ Error loading material-smiles-dictionary.csv: {e}")
        return {}


def create_deterministic_representative_blend(polymer1_family: str, polymer2_family: str, additive: str, 
                                             property_name: str, ki_overrides: dict, with_additive: bool = True) -> float:
    """Create the EXACT same representative blend deterministically - no random filtering!"""
    
    # Get the first grade from each family
    families = load_polymer_families_from_csv()
    if polymer1_family not in families or polymer2_family not in families:
        return None
    
    polymer1_grade = families[polymer1_family][0]['grade']
    polymer2_grade = families[polymer2_family][0]['grade']
    
    # Debug: Print the exact blend we're creating
    blend_type = "40:40:20 with Glycerol" if with_additive else "50:50:0 without Glycerol"
    ki_key = f"{additive}-{polymer1_family}" if ki_overrides else "none"
    ki_value = ki_overrides.get(ki_key, 0.0) if ki_overrides else 0.0
    # print(f"    Creating blend: {polymer1_grade} vs {polymer2_grade} ({blend_type}) with KI={ki_value}")
    
    try:
        # Use the proper material mapping function to get polymers with all property values
        property_config = PROPERTY_CONFIGS[property_name]
        material_mapping = property_config['create_material_mapping'](enable_additives=True)
        
        # Get the specific polymer grades we need (material mapping uses family_grade format)
        polymer1_key = f"{polymer1_family}_{polymer1_grade}"
        polymer2_key = f"{polymer2_family}_{polymer2_grade}"
        
        if polymer1_key not in material_mapping or polymer2_key not in material_mapping:
            return None
        
        polymer1_data = material_mapping[polymer1_key]
        polymer2_data = material_mapping[polymer2_key]
        
        # Create the EXACT blend composition we want
        if with_additive:
            # 40:40:20 composition with Glycerol
            additive_key = f"{additive}_{additive}"
            if additive_key not in material_mapping:
                return None
            
            additive_data = material_mapping[additive_key]
            polymers = [polymer1_data, polymer2_data, additive_data]
            compositions = [0.4, 0.4, 0.2]  # 40:40:20
        else:
            # 50:50:0 composition without Glycerol
            polymers = [polymer1_data, polymer2_data]
            compositions = [0.5, 0.5]  # 50:50:0
        
        # Get the property-specific blend creation function
        create_blend_row_func = property_config['create_blend_row_func']
        
        # Create DETERMINISTIC environmental config to avoid random parameters
        if property_name == 'wvtr':
            # WVTR uses temperature, humidity, and thickness
            deterministic_env_config = {
                property_name: {
                    'temperature': {
                        'min': 38, 'max': 38,  # Fixed temperature: 38°C
                        'reference': 36,        # Reference temperature
                        'power_law': 0.44,     # Temperature scaling exponent
                        'scaling_type': 'power_law'
                    },
                    'humidity': {
                        'min': 90, 'max': 90,  # Fixed humidity: 90%
                        'reference': 85.0,     # Reference humidity
                        'power_law': 0.3,     # Humidity scaling exponent
                        'scaling_type': 'power_law'
                    },
                    'thickness': {
                        'min': 100, 'max': 100,  # Fixed thickness: 100 μm
                        'power_law': 0.5,        # Thickness scaling exponent
                        'reference': 50.0,       # Reference thickness
                        'scaling_type': 'fixed'
                    }
                }
            }
        elif property_name == 'otr':
            # OTR uses temperature, humidity, and thickness with different scaling
            deterministic_env_config = {
                property_name: {
                    'temperature': {
                        'min': 38, 'max': 38,  # Fixed temperature: 38°C
                        'reference': 23.0,      # Reference temperature
                        'max_scale': 5.0,      # Maximum scaling factor
                        'divisor': 10.0,       # Sensitivity control
                        'scaling_type': 'logarithmic'
                    },
                    'humidity': {
                        'min': 90, 'max': 90,  # Fixed humidity: 90%
                        'reference': 50.0,     # Reference humidity
                        'max_scale': 3.0,     # Maximum scaling factor
                        'divisor': 20.0,      # Sensitivity control
                        'scaling_type': 'logarithmic'
                    },
                    'thickness': {
                        'min': 100, 'max': 100,  # Fixed thickness: 100 μm
                        'power_law': 0.1,        # Thickness scaling exponent
                        'reference': 25.0,       # Reference thickness
                        'scaling_type': 'dynamic'
                    }
                }
            }
        else:
            # All other properties (eab, ts, cobb, seal, compost) only use thickness
            deterministic_env_config = {
                property_name: {
                    'thickness': {
                        'min': 100, 'max': 100,  # Fixed thickness: 100 μm
                        'power_law': 0.5,        # Thickness scaling exponent
                        'reference': 50.0,       # Reference thickness
                        'scaling_type': 'fixed'
                    }
                }
            }
        
        # Create the blend row directly with deterministic environmental parameters
        blend_row = create_blend_row_func(
            polymers=polymers,
            compositions=compositions,
            blend_number=1,  # Use 1 for consistency
            rule_tracker=None,
            selected_rules=None,
            environmental_config=deterministic_env_config
        )
        
        # Apply UMM3 correction if KI overrides are provided
        if ki_overrides:
            # Initialize UMM3 correction with overrides
            umm3 = UMM3Correction.from_config_files()
            umm3.ki_overrides = ki_overrides  # Set the overrides after creation
            
            # Debug: Print KI overrides being applied
            ki_key = f"{additive}-{polymer1_family}"
            ki_value = ki_overrides.get(ki_key, 0.0)
            # print(f"    Applying KI override: {ki_key} = {ki_value}")
            
            # Load family compatibility config
            family_config = load_family_compatibility_config(config_dir="train/simulation/config/compatibility", property_name=property_name)
            
            # Apply correction to the property value
            # Handle different property field names for different properties
            if property_name == 'eab':
                property_value = blend_row['property1']  # EAB uses property1
            else:
                property_value = blend_row['property']   # Other properties use property
                
            # Create corrected_values with the correct property key for UMM3
            if property_name == 'eab':
                corrected_values = {'property1': property_value}  # EAB uses property1
            else:
                corrected_values = {property_name: property_value}  # Other properties use property name
            original_value = property_value
            
            # For UMM3 correction, we need to include the additive in the polymers list
            if with_additive:
                # Add Glycerol to polymers and compositions for UMM3 correction
                additive_data = material_mapping[additive_key]
                polymers_for_umm3 = polymers + [additive_data]
                compositions_for_umm3 = compositions + [0.2]  # 20% additive
            else:
                polymers_for_umm3 = polymers
                compositions_for_umm3 = compositions
                
            corrected_values = umm3.apply_pairwise_compatibility_corrections(
                corrected_values, polymers_for_umm3, compositions_for_umm3, family_config, property_name
            )
            # Extract final value with correct property key
            if property_name == 'eab':
                final_value = corrected_values['property1']  # EAB uses property1
            else:
                final_value = corrected_values[property_name]  # Other properties use property name
            
            # Debug: Print the correction effect
            # print(f"    UMM3 correction: {original_value:.6f} -> {final_value:.6f} (change: {final_value - original_value:.6f})")
            
            return final_value
        else:
            # Handle different property field names for different properties
            if property_name == 'eab':
                return blend_row['property1']  # EAB uses property1
            else:
                return blend_row['property']   # Other properties use property
        
    except Exception as e:
        return None


def calculate_family_percentage_change(polymer_family: str, additive: str, property_name: str, 
                                     ki_overrides: dict) -> Dict[str, Any]:
    """Calculate percentage change for a polymer family across all its binary blends"""
    families = load_polymer_families_from_csv()
    
    if polymer_family not in families:
        return None
    
    # Get all other polymer families for binary blends
    other_families = [fam for fam in families.keys() if fam != polymer_family]
    
    binary_blend_results = []
    
    for other_family in other_families:
        # Create binary blend: polymer_family vs other_family using DETERMINISTIC representative blends
        no_additive_value = create_deterministic_representative_blend(
            polymer_family, other_family, additive, property_name, ki_overrides, with_additive=False
        )
        with_additive_value = create_deterministic_representative_blend(
            polymer_family, other_family, additive, property_name, ki_overrides, with_additive=True
        )
        
        if no_additive_value is not None and with_additive_value is not None:
            if no_additive_value != 0:
                percentage_change = ((with_additive_value - no_additive_value) / no_additive_value) * 100
            else:
                percentage_change = 0.0
            
            binary_blend_results.append({
                'other_family': other_family,
                'no_additive_value': no_additive_value,
                'with_additive_value': with_additive_value,
                'percentage_change': percentage_change
            })
    
    if not binary_blend_results:
        return None
    
    # Calculate average percentage change for this polymer family
    avg_percentage_change = sum(result['percentage_change'] for result in binary_blend_results) / len(binary_blend_results)
    
    return {
        'polymer_family': polymer_family,
        'additive': additive,
        'binary_blends': binary_blend_results,
        'average_percentage_change': avg_percentage_change,
        'num_blends': len(binary_blend_results)
    }


def run_optimization(property_name: str, tolerance: float, max_iterations: int, learning_rate: float, seed: int) -> bool:
    """Run full optimization for all targets in targets.csv with gradient descent"""
    print(f"🚀 Starting optimization for {property_name}")
    print(f"   Tolerance: ±{tolerance}%")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Random seed: {seed}")
    
    # Set random seeds for reproducibility
    set_random_seeds(seed)
    
    # Load targets from CSV
    targets = load_targets_from_csv()
    if not targets:
        print("❌ No targets found in targets.csv")
        return False
    
    print(f"📋 Found {len(targets)} targets to optimize:")
    for target in targets:
        print(f"   - {target['polymer_family']} + {target['additive']}: {target['target_percentage']}%")
    
    # Initialize KI vector for all targets simultaneously
    n_targets = len(targets)
    ki_vector = np.zeros(n_targets)  # Initialize all KI values to 0.0
    best_ki_vector = ki_vector.copy()
    best_total_error = float('inf')
    
    print(f"\n🎯 Optimizing ALL {n_targets} targets simultaneously")
    print(f"   Vector approach: accounting for KI interactions")
    print(f"   {'='*80}")
    print(f"   {'ITERATION':>8} | {'TOTAL ERROR':>12} | {'INDIVIDUAL ERRORS':>50}")
    print(f"   {'='*80}")
    print(f"   {'-' * 100}")
    
    all_results = []
    adaptive_lr = learning_rate
    
    # Previous values for gradient calculation
    prev_ki_vector = None
    prev_individual_errors = None
    
    for iteration in range(max_iterations):
        # Create KI overrides with ALL KI values from the vector
        ki_overrides = {}
        for target_idx, target in enumerate(targets):
            ki_key = f"{target['additive']}-{target['polymer_family']}"
            ki_overrides[ki_key] = ki_vector[target_idx]
        
        # Calculate percentage changes for ALL polymer families using the same ki_overrides
        individual_errors = []
        individual_changes = []
        
        for target_idx, target in enumerate(targets):
            # Calculate percentage change for this polymer family using ALL KI values
            family_result = calculate_family_percentage_change(
                target['polymer_family'], target['additive'], property_name, ki_overrides
            )
            
            if family_result is None:
                print(f"   ❌ Failed to calculate percentage change for {target['polymer_family']}")
                individual_errors.append(float('inf'))
                individual_changes.append(0.0)
                continue
            
            avg_change = family_result['average_percentage_change']
            error = abs(avg_change - target['target_percentage'])
            individual_errors.append(error)
            individual_changes.append(avg_change)
        
        # Calculate total error (sum of all individual errors)
        total_error = sum(individual_errors)
        
        # Log results for this iteration
        for target_idx, target in enumerate(targets):
            all_results.append({
                'iteration': iteration + 1,
                'ki_value': ki_vector[target_idx],
                'percentage_change': individual_changes[target_idx],
                'error': individual_errors[target_idx],
                'polymer_family': target['polymer_family'],
                'additive': target['additive']
            })
        
        # Print iteration summary
        # Check if all targets are within tolerance for status display
        all_within_tolerance_status = True
        for i, target in enumerate(targets):
            individual_error = abs(individual_errors[i])
            target_value = target['target_percentage']
            tolerance_threshold = target_value * (tolerance / 100.0)
            if individual_error > tolerance_threshold:
                all_within_tolerance_status = False
                break
        
        status = "✅ TARGET!" if all_within_tolerance_status else "🔄"
        error_str = " | ".join([f"{target['polymer_family']}:{err:.1f}" for target, err in zip(targets, individual_errors)])
        print(f"   {iteration+1:8d} | {total_error:12.1f} | {error_str}")
        
        # Check if we hit all targets (each target within ±tolerance% of its target value)
        all_within_tolerance = True
        for i, target in enumerate(targets):
            individual_error = abs(individual_errors[i])
            target_value = target['target_percentage']
            tolerance_threshold = target_value * (tolerance / 100.0)  # Convert tolerance to percentage of target
            
            if individual_error > tolerance_threshold:
                all_within_tolerance = False
                break
        
        if all_within_tolerance:
            best_ki_vector = ki_vector.copy()
            best_total_error = total_error
            print(f"   🎉 All targets achieved! Total error: {total_error:.1f}")
            break
        
        # Update best KI vector if this is better
        if total_error < best_total_error:
            best_total_error = total_error
            best_ki_vector = ki_vector.copy()
        
        # Vector gradient descent
        if iteration > 0 and prev_ki_vector is not None and prev_individual_errors is not None:
            # Calculate gradients for each KI value individually
            ki_gradients = np.zeros(n_targets)
            
            for target_idx in range(n_targets):
                ki_diff = ki_vector[target_idx] - prev_ki_vector[target_idx]
                error_diff = individual_errors[target_idx] - prev_individual_errors[target_idx]
                
                if abs(ki_diff) > 1e-8:
                    gradient = error_diff / ki_diff
                    ki_gradients[target_idx] = max(-10.0, min(10.0, gradient))
                    # print(f"   Debug {targets[target_idx]['polymer_family']}: ki_diff={ki_diff:.4f}, error_diff={error_diff:.4f}, gradient={gradient:.4f}")
                else:
                    # If KI difference is too small, use error direction
                    if individual_errors[target_idx] > tolerance:
                        # For WVTR: if percentage_change < target, we want to increase KI (positive gradient)
                        # For WVTR: if percentage_change > target, we want to decrease KI (negative gradient)
                        if individual_changes[target_idx] < targets[target_idx]['target_percentage']:
                            ki_gradients[target_idx] = 0.1  # Positive gradient to increase KI
                        else:
                            ki_gradients[target_idx] = -0.1  # Negative gradient to decrease KI
                        # print(f"   Debug {targets[target_idx]['polymer_family']}: using fallback gradient={ki_gradients[target_idx]:.4f} (change={individual_changes[target_idx]:.1f}% vs target={targets[target_idx]['target_percentage']:.1f}%)")
                    else:
                        ki_gradients[target_idx] = 0.0
            
            # Update KI vector - each KI gets its own update based on its individual gradient
            # For WVTR: if gradient is positive (we want to increase KI), move KI in same direction
            # This means: ki_updates = +adaptive_lr * ki_gradients for WVTR
            ki_updates = adaptive_lr * ki_gradients
            
            # No KI update clamping - allow larger changes
            # ki_updates = np.clip(ki_updates, -0.5, 0.5)
            # print(f"   Debug: ki_gradients={ki_gradients}")
            # print(f"   Debug: ki_updates={ki_updates}")
            ki_vector = ki_vector + ki_updates
            
        else:
            # First iteration - try small steps based on individual errors
            if iteration == 0:
                # Stay at 0.0 for first test
                ki_vector = np.zeros(n_targets)
            else:
                # Second iteration - choose directions based on individual errors
                for target_idx in range(n_targets):
                    if individual_changes[target_idx] < targets[target_idx]['target_percentage']:
                        ki_vector[target_idx] = 0.1  # Small positive step
                    else:
                        ki_vector[target_idx] = -0.1  # Small negative step
                print(f"   🚀 Directional steps: {ki_vector}")
        
        # No KI value clamping - allow unlimited values for extreme targets
        # ki_vector = np.clip(ki_vector, -2.0, 2.0)
        
        # Store previous values
        prev_ki_vector = ki_vector.copy()
        prev_individual_errors = individual_errors.copy()
        
        # Early stopping if we're not making progress
        if iteration > 5:
            recent_errors = [r['error'] for r in all_results[-n_targets*3:]]
            if len(set([round(e, 1) for e in recent_errors])) == 1:
                print(f"   ⚠️  Early stopping - no progress in last 3 iterations")
                break
    
    # Print final results
    print(f"\n{'='*80}")
    print(f"🎯 OPTIMIZATION COMPLETED")
    print(f"{'='*80}")
    print(f"Best Total Error: {best_total_error:.2f}")
    print(f"\nOptimized KI Values:")
    print(f"{'Polymer':<12} | {'KI Value':<10} | {'Target':<8} | {'Achieved':<10} | {'Error':<8}")
    print(f"{'-'*60}")
    
    # Calculate final individual errors using ALL KI values
    final_ki_overrides = {}
    for target_idx, target in enumerate(targets):
        ki_key = f"{target['additive']}-{target['polymer_family']}"
        final_ki_overrides[ki_key] = best_ki_vector[target_idx]
    
    final_errors = []
    for target_idx, target in enumerate(targets):
        family_result = calculate_family_percentage_change(
            target['polymer_family'], target['additive'], property_name, final_ki_overrides
        )
        
        if family_result is not None:
            avg_change = family_result['average_percentage_change']
            error = abs(avg_change - target['target_percentage'])
            final_errors.append(error)
            
            # Check if within tolerance using percentage of target value
            target_value = target['target_percentage']
            tolerance_threshold = target_value * (tolerance / 100.0)
            status = "✅" if error <= tolerance_threshold else "❌"
            print(f"{target['polymer_family']:<12} | {best_ki_vector[target_idx]:<10.3f} | {target['target_percentage']:<8.0f}% | {avg_change:<10.1f}% | {error:<8.1f}% {status}")
        else:
            final_errors.append(float('inf'))
            print(f"{target['polymer_family']:<12} | {best_ki_vector[target_idx]:<10.3f} | {target['target_percentage']:<8.0f}% | {'FAILED':<10} | {'∞':<8} ❌")
    
    # Count targets within tolerance (using percentage of target value)
    success_count = 0
    for i, target in enumerate(targets):
        individual_error = abs(final_errors[i])
        target_value = target['target_percentage']
        tolerance_threshold = target_value * (tolerance / 100.0)
        if individual_error <= tolerance_threshold:
            success_count += 1
    
    print(f"{'-'*60}")
    print(f"SUCCESS: {success_count}/{n_targets} targets achieved within ±{tolerance}% of target value")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('optimization_results.csv', index=False)
    print(f"\n📊 All optimizations completed!")
    print(f"   Results saved to: optimization_results.csv")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Optimize KI values for additive effects')
    parser.add_argument('--property', required=True, 
                       choices=['wvtr', 'otr', 'ts', 'eab', 'cobb', 'seal', 'compost'],
                       help='Property to optimize')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='Tolerance for target achievement (default: 5.0)')
    parser.add_argument('--max-iterations', type=int, default=50,
                       help='Maximum optimization iterations (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=0.1,
                       help='Learning rate for gradient descent (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Run optimization
    success = run_optimization(
        args.property,
        args.tolerance,
        args.max_iterations,
        args.learning_rate,
        args.seed
    )
    
    if success:
        print("\n🎉 Optimization completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Optimization failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
