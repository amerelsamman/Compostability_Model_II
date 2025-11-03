"""
Results Handler Module
Functions for evaluating performance and saving detailed results
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from .data_loader import load_existing_ki_values
from .family_manager import extract_polymer_families_from_blend, get_polymer_family_groups, get_family_group_for_polymer
from .simulation_engine import simulate_validation_blend


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
