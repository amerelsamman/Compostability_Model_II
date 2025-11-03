#!/usr/bin/env python3
"""
Fully Modularized Blend Optimization Script
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

from simulation_common import set_random_seeds

# Import ALL functions from our modules (NO imports from original file)
from optimization_modules import (
    # Data loading
    load_validation_data,
    load_material_families,
    load_existing_ki_values,
    
    # Family management
    extract_polymer_families_from_blend,
    get_polymer_family_groups,
    get_family_group_for_polymer,
    get_ki_overrides_from_vector,
    
    # Simulation
    simulate_validation_blend,
    calculate_validation_mae,
    calculate_blend_specific_gradients,
    
    # Optimization core
    build_ki_vector,
    calculate_blend_learning_rates,
    optimize_blend_ki_values,
    update_compatibility_file,
    optimize_family_group_ki_values,
    update_family_group_compatibility_file,
    
    # Visualization
    create_optimization_plots,
    create_validation_performance_plots,
    create_testing_performance_plot,
    
    # Results handling
    evaluate_performance_on_dataset,
    save_detailed_results_csv
)


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
    parser.add_argument('--hard-freeze', action='store_true', default=False,
                       help='Freeze blends whose relative MAE is within tolerance (stop updating from them)')
    parser.add_argument('--freeze-tolerance', type=float, default=5.0,
                       help='Relative error tolerance in percent for hard-freeze (default: 5.0)')
    
    # Adaptive learning rate control parameters
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
    
    # Calculate initial predictions (before optimization) for plotting
    print("📊 Calculating initial predictions...")
    initial_ki_vector, initial_pair_mapping = build_ki_vector(training_df, family_mapping, args.property)
    initial_mae, initial_predictions, initial_actual = evaluate_performance_on_dataset(
        training_df, args.property, family_mapping, initial_ki_vector, initial_pair_mapping, 
        use_optimized_ki=False, use_family_groups=False
    )
    print(f"Initial MAE: {initial_mae:.4f}")
    
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
            training_df=training_df,
            hard_freeze=args.hard_freeze,
            freeze_tolerance=args.freeze_tolerance / 100.0
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
                training_df, args.property, initial_mae, training_mae, 
                initial_predictions, training_predictions, ".", ""
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
                testing_actual, testing_predictions, ".", ""
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
