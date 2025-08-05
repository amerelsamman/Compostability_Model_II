#!/usr/bin/env python3
"""
Modularized training script for Differentiable Label Optimization.
Uses the modularized components from the modules/ directory.
"""

import torch
import numpy as np
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')

from modules_home import DifferentiableLabelOptimizer
from sklearn.metrics import r2_score, mean_absolute_error


def main():
    """Main function to run Differentiable Label Optimization with modular components."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DLO model with modular components')
    parser.add_argument('--data_file', type=str, default='data/training_features.csv', 
                       help='Path to training data file (default: data/training.csv)')
    parser.add_argument('--save_dir', type=str, default='models/v1/', 
                       help='Directory to save model (default: models/v1/)')
    parser.add_argument('--model_name', type=str, default='dlo_model', 
                       help='Model name (default: dlo_model)')
    parser.add_argument('--epochs', type=int, default=1000, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--oversample_high_maxl', action='store_true', default=True,
                       help='Oversample samples with high max_L values (default: True)')
    parser.add_argument('--oversample_low_maxl', action='store_true', default=True,
                       help='Oversample samples with low max_L values (default: True)')
    parser.add_argument('--high_oversample_factor', type=int, default=10,
                       help='Oversampling factor for high max_L samples (default: 5)')
    parser.add_argument('--low_oversample_factor', type=int, default=10,
                       help='Oversampling factor for low max_L samples (default: 5)')
    parser.add_argument('--high_maxl_threshold', type=float, default=90.0,
                       help='Threshold for high max_L values (default: 90.0)')
    parser.add_argument('--low_maxl_threshold', type=float, default=5.0,
                       help='Threshold for low max_L values (default: 5.0)')
    parser.add_argument('--high_maxl_weight', type=float, default=3.0,
                       help='Weight multiplier for high max_L samples in loss (default: 3.0)')
    parser.add_argument('--low_maxl_weight', type=float, default=3.0,
                       help='Weight multiplier for low max_L samples in loss (default: 3.0)')
    parser.add_argument('--no_weighted_loss', action='store_true',
                       help='Disable weighted loss function (use standard MSE)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Data file: {args.data_file}")
    print(f"Save directory: {args.save_dir}")
    
    # Debug: Check if data file exists and show its contents
    import os
    if os.path.exists(args.data_file):
        print(f"✅ Data file exists: {args.data_file}")
        import pandas as pd
        df = pd.read_csv(args.data_file)
        print(f"Data shape: {df.shape}")
        print(f"Property1 range: {df['property1'].min()} - {df['property1'].max()}")
        print(f"Property2 range: {df['property2'].min()} - {df['property2'].max()}")
        print(f"Sample property1 values: {df['property1'].head().tolist()}")
    else:
        print(f"❌ Data file does not exist: {args.data_file}")
        sys.exit(1)
    
    # Initialize optimizer
    dlo = DifferentiableLabelOptimizer(device=device)
    
    # Prepare data
    features, labels, soft_indices = dlo.prepare_data(
        data_file=args.data_file,
        use_custom_labels=True,  # Use custom label assignments from CSV
        oversample_high_maxl=args.oversample_high_maxl,  # Oversample high max_L samples
        oversample_low_maxl=args.oversample_low_maxl,  # Oversample low max_L samples
        high_oversample_factor=args.high_oversample_factor,  # Oversampling factor for high max_L
        low_oversample_factor=args.low_oversample_factor,  # Oversampling factor for low max_L
        high_maxl_threshold=args.high_maxl_threshold,  # Threshold for high max_L
        low_maxl_threshold=args.low_maxl_threshold  # Threshold for low max_L
    )
    
    # Train model
    history = dlo.train(
        features=features,
        labels=labels,
        soft_indices=soft_indices,
        epochs=args.epochs,
        batch_size=16,
        patience=20,
        hard_label_weight=1.0,
        soft_label_weight=0.5,
        use_weighted_loss=not args.no_weighted_loss,  # Use custom weighted loss
        high_maxl_weight=args.high_maxl_weight,  # 3x weight for high max_L samples
        low_maxl_weight=args.low_maxl_weight,  # 3x weight for low max_L samples
        high_maxl_threshold=args.high_maxl_threshold,  # Threshold for high max_L
        low_maxl_threshold=args.low_maxl_threshold  # Threshold for low max_L
    )
    
    # Get label changes
    label_changes = dlo.get_soft_label_changes(labels)
    
    # Evaluate final performance
    final_metrics, predictions = dlo.evaluate(features, labels)
    
    print("\n" + "="*60)
    print("DIFFERENTIABLE LABEL OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Final Performance:")
    print(f"Property1 - R²: {final_metrics['property1_r2']:.4f}, MAE: {final_metrics['property1_mae']:.4f}")
    print(f"Property2 - R²: {final_metrics['property2_r2']:.4f}, MAE: {final_metrics['property2_mae']:.4f}")
    print(f"Overall - R²: {final_metrics['overall_r2']:.4f}, MAE: {final_metrics['overall_mae']:.4f}")
    
    print(f"\nSoft Label Statistics:")
    print(f"Number of soft labels: {len(soft_indices)}")
    if len(label_changes['changes']) > 0:
        print(f"Average label change: {np.mean(np.abs(label_changes['changes'])):.4f}")
        print(f"Max label change: {np.max(np.abs(label_changes['changes'])):.4f}")
    
    # Analyze performance on high and low max_L samples
    high_maxl_mask = labels[:, 0] > args.high_maxl_threshold
    low_maxl_mask = labels[:, 0] < args.low_maxl_threshold
    
    if np.any(high_maxl_mask):
        high_maxl_metrics = {}
        for i, prop_name in enumerate(['property1', 'property2']):
            high_maxl_r2 = r2_score(labels[high_maxl_mask, i], predictions[high_maxl_mask, i])
            high_maxl_mae = mean_absolute_error(labels[high_maxl_mask, i], predictions[high_maxl_mask, i])
            high_maxl_metrics[f'{prop_name}_r2'] = high_maxl_r2
            high_maxl_metrics[f'{prop_name}_mae'] = high_maxl_mae
        
        print(f"\nHigh max_L (> {args.high_maxl_threshold}) Performance:")
        print(f"Number of high max_L samples: {np.sum(high_maxl_mask)}")
        print(f"Property1 (max_L) - R²: {high_maxl_metrics['property1_r2']:.4f}, MAE: {high_maxl_metrics['property1_mae']:.4f}")
        print(f"Property2 (t0) - R²: {high_maxl_metrics['property2_r2']:.4f}, MAE: {high_maxl_metrics['property2_mae']:.4f}")
    
    if np.any(low_maxl_mask):
        low_maxl_metrics = {}
        for i, prop_name in enumerate(['property1', 'property2']):
            low_maxl_r2 = r2_score(labels[low_maxl_mask, i], predictions[low_maxl_mask, i])
            low_maxl_mae = mean_absolute_error(labels[low_maxl_mask, i], predictions[low_maxl_mask, i])
            low_maxl_metrics[f'{prop_name}_r2'] = low_maxl_r2
            low_maxl_metrics[f'{prop_name}_mae'] = low_maxl_mae
        
        print(f"\nLow max_L (< {args.low_maxl_threshold}) Performance:")
        print(f"Number of low max_L samples: {np.sum(low_maxl_mask)}")
        print(f"Property1 (max_L) - R²: {low_maxl_metrics['property1_r2']:.4f}, MAE: {low_maxl_metrics['property1_mae']:.4f}")
        print(f"Property2 (t0) - R²: {low_maxl_metrics['property2_r2']:.4f}, MAE: {low_maxl_metrics['property2_mae']:.4f}")
    
    # Compare with overall performance
    if np.any(high_maxl_mask) or np.any(low_maxl_mask):
        print(f"\nPerformance Comparison:")
        if np.any(high_maxl_mask):
            print(f"Overall Property1 R²: {final_metrics['property1_r2']:.4f} vs High max_L: {high_maxl_metrics['property1_r2']:.4f}")
            print(f"Overall Property2 R²: {final_metrics['property2_r2']:.4f} vs High max_L: {high_maxl_metrics['property2_r2']:.4f}")
        if np.any(low_maxl_mask):
            print(f"Overall Property1 R²: {final_metrics['property1_r2']:.4f} vs Low max_L: {low_maxl_metrics['property1_r2']:.4f}")
            print(f"Overall Property2 R²: {final_metrics['property2_r2']:.4f} vs Low max_L: {low_maxl_metrics['property2_r2']:.4f}")
    
    # Create plots
    dlo.plot_results(features, labels, history, label_changes, save_dir=args.save_dir)
    
    # Save results
    results_df = dlo.save_results(features, labels, label_changes, save_dir=args.save_dir)
    
    # Save the trained model
    print("\n" + "="*60)
    print("SAVING TRAINED MODEL")
    print("="*60)
    model_paths = dlo.save_model(save_dir=args.save_dir, model_name=args.model_name)
    
    # Generate sigmoid curves from optimized labels
    print("\n" + "="*60)
    print("GENERATING SIGMOID CURVES")
    print("="*60)
    
    # Get optimized labels
    optimized_labels = dlo.dataset.soft_labels.detach().cpu().numpy()
    max_L_values = optimized_labels[:, 0]  # property1 = max_L
    t0_values = optimized_labels[:, 1]     # property2 = t0
    
    # Calculate k0 for disintegration curves (200 days)
    # For training, we need to determine majority polymer behavior for each sample
    # Load original data to get polymer information
    original_data = pd.read_csv(args.data_file)
    
    k0_values_disintegration = []
    for i, (max_L, t0) in enumerate(zip(max_L_values, t0_values)):
        # Determine majority polymer behavior for this sample
        # Check if majority polymer has max_L > 5 or < 5
        majority_high_disintegration = None  # Default to original logic for training
        
        # For now, use original logic (None) - this can be enhanced later if needed
        # Use default thickness (50μm) for training curves
        k0 = dlo.calculate_k0_from_sigmoid_params(max_L, t0, t_max=200.0, 
                                                 majority_polymer_high_disintegration=majority_high_disintegration,
                                                 actual_thickness=0.050)  # 50μm default
        k0_values_disintegration.append(k0)
    
    k0_values_disintegration = np.array(k0_values_disintegration)
    
    print(f"Disintegration k0 values - Min: {k0_values_disintegration.min():.4f}, Max: {k0_values_disintegration.max():.4f}, Mean: {k0_values_disintegration.mean():.4f}")
    
    # Generate and save disintegration curves
    disintegration_df = dlo.generate_sigmoid_curves(max_L_values, t0_values, k0_values_disintegration, 
                                                   days=200, curve_type='disintegration', save_dir=args.save_dir,
                                                   actual_thickness=0.050)  # 50μm default for training
    
    # Calculate biodegradation parameters (t0 doubled, 400 days)
    t0_values_biodegradation = t0_values * 2.0  # Double the t0 values
    k0_values_biodegradation = []
    for max_L, t0_bio in zip(max_L_values, t0_values_biodegradation):
        # Use same logic as disintegration for consistency
        majority_high_disintegration = None  # Default to original logic for training
        # Use default thickness (50μm) for training curves
        k0 = dlo.calculate_k0_from_sigmoid_params(max_L, t0_bio, t_max=400.0, 
                                                 majority_polymer_high_disintegration=majority_high_disintegration,
                                                 actual_thickness=0.050)  # 50μm default
        k0_values_biodegradation.append(k0)
    
    k0_values_biodegradation = np.array(k0_values_biodegradation)
    
    print(f"Biodegradation k0 values - Min: {k0_values_biodegradation.min():.4f}, Max: {k0_values_biodegradation.max():.4f}, Mean: {k0_values_biodegradation.mean():.4f}")
    
    # Generate and save biodegradation curves
    biodegradation_df = dlo.generate_sigmoid_curves(max_L_values, t0_values_biodegradation, k0_values_biodegradation, 
                                                   days=400, curve_type='biodegradation', save_dir=args.save_dir,
                                                   actual_thickness=0.050)  # 50μm default for training
    
    return dlo, results_df, final_metrics


if __name__ == "__main__":
    dlo, results, metrics = main() 