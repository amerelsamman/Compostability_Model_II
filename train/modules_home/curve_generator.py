"""
Curve generation module for compostability predictions.
This consolidates all curve generation logic into a single interface.
"""

import os
import numpy as np
import pandas as pd
from .utils import (
    calculate_k0_from_sigmoid_params,
    generate_sigmoid_curves,
    generate_quintic_biodegradation_curve
)


def generate_compostability_curves(max_L_pred, t0_pred, thickness, 
                                 output_dir="test_results/predict_blend_properties",
                                 save_csv=True, save_plot=True):
    """
    Generate all curves for compostability prediction in one function.
    
    Args:
        max_L_pred (float): Predicted maximum disintegration level
        t0_pred (float): Predicted time to 50% disintegration
        thickness (float): Material thickness in mm
        output_dir (str): Directory to save outputs
        save_csv (bool): Whether to save CSV files
        save_plot (bool): Whether to save plot files
    
    Returns:
        dict: Dictionary containing all curve data and metadata
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate k0 values using curve generation modules
        k0_disintegration = calculate_k0_from_sigmoid_params(
            max_L_pred, t0_pred, t_max=200.0, 
            majority_polymer_high_disintegration=True,
            actual_thickness=thickness
        )
        k0_biodegradation = calculate_k0_from_sigmoid_params(
            max_L_pred, t0_pred * 2.0, t_max=400.0, 
            majority_polymer_high_disintegration=True,
            actual_thickness=thickness
        )
        
        # Generate disintegration curves
        disintegration_df = generate_sigmoid_curves(
            np.array([max_L_pred]), 
            np.array([t0_pred]), 
            np.array([k0_disintegration]), 
            days=200, 
            curve_type='disintegration',
            save_csv=save_csv,
            save_plot=save_plot,
            save_dir=output_dir,
            actual_thickness=thickness
        )
        
        # Generate biodegradation curves
        biodegradation_df = generate_quintic_biodegradation_curve(
            disintegration_df, 
            t0_pred, 
            max_L_pred, 
            days=400, 
            save_csv=save_csv,
            save_plot=save_plot,
            save_dir=output_dir,
            actual_thickness=thickness
        )
        
        # Determine home-compostable status
        is_home_compostable = max_L_pred >= 90.0
        
        # Calculate maximum biodegradation
        max_biodegradation = (biodegradation_df['biodegradation'].max() 
                             if biodegradation_df is not None and not biodegradation_df.empty 
                             else max_L_pred)
        
        # Return comprehensive results
        return {
            'max_biodegradation': max_biodegradation,
            'is_home_compostable': is_home_compostable,
            'k0_disintegration': k0_disintegration,
            'k0_biodegradation': k0_biodegradation,
            'disintegration_curve': disintegration_df,
            'biodegradation_curve': biodegradation_df,
            'disintegration_df': disintegration_df,  # Keep both names for compatibility
            'biodegradation_df': biodegradation_df   # Keep both names for compatibility
        }
        
    except Exception as e:
        print(f"❌ Curve generation failed: {e}")
        return None
