#!/usr/bin/env python3
"""
Test script for quartic biodegradation curve generation.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add modules to path
sys.path.append('modules')

from modules_home.utils import generate_sigmoid_curves, generate_cubic_biodegradation_curve

def test_cubic_biodegradation():
    """Test the cubic biodegradation curve generation."""
    print("Testing cubic biodegradation curve generation...")
    
    # Create test parameters
    max_L = 85.0  # Maximum disintegration level
    t0 = 45.0     # Time to 50% disintegration
    k0 = 0.15     # Rate constant
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate disintegration curve first (sigmoid)
    print(f"\nGenerating disintegration curve with:")
    print(f"  max_L: {max_L}")
    print(f"  t0: {t0}")
    print(f"  k0: {k0}")
    
    disintegration_df = generate_sigmoid_curves(
        np.array([max_L]), 
        np.array([t0]), 
        np.array([k0]), 
        days=200, 
        curve_type='disintegration',
        save_dir=output_dir,
        save_csv=True,
        save_plot=True
    )
    
    print(f"Disintegration curve generated with {len(disintegration_df)} data points")
    
    # Now generate quartic biodegradation curve based on disintegration
    print(f"\nGenerating quartic biodegradation curve...")
    
    biodegradation_df = generate_cubic_biodegradation_curve(
        disintegration_df,
        t0,
        max_L,
        days=400,
        save_dir=output_dir,
        save_csv=True,
        save_plot=True
    )
    
    print(f"Cubic biodegradation curve generated with {len(biodegradation_df)} data points")
    
    # Display some key values
    print(f"\nKey values from cubic curve:")
    print(f"  At day 0: {biodegradation_df[biodegradation_df['day'] == 0]['biodegradation'].iloc[0]:.2f}%")
    print(f"  At day {t0}: {biodegradation_df[biodegradation_df['day'] == t0]['biodegradation'].iloc[0]:.2f}%")
    print(f"  At day {t0+20}: {biodegradation_df[biodegradation_df['day'] == t0+20]['biodegradation'].iloc[0]:.2f}%")
    print(f"  At day 400: {biodegradation_df[biodegradation_df['day'] == 400]['biodegradation'].iloc[0]:.2f}%")
    
    print(f"\nTest completed! Check {output_dir}/ for generated files:")
    print(f"  - sigmoid_disintegration_curves.csv")
    print(f"  - sigmoid_disintegration_curves.png")
    print(f"  - cubic_biodegradation_curves.csv")
    print(f"  - cubic_vs_sigmoid_comparison.png")

if __name__ == "__main__":
    test_cubic_biodegradation()
