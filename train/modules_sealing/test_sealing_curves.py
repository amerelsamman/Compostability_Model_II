#!/usr/bin/env python3
"""
Test script for sealing profile generation.
Demonstrates the usage of the sealing curve generator.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add the parent directory to the path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules_sealing.curve_generator import generate_sealing_profile, SealingCurveGenerator
from modules_sealing.utils import calculate_boundary_points

def test_sealing_curve_generation():
    """Test the sealing curve generation with sample data."""
    
    # Sample polymer blend data
    polymers = [
        {
            'melt temperature': 120.0,  # Melting temperature
            'property': 15.5,           # Adhesion strength (N/15mm)
            'degradation temperature': 250.0,  # Degradation temperature
            'grade': 'PLA_4032D'
        },
        {
            'melt temperature': 180.0,  # Melting temperature  
            'property': 8.2,            # Adhesion strength (N/15mm)
            'degradation temperature': 220.0,  # Degradation temperature
            'grade': 'PHBV_Y1000P'
        }
    ]
    
    compositions = [0.6, 0.4]  # 60% PLA, 40% PHBV
    predicted_adhesion_strength = 12.3  # ML-predicted adhesion strength
    
    print("Testing Sealing Profile Generation")
    print("=" * 50)
    print(f"Polymers: {[p['grade'] for p in polymers]}")
    print(f"Compositions: {compositions}")
    print(f"Predicted adhesion strength: {predicted_adhesion_strength} N/15mm")
    print()
    
    # Test boundary point calculation
    print("Boundary Points:")
    boundary_points = calculate_boundary_points(polymers, compositions, predicted_adhesion_strength)
    for name, (temp, strength) in boundary_points.items():
        print(f"  {name}: ({temp:.1f}°C, {strength:.1f} N/15mm)")
    print()
    
    # Test curve generation
    print("Generating sealing profile...")
    result = generate_sealing_profile(
        polymers=polymers,
        compositions=compositions,
        predicted_adhesion_strength=predicted_adhesion_strength,
        temperature_range=(0, 250),
        num_points=50,
        save_csv=True,
        save_plot=True,
        save_dir='test_results/sealing_profiles',
        blend_name='test_blend'
    )
    
    if result['is_valid']:
        print("✅ Sealing profile generated successfully!")
        print(f"   Temperature range: {result['temperatures'].min():.1f}°C to {result['temperatures'].max():.1f}°C")
        print(f"   Strength range: {result['strengths'].min():.1f} to {result['strengths'].max():.1f} N/15mm")
        print(f"   Curve data shape: {result['curve_data'].shape}")
    else:
        print("❌ Sealing profile validation failed!")
    
    return result

def test_multiple_blends():
    """Test sealing curve generation for multiple blends."""
    
    # Different blend compositions
    test_cases = [
        {
            'name': 'PLA_dominant',
            'polymers': [
                {'melt temperature': 120.0, 'property': 15.5, 'degradation temperature': 250.0, 'grade': 'PLA_4032D'},
                {'melt temperature': 180.0, 'property': 8.2, 'degradation temperature': 220.0, 'grade': 'PHBV_Y1000P'}
            ],
            'compositions': [0.8, 0.2],
            'predicted_strength': 14.1
        },
        {
            'name': 'PHBV_dominant', 
            'polymers': [
                {'melt temperature': 120.0, 'property': 15.5, 'degradation temperature': 250.0, 'grade': 'PLA_4032D'},
                {'melt temperature': 180.0, 'property': 8.2, 'degradation temperature': 220.0, 'grade': 'PHBV_Y1000P'}
            ],
            'compositions': [0.2, 0.8],
            'predicted_strength': 9.8
        },
        {
            'name': 'balanced_blend',
            'polymers': [
                {'melt temperature': 120.0, 'property': 15.5, 'degradation temperature': 250.0, 'grade': 'PLA_4032D'},
                {'melt temperature': 180.0, 'property': 8.2, 'degradation temperature': 220.0, 'grade': 'PHBV_Y1000P'}
            ],
            'compositions': [0.5, 0.5],
            'predicted_strength': 11.9
        }
    ]
    
    print("\nTesting Multiple Blends")
    print("=" * 50)
    
    for case in test_cases:
        print(f"\nGenerating profile for {case['name']}...")
        result = generate_sealing_profile(
            polymers=case['polymers'],
            compositions=case['compositions'],
            predicted_adhesion_strength=case['predicted_strength'],
            temperature_range=(0, 250),
            num_points=30,
            save_csv=False,
            save_plot=True,
            save_dir='test_results/sealing_profiles',
            blend_name=case['name']
        )
        
        if result['is_valid']:
            print(f"  ✅ {case['name']}: Valid curve generated")
        else:
            print(f"  ❌ {case['name']}: Curve validation failed")

if __name__ == "__main__":
    # Run tests
    test_sealing_curve_generation()
    test_multiple_blends()
    
    print("\n" + "=" * 50)
    print("Test completed! Check 'test_results/sealing_profiles/' for outputs.")
