#!/usr/bin/env python3
"""
Output formatter for polymer blend property prediction.
"""

import logging
from modules.prediction_utils import PROPERTY_CONFIGS

# Set up logging
logger = logging.getLogger(__name__)

def print_header(mode, polymer_input):
    """Print the header for the prediction script."""
    print("=== UNIFIED POLYMER BLEND PREDICTION SCRIPT ===")
    print(f"Mode: {'All Properties' if mode == 'all' else 'Single Property'}")
    print(f"Input: {polymer_input}")
    print(f"Dictionary: material-smiles-dictionary.csv")

def print_all_properties_summary(results):
    """Print summary of all property predictions."""
    if results:
        print(f"\n=== ALL PROPERTIES PREDICTION SUMMARY ===")
        for result in results:
            print(f"{result['name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print("❌ No properties could be predicted")

def print_input_summary(polymers, available_env_params):
    """Print summary of input parameters."""
    print(f"\n=== INPUT SUMMARY ===")
    print(f"Number of polymers: {len(polymers)}")
    for i, (material, grade, vol_fraction) in enumerate(polymers, 1):
        print(f"  Polymer {i}: {material} {grade} ({vol_fraction:.2f})")
    
    for param, value in available_env_params.items():
        print(f"{param}: {value}")

def print_single_property_header(mode):
    """Print header for single property prediction."""
    config = PROPERTY_CONFIGS[mode]
    print(f"Property: {config['name']}")
    print(f"Unit: {config['unit']}")

def print_single_property_results(result, polymers):
    """Print results for single property prediction."""
    if result:
        config = PROPERTY_CONFIGS[result['property_type']]
        print(f"\n=== {config['name'].upper()} PREDICTION RESULTS ===")
        print(f"Predicted {config['name']}: {result['prediction']:.2f} {config['unit']}")
        
        print(f"\n=== INPUT SUMMARY ===")
        print(f"Number of polymers: {len(polymers)}")
        for i, (material, grade, vol_fraction) in enumerate(polymers, 1):
            print(f"  Polymer {i}: {material} {grade} ({vol_fraction:.2f})")
        
        for param, value in result['env_params'].items():
            print(f"{param}: {value}")
    else:
        config = PROPERTY_CONFIGS[result['property_type']] if result else None
        if config:
            print(f"❌ Failed to predict {config['name']}")
        else:
            print("❌ Failed to predict property")

def print_all_properties_header():
    """Print header for all properties prediction."""
    print(f"\n=== PREDICTING ALL PROPERTIES ===") 