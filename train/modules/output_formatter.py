#!/usr/bin/env python3
"""
Output formatter for polymer blend property prediction.
"""

import logging
from .prediction_utils import PROPERTY_CONFIGS

# Set up logging
logger = logging.getLogger(__name__)

def print_header(mode, polymer_input):
    """Print the header for the prediction script."""
    print("=== UNIFIED POLYMER BLEND PREDICTION SCRIPT ===")
    print(f"Mode: {'All Properties' if mode == 'all' else 'Single Property'}")
    print(f"Input: {polymer_input}")
    print(f"Dictionary: material-smiles-dictionary.csv")

def print_clean_summary(results):
    """Print a clean, bullet-point summary of all property predictions."""
    if not results:
        print("❌ No properties could be predicted")
        return
    
    print("\n=== PREDICTION RESULTS ===")
    
    # Define the order we want to display properties
    property_order = ['wvtr', 'cobb', 'ts', 'eab', 'otr', 'adhesion', 'compost']
    
    for prop_type in property_order:
        # Find the result for this property type
        result = None
        for r in results:
            if r['property_type'] == prop_type:
                result = r
                break
        
        if result:
            if prop_type == 'compost':
                print(f"• Max Disintegration - {result['prediction']:.1f}%")
                # Add Max Biodegradation if available (new model)
                if 'max_biodegradation' in result:
                    print(f"• Max Biodegradation - {result['max_biodegradation']:.1f}%")
            elif prop_type == 'adhesion' and 'sealing_temp_pred' in result:
                # Dual property: adhesion strength + sealing temperature
                print(f"• Adhesion Strength - {result['prediction']:.2f} {result['unit']}")
                print(f"• Max Sealing Temperature - {result['sealing_temp_pred']:.1f}°C")
            else:
                print(f"• {result['name']} - {result['prediction']:.2f} {result['unit']}")

def print_all_properties_summary(results):
    """Print summary of all property predictions with error bars."""
    if results:
        print(f"\n=== ALL PROPERTIES PREDICTION SUMMARY WITH UNCERTAINTY ===")
        for result in results:
            if result['property_type'] == 'compost':
                # Special handling for compostability results
                print(f"{result['name']}: {result['prediction']:.1f} {result['unit']}")
                print(f"  90-Day Disintegration: {result['day_90_disintegration']:.1f}%")
                print(f"  Home Compostable: {'✅ Yes' if result['is_home_compostable'] else '❌ No'}")
            elif 'error_bounds' in result:
                bounds = result['error_bounds']
                print(f"{result['name']}: {bounds['prediction']:.2f} ± {bounds['model_error']:.2f} {result['unit']}")
                print(f"  Range: [{bounds['lower_bound']:.2f}, {bounds['upper_bound']:.2f}] {result['unit']}")
                print(f"  Experimental Std: {bounds['experimental_std']:.2f} {result['unit']}")
            else:
                print(f"{result['name']}: {result['prediction']:.2f} {result['unit']} (no error data)")
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
    """Print results for single property prediction with error bars."""
    if result:
        config = PROPERTY_CONFIGS[result['property_type']]
        
        # Print error results if available
        if 'error_bounds' in result and 'error_calculator' in result:
            error_output = result['error_calculator'].format_error_results(
                result['property_type'], 
                result['error_bounds'], 
                config['name'], 
                config['unit']
            )
            print(error_output)
        else:
            # Fallback to simple output if no error data
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