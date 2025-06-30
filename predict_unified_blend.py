#!/usr/bin/env python3
"""
Unified Polymer Blend Prediction Script
Predicts multiple properties for polymer blends using transfer learning models.

Usage:
  # All properties
  python predict_unified_blend.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  
  # Single property
  python predict_unified_blend.py wvtr "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  
  # With environmental parameters
  python predict_unified_blend.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5" temperature=25 rh=60 thickness=100

Format:
  All/WVTR: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Temperature, RH, Thickness"
  TS/EAB: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Thickness"
  Cobb: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..."
"""

import sys
import logging
from modules.input_parser import validate_input, load_and_validate_material_dictionary, parse_polymer_input
from modules.output_formatter import (
    print_header, print_all_properties_summary, print_input_summary,
    print_single_property_header, print_single_property_results, print_all_properties_header
)
from modules.prediction_engine import predict_single_property

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function to run the unified blend prediction."""
    # Validate input
    mode, polymer_input, available_env_params = validate_input()
    if mode is None:
        sys.exit(1)
    
    # Print header
    print_header(mode, polymer_input)
    
    # Load material dictionary
    material_dict = load_and_validate_material_dictionary()
    if material_dict is None:
        return
    
    # Parse polymer input
    polymers, parsed_env_params = parse_polymer_input(polymer_input, mode)
    if polymers is None:
        return
    
    # Merge environmental parameters (command line takes precedence)
    if parsed_env_params:
        available_env_params.update(parsed_env_params)
    
    if mode == 'all':
        # Predict all properties
        print_all_properties_header()
        results = []
        
        for prop_type in ['wvtr', 'ts', 'eab', 'cobb']:
            result = predict_single_property(prop_type, polymers, available_env_params, material_dict)
            if result:
                results.append(result)
        
        # Print summary
        print_all_properties_summary(results)
        print_input_summary(polymers, available_env_params)
        
        return results
        
    else:
        # Single property mode
        print_single_property_header(mode)
        
        result = predict_single_property(mode, polymers, available_env_params, material_dict)
        print_single_property_results(result, polymers)
        
        if result:
            return result['prediction']
        else:
            return None

if __name__ == "__main__":
    main() 