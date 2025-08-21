#!/usr/bin/env python3
"""
Input parser for polymer blend property prediction.
"""

import sys
import logging
from modules.prediction_utils import load_material_dictionary, parse_command_line_input

# Set up logging
logger = logging.getLogger(__name__)

def parse_environmental_parameters():
    """
    Parse environmental parameters from command line arguments.
    
    Returns:
        dict of environmental parameters
    """
    available_env_params = {}
    for arg in sys.argv[3:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            try:
                # Map key=value parameters to expected column names
                key = key.strip().lower()
                if key in ['temperature', 'temp', 't']:
                    available_env_params['Temperature (C)'] = float(value.strip())
                elif key in ['rh', 'humidity', 'relative_humidity']:
                    available_env_params['RH (%)'] = float(value.strip())
                elif key in ['thickness', 'thick', 't']:
                    available_env_params['Thickness (um)'] = float(value.strip())
                elif key in ['sealing_temperature', 'sealing_temp', 'st']:
                    available_env_params['Sealing Temperature (C)'] = float(value.strip())
                else:
                    logger.warning(f"Unknown environmental parameter: {key}")
            except ValueError:
                logger.warning(f"Invalid environmental parameter: {arg}")
    
    return available_env_params

def validate_input():
    """
    Validate command line input and return parsed arguments.
    
    Returns:
        tuple of (mode, polymer_input, available_env_params) or (None, None, None) if invalid
    """
    if len(sys.argv) < 3:
        print("Usage: python predict_unified_blend.py <mode> <polymer_input> [env_params...]")
        print("Modes: wvtr, ts, eab, cobb, all")
        print("Example: python predict_unified_blend.py all 'PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5'")
        print("Example with env params: python predict_unified_blend.py all 'PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5' temperature=25 rh=60 thickness=100")
        return None, None, None
    
    mode = sys.argv[1].lower()
    polymer_input = sys.argv[2]
    
    # Parse environmental parameters if provided
    available_env_params = parse_environmental_parameters()
    
    return mode, polymer_input, available_env_params

def load_and_validate_material_dictionary():
    """
    Load and validate the material dictionary.
    
    Returns:
        material dictionary or None if failed
    """
    material_dict = load_material_dictionary('material-smiles-dictionary.csv')
    if material_dict is None:
        return None
    return material_dict

def parse_polymer_input(polymer_input, mode):
    """
    Parse polymer input and return polymers and environmental parameters.
    
    Args:
        polymer_input: polymer input string
        mode: prediction mode
    
    Returns:
        tuple of (polymers, parsed_env_params) or (None, None) if failed
    """
    polymers, parsed_env_params = parse_command_line_input(polymer_input, mode)
    if polymers is None:
        return None, None
    return polymers, parsed_env_params 