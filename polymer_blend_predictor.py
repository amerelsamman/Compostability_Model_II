#!/usr/bin/env python3
"""
Polymer Blend Property Predictor
A simple interface for predicting polymer blend properties.

Usage:
    from polymer_blend_predictor import predict_property
    
    # Predict WVTR
    result = predict_property(
        polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
        property_name="wvtr",
        temperature=25,
        rh=60,
        thickness=100
    )
    
    # Predict Tensile Strength
    result = predict_property(
        polymers=[("PLA", "4032D", 0.7), ("PCL", "Capa 6500", 0.3)],
        property_name="ts",
        thickness=50
    )
"""

import os
import sys
import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

# Add the current directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.input_parser import load_and_validate_material_dictionary
from modules.prediction_engine import predict_single_property
from modules.prediction_utils import PROPERTY_CONFIGS

# Configure logging to be less verbose by default
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def predict_property(
    polymers: List[Tuple[str, str, float]],
    property_name: str,
    temperature: Optional[float] = None,
    rh: Optional[float] = None,
    thickness: Optional[float] = None,
    include_errors: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Predict a specific property for a polymer blend.
    
    Args:
        polymers: List of tuples (material_name, grade, volume_fraction)
                 Example: [("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)]
        property_name: Property to predict. Options: "wvtr", "ts", "eab", "cobb"
        temperature: Temperature in Celsius (required for WVTR)
        rh: Relative humidity in percent (required for WVTR)
        thickness: Thickness in micrometers (required for WVTR, TS, EAB)
        include_errors: Whether to include error bounds in the result
        verbose: Whether to show detailed logging
    
    Returns:
        Dictionary containing:
        - 'success': bool - Whether prediction was successful
        - 'prediction': float - Predicted property value
        - 'property_name': str - Name of the property
        - 'unit': str - Unit of the property
        - 'error_message': str - Error message if prediction failed
        - 'error_bounds': dict - Error bounds if include_errors=True
    
    Example:
        >>> result = predict_property(
        ...     polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
        ...     property_name="wvtr",
        ...     temperature=25,
        ...     rh=60,
        ...     thickness=100
        ... )
        >>> print(f"WVTR: {result['prediction']:.2f} {result['unit']}")
    """
    
    # Configure logging level
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Validate inputs
    try:
        # Validate property name
        if property_name.lower() not in PROPERTY_CONFIGS:
            return {
                'success': False,
                'error_message': f"Invalid property name '{property_name}'. Must be one of: {list(PROPERTY_CONFIGS.keys())}"
            }
        
        property_name = property_name.lower()
        config = PROPERTY_CONFIGS[property_name]
        
        # Validate polymers input
        if not polymers or not isinstance(polymers, list):
            return {
                'success': False,
                'error_message': "Polymers must be a non-empty list of tuples"
            }
        
        # Validate volume fractions
        total_fraction = sum(vol_frac for _, _, vol_frac in polymers)
        if not np.isclose(total_fraction, 1.0, atol=1e-5):
            return {
                'success': False,
                'error_message': f"Volume fractions must sum to 1.0, got {total_fraction:.4f}"
            }
        
        # Validate required environmental parameters
        available_env_params = {}
        
        if temperature is not None:
            available_env_params['Temperature (C)'] = temperature
        if rh is not None:
            available_env_params['RH (%)'] = rh
        if thickness is not None:
            available_env_params['Thickness (um)'] = thickness
        
        # Check required parameters for each property
        required_params = config['env_params']
        for param in required_params:
            param_name = param.split(' (')[0].lower()  # Extract parameter name without unit
            if param_name == 'temperature' and temperature is None:
                return {
                    'success': False,
                    'error_message': f"Property '{property_name}' requires temperature parameter"
                }
            elif param_name == 'rh' and rh is None:
                return {
                    'success': False,
                    'error_message': f"Property '{property_name}' requires rh (relative humidity) parameter"
                }
            elif param_name == 'thickness' and thickness is None:
                return {
                    'success': False,
                    'error_message': f"Property '{property_name}' requires thickness parameter"
                }
        
        # Load material dictionary
        material_dict = load_and_validate_material_dictionary()
        if material_dict is None:
            return {
                'success': False,
                'error_message': "Failed to load material dictionary. Make sure 'material-smiles-dictionary.csv' exists."
            }
        
        # Validate that all polymers exist in the dictionary
        for material, grade, vol_frac in polymers:
            key = (material, grade)
            if key not in material_dict:
                available_materials = list(set(mat for mat, _ in material_dict.keys()))
                return {
                    'success': False,
                    'error_message': f"Material/Grade combination '{material} {grade}' not found in dictionary. Available materials: {available_materials}"
                }
        
        # Make prediction
        result = predict_single_property(
            property_type=property_name,
            polymers=polymers,
            available_env_params=available_env_params,
            material_dict=material_dict,
            include_errors=include_errors
        )
        
        if result is None:
            return {
                'success': False,
                'error_message': "Prediction failed. Check logs for details."
            }
        
        # Format response
        response = {
            'success': True,
            'prediction': result['prediction'],
            'property_name': result['name'],
            'unit': result['unit'],
            'polymers': polymers,
            'environmental_params': available_env_params
        }
        
        if include_errors and 'error_bounds' in result:
            response['error_bounds'] = result['error_bounds']
        
        return response
        
    except Exception as e:
        return {
            'success': False,
            'error_message': f"Unexpected error: {str(e)}"
        }

def predict_all_properties(
    polymers: List[Tuple[str, str, float]],
    temperature: Optional[float] = None,
    rh: Optional[float] = None,
    thickness: Optional[float] = None,
    include_errors: bool = False,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Predict all properties for a polymer blend.
    
    Args:
        polymers: List of tuples (material_name, grade, volume_fraction)
        temperature: Temperature in Celsius (for WVTR)
        rh: Relative humidity in percent (for WVTR)
        thickness: Thickness in micrometers (for WVTR, TS, EAB)
        include_errors: Whether to include error bounds
        verbose: Whether to show detailed logging
    
    Returns:
        Dictionary with results for each property
    """
    
    results = {}
    
    for prop_name in PROPERTY_CONFIGS.keys():
        result = predict_property(
            polymers=polymers,
            property_name=prop_name,
            temperature=temperature,
            rh=rh,
            thickness=thickness,
            include_errors=include_errors,
            verbose=verbose
        )
        results[prop_name] = result
    
    return results

def get_available_materials() -> List[Tuple[str, str]]:
    """
    Get list of available materials and grades.
    
    Returns:
        List of tuples (material_name, grade)
    """
    try:
        material_dict = load_and_validate_material_dictionary()
        if material_dict is None:
            return []
        
        return list(material_dict.keys())
    except Exception:
        return []

def get_property_info() -> Dict[str, Dict[str, Any]]:
    """
    Get information about available properties.
    
    Returns:
        Dictionary with property information
    """
    return PROPERTY_CONFIGS

# Example usage and testing
if __name__ == "__main__":
    print("=== Polymer Blend Property Predictor ===")
    
    # Example 1: Predict WVTR for PLA/PBAT blend
    print("\n1. Predicting WVTR for PLA/PBAT blend:")
    result = predict_property(
        polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
        property_name="wvtr",
        temperature=25,
        rh=60,
        thickness=100,
        verbose=True
    )
    
    if result['success']:
        print(f"   {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print(f"   Error: {result['error_message']}")
    
    # Example 2: Predict Tensile Strength for PLA/PCL blend
    print("\n2. Predicting Tensile Strength for PLA/PCL blend:")
    result = predict_property(
        polymers=[("PLA", "4032D", 0.7), ("PCL", "Capa 6500", 0.3)],
        property_name="ts",
        thickness=50,
        verbose=True
    )
    
    if result['success']:
        print(f"   {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print(f"   Error: {result['error_message']}")
    
    # Example 3: Predict all properties
    print("\n3. Predicting all properties for PLA/PBAT blend:")
    all_results = predict_all_properties(
        polymers=[("PLA", "4032D", 0.6), ("PBAT", "Ecoworld", 0.4)],
        temperature=23,
        rh=50,
        thickness=75,
        verbose=True
    )
    
    for prop_name, result in all_results.items():
        if result['success']:
            print(f"   {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
        else:
            print(f"   {prop_name.upper()}: Error - {result['error_message']}")
    
    # Example 4: Show available materials
    print("\n4. Available materials:")
    materials = get_available_materials()
    for material, grade in materials[:10]:  # Show first 10
        print(f"   {material}, {grade}")
    print(f"   ... and {len(materials) - 10} more") 