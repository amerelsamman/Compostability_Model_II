#!/usr/bin/env python3
"""
Utilities module for sealing profile generation.
Contains helper functions for boundary point calculations and curve validation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

def calculate_boundary_points(polymers: List[Dict], compositions: List[float], 
                            predicted_adhesion_strength: float) -> Dict[str, Tuple[float, float]]:
    """
    Calculate the 4 boundary points for sealing profile generation.
    
    Args:
        polymers: List of polymer dictionaries with properties
        compositions: Volume fractions of each polymer
        predicted_adhesion_strength: ML-predicted adhesion strength (primary method for Point 3)
    
    Returns:
        Dictionary with boundary point names and (temperature, strength) tuples
        For neat polymers: Uses midpoint between melt and degradation temperature
        For blends: Uses ML model prediction as primary, rule of mixtures as fallback for Point 3
    """
    # Check if this is a neat polymer system (single polymer with 100% composition)
    is_neat_polymer = (len(polymers) == 1 and len([c for c in compositions if c > 0]) == 1 and 
                      any(c > 0.99 for c in compositions))
    
    # Point 1: (T1 - 20°C, 1) - Initial sealing at low temperature
    t1 = min(p['melt temperature'] for p in polymers)  # Lowest melting temperature
    point1 = (t1 - 20, 1.0)
    
    # Point 2: (T1, property1 * vol_fraction1) - First polymer's weighted sealing strength
    first_polymer_adhesion = polymers[0]['property'] * compositions[0]  # property weighted by composition
    point2 = (t1, first_polymer_adhesion)
    
    if is_neat_polymer:
        # Specialized rule for neat polymers: Use midpoint between melt and degradation temperature
        t_degradation = min(p['degradation temperature'] for p in polymers)
        
        # Calculate midpoint temperature between melt and degradation
        t_midpoint = (t1 + t_degradation) / 2
        
        # Calculate plateau strength (20% increase from base property)
        plateau_strength = first_polymer_adhesion * 1.2  # 20% increase
        
        point3 = (t_midpoint, plateau_strength)
        print(f"Neat polymer detected: Using midpoint rule - {t_midpoint:.1f}°C, {plateau_strength:.1f} N/15mm")
        
    else:
        # Blend system: Use ML model as primary, rule of mixtures as fallback
        # Point 3: (T_blend_ROM, ml_predicted_strength) - Blend's ML-predicted sealing strength
        # Calculate rule of mixtures temperature
        t_blend_rom = sum(comp * p['melt temperature'] for comp, p in zip(compositions, polymers))
        
        # Calculate rule of mixtures strength (backup method)
        rom_strength = sum(comp * p['property'] for comp, p in zip(compositions, polymers))
        
        # Use ML prediction as primary, rule of mixtures as fallback if ML is invalid
        if predicted_adhesion_strength > 0 and not np.isnan(predicted_adhesion_strength):
            point3_strength = predicted_adhesion_strength
            strength_source = "ml_model"
            print(f"Using ML model prediction for Boundary Point 3: {point3_strength:.1f} N/15mm")
        else:
            point3_strength = rom_strength
            strength_source = "rule_of_mixtures_fallback"
            print(f"⚠️ Using rule of mixtures as fallback for Boundary Point 3 (ML was invalid: {predicted_adhesion_strength})")
        
        point3 = (t_blend_rom, point3_strength)
    
    # Point 4: (T_degradation, 0) - Degradation point (use actual degradation temperature)
    # Use the lowest degradation temperature in the blend
    t_degradation = min(p['degradation temperature'] for p in polymers)
    point4 = (t_degradation, 0.0)
    
    return {
        'initial_sealing': point1,
        'first_polymer_max': point2, 
        'blend_predicted': point3,
        'degradation': point4
    }

def validate_sealing_curve(temperatures: np.ndarray, strengths: np.ndarray) -> bool:
    """
    Validate that the generated sealing curve meets physical constraints.
    
    Args:
        temperatures: Array of temperature values
        strengths: Array of sealing strength values
    
    Returns:
        Boolean indicating if curve is valid
    """
    # Check for reasonable strength values (0 to 100 N/15mm)
    strength_bounds = np.all((strengths >= 0) & (strengths <= 100))
    
    # Check for reasonable temperature range (0 to 300°C)
    temp_bounds = np.all((temperatures >= 0) & (temperatures <= 300))
    
    # Check for no NaN or infinite values
    no_nan = np.all(np.isfinite(strengths)) and np.all(np.isfinite(temperatures))
    
    # Check that curve has some variation (not all zeros)
    has_variation = np.max(strengths) > 0.1
    
    return strength_bounds and temp_bounds and no_nan and has_variation

def calculate_rule_of_mixtures_temperature(polymers: List[Dict], compositions: List[float]) -> float:
    """
    Calculate rule of mixtures temperature for the blend.
    
    Args:
        polymers: List of polymer dictionaries
        compositions: Volume fractions of each polymer
    
    Returns:
        Rule of mixtures temperature
    """
    return sum(comp * p['melt temperature'] for comp, p in zip(compositions, polymers))

def get_polymer_properties(polymers: List[Dict], compositions: List[float]) -> Dict[str, float]:
    """
    Extract key properties from polymer blend for sealing profile generation.
    
    Args:
        polymers: List of polymer dictionaries
        compositions: Volume fractions of each polymer
    
    Returns:
        Dictionary with key blend properties
    """
    return {
        't1': min(p['melt temperature'] for p in polymers),
        't_blend_rom': calculate_rule_of_mixtures_temperature(polymers, compositions),
        'first_polymer_adhesion': polymers[0]['property'],
        't_degradation': min(p['degradation temperature'] for p in polymers)
    }
