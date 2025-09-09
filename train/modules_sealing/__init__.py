#!/usr/bin/env python3
"""
Modules package for sealing profile generation functionality.
Contains the sealing curve generation components for polymer blend adhesion.
"""

from .curve_generator import generate_sealing_profile, SealingCurveGenerator
from .utils import calculate_boundary_points, validate_sealing_curve

__all__ = [
    'generate_sealing_profile',
    'SealingCurveGenerator', 
    'calculate_boundary_points',
    'validate_sealing_curve'
]