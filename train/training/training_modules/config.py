"""
Configuration module for unified training pipeline.
EXACTLY matches the original training_config.py functionality.
"""

from dataclasses import dataclass
from typing import List

@dataclass
class PropertyConfig:
    """Configuration for a single property's training - EXACTLY matching original"""
    name: str
    target_columns: List[str]  # List of target column names
    is_dual_property: bool      # True if property has multiple targets (TS, EAB, EOL)
    default_last_n_training: int  # Default number of last blends to put in training
    default_last_n_testing: int   # Default number of last blends to put in testing
    oversampling_factor: int      # How many times to repeat last N blends (0 = no oversampling)
    log_offset: float = 0.0      # Offset for log transformation (e.g., 1e-10 for Cobb)
    remove_zero_targets: bool = True  # Whether to remove rows with target == 0
    handle_nan_targets: bool = False  # Whether to handle NaN/infinite targets (WVTR/OTR style)

# Property configurations - EXACTLY matching current behavior
PROPERTY_CONFIGS = {
    'ts': PropertyConfig(
        name='Tensile Strength',
        target_columns=['property1', 'property2'],  # MD and TD
        is_dual_property=True,
        default_last_n_training=4,
        default_last_n_testing=0,
        oversampling_factor=10,  # 10x oversampling of last 4 blends
        log_offset=0.0,
        remove_zero_targets=False,
        handle_nan_targets=False
    ),
    
    'cobb': PropertyConfig(
        name='Cobb Angle',
        target_columns=['property'],
        is_dual_property=False,
        default_last_n_training=10,
        default_last_n_testing=0,
        oversampling_factor=0,  # No oversampling
        log_offset=1e-10,
        remove_zero_targets=True,
        handle_nan_targets=False
    ),
    
    'wvtr': PropertyConfig(
        name='Water Vapor Transmission Rate',
        target_columns=['property'],
        is_dual_property=False,
        default_last_n_training=21,
        default_last_n_testing=0,
        oversampling_factor=0,  # No oversampling
        log_offset=0.0,
        remove_zero_targets=False,
        handle_nan_targets=True
    ),
    
    'otr': PropertyConfig(
        name='Oxygen Transmission Rate',
        target_columns=['property'],
        is_dual_property=False,
        default_last_n_training=0,  # No last N in training
        default_last_n_testing=2,   # Last 2 in testing
        oversampling_factor=0,  # No oversampling
        log_offset=0.0,
        remove_zero_targets=False,
        handle_nan_targets=True
    ),
    
    'adhesion': PropertyConfig(
        name='Adhesion',
        target_columns=['property'],  # Single property: sealing strength (adhesion strength)
        is_dual_property=False,
        default_last_n_training=0,
        default_last_n_testing=5,  # Last 5 in testing
        oversampling_factor=0,  # No oversampling
        log_offset=1e-10,
        remove_zero_targets=True,
        handle_nan_targets=False
    ),
    
    'eab': PropertyConfig(
        name='Elongation at Break',
        target_columns=['property1', 'property2'],  # EAB1 and EAB2
        is_dual_property=True,
        default_last_n_training=4,
        default_last_n_testing=0,
        oversampling_factor=2,  # 2x oversampling of last 4 blends
        log_offset=0.0,
        remove_zero_targets=False,
        handle_nan_targets=False
    ),
    
    'eol': PropertyConfig(
        name='Compostability (End of Life)',
        target_columns=['max_L', 't0'],  # max_L and t0
        is_dual_property=True,
        default_last_n_training=4,
        default_last_n_testing=0,
        oversampling_factor=10,  # 10x oversampling of last 4 blends
        log_offset=0.0,
        remove_zero_targets=False,
        handle_nan_targets=False
    )
}

def get_property_config(property_name: str) -> PropertyConfig:
    """Get configuration for a specific property - EXACTLY matching original"""
    if property_name not in PROPERTY_CONFIGS:
        raise ValueError(f"Unknown property: {property_name}. Available: {list(PROPERTY_CONFIGS.keys())}")
    return PROPERTY_CONFIGS[property_name]

def get_available_properties() -> List[str]:
    """Get list of available properties - EXACTLY matching original"""
    return list(PROPERTY_CONFIGS.keys())
