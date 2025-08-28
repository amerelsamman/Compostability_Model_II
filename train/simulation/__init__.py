"""
Simulation package for polymer blend property augmentation.
"""

__version__ = "1.0.0"
__author__ = "BlendModels Team"

# Import common functions for easy access
from .simulation_common import (
    load_material_smiles_dict,
    generate_random_composition,
    get_random_polymer_combination,
    create_blend_row_base,
    run_augmentation_loop,
    combine_with_original_data,
    create_ml_dataset,
    save_augmented_data,
    set_random_seeds
)

__all__ = [
    'load_material_smiles_dict',
    'generate_random_composition',
    'get_random_polymer_combination',
    'create_blend_row_base',
    'run_augmentation_loop',
    'combine_with_original_data',
    'create_ml_dataset',
    'save_augmented_data',
    'set_random_seeds'
]
