"""
Optimization Modules
Modular components for blend optimization system
"""

# Import all functions to maintain exact same interface
from .data_loader import (
    load_validation_data,
    load_material_families,
    load_existing_ki_values
)

from .family_manager import (
    extract_polymer_families_from_blend,
    get_polymer_family_groups,
    get_family_group_for_polymer,
    get_ki_overrides_from_vector
)

from .simulation_engine import (
    simulate_validation_blend,
    calculate_validation_mae,
    calculate_blend_specific_gradients
)

from .optimization_core import (
    build_ki_vector,
    calculate_blend_learning_rates,
    optimize_blend_ki_values,
    update_compatibility_file,
    optimize_family_group_ki_values,
    update_family_group_compatibility_file
)

from .visualization import (
    create_optimization_plots,
    create_validation_performance_plots,
    create_testing_performance_plot
)

from .results_handler import (
    evaluate_performance_on_dataset,
    save_detailed_results_csv
)

__all__ = [
    # Data loading
    'load_validation_data',
    'load_material_families', 
    'load_existing_ki_values',
    
    # Family management
    'extract_polymer_families_from_blend',
    'get_polymer_family_groups',
    'get_family_group_for_polymer',
    'get_ki_overrides_from_vector',
    
    # Simulation
    'simulate_validation_blend',
    'calculate_validation_mae',
    'calculate_blend_specific_gradients',
    
    # Optimization core
    'build_ki_vector',
    'calculate_blend_learning_rates',
    'optimize_blend_ki_values',
    'update_compatibility_file',
    'optimize_family_group_ki_values',
    'update_family_group_compatibility_file',
    
    # Visualization
    'create_optimization_plots',
    'create_validation_performance_plots',
    'create_testing_performance_plot',
    
    # Results handling
    'evaluate_performance_on_dataset',
    'save_detailed_results_csv'
]
