#!/usr/bin/env python3
"""
Polymer Blend Database Generator
Generates a comprehensive database of polymer blend property predictions
in the same format as blenddatabase.csv.

Tests various polymer blend combinations and predicts all properties:
wvtr, ts, eab, cobb, otr, seal, compost (disintegration only)

Output format matches blenddatabase.csv with columns:
blend_type, polymer1-4, grade1-4, concentration1-4, thickness_um, 
temperature_c, humidity_percent, wvtr_prediction, wvtr_unit, 
ts_prediction, ts_unit, eab_prediction, eab_unit, cobb_prediction, 
cobb_unit, otr_prediction, otr_unit, seal_prediction, seal_unit,
compost_prediction, compost_unit, compost_day_30, 
compost_day_90, compost_day_180, compost_max, compost_disintegration_time_profile
"""

import os
import sys
import pandas as pd
import numpy as np
from itertools import combinations, product
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
import argparse
import json
import random

# Set global seed for reproducible results
random.seed(42)
np.random.seed(42)
import time

# Import the modules directly instead of using subprocess
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.input_parser import validate_input, load_and_validate_material_dictionary, parse_polymer_input
from modules.prediction_engine import predict_blend_property
from modules.prediction_utils import PROPERTY_CONFIGS

# Import curve generation modules for compostability
try:
    from modules_home.curve_generator import generate_compostability_curves
    CURVE_GENERATION_AVAILABLE = True
except ImportError as e:
    CURVE_GENERATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Property configurations
PROPERTY_CONFIGS = {
    'wvtr': {
        'name': 'WVTR',
        'unit': 'g/m²/day',
        'env_params': ['temperature', 'rh', 'thickness'],
        'default_env': {'temperature': 38, 'rh': 90},  # No thickness default
        'parse_pattern': '• WVTR -',
        'log_scale': True
    },
    'ts': {
        'name': 'Tensile Strength',
        'unit': 'MPa',
        'env_params': ['thickness'],
        'default_env': {},  # No thickness default
        'parse_pattern': '• Tensile Strength -',
        'log_scale': True
    },
    'eab': {
        'name': 'Elongation at Break',
        'unit': '%',
        'env_params': ['thickness'],
        'default_env': {},  # No thickness default
        'parse_pattern': '• Elongation at Break -',
        'log_scale': True
    },
    'cobb': {
        'name': 'Cobb Value',
        'unit': 'g/m²',
        'env_params': [],
        'default_env': {},
        'parse_pattern': '• Cobb Value -',
        'log_scale': True
    },
    'otr': {
        'name': 'Oxygen Transmission Rate',
        'unit': 'cc/m²/day',
        'env_params': ['temperature', 'rh', 'thickness'],
        'default_env': {'temperature': 23, 'rh': 50},  # No thickness default
        'parse_pattern': '• OTR -',
        'log_scale': True
    },
    'compost': {
        'name': 'Max Disintegration',
        'unit': '% disintegration',
        'env_params': ['thickness'],
        'default_env': {},  # No thickness default
        'parse_pattern': '• Max Disintegration -',
        'log_scale': False
    },
    'all': {
        'name': 'All Properties',
        'unit': 'mixed',
        'env_params': ['temperature', 'rh', 'thickness'],
        'default_env': {'temperature': 38, 'rh': 90},  # No thickness default
        'parse_pattern': None,  # Special handling for all properties
        'log_scale': False
    }
}

class PolymerBlendDatabaseGenerator:
    def __init__(self, property_type='wvtr', material_dict_path='material-smiles-dictionary-db.csv', max_materials=22):
        """Initialize the database generator."""
        self.property_type = property_type.lower()
        if self.property_type not in PROPERTY_CONFIGS:
            raise ValueError(f"Unsupported property type: {property_type}. Supported: {list(PROPERTY_CONFIGS.keys())}")
        
        self.config = PROPERTY_CONFIGS[self.property_type]
        self.material_dict_path = material_dict_path
        self.max_materials = max_materials
        self.materials = self._load_materials()
        self.results = []
        self.fixed_env_params = self.config['default_env'].copy()
        
        # Load material dictionary once
        self.material_dict = load_and_validate_material_dictionary(self.material_dict_path)
        if self.material_dict is None:
            raise ValueError("Failed to load material dictionary")
        
    def _load_materials(self) -> List[Tuple[str, str]]:
        """Load available materials from the dictionary."""
        try:
            df = pd.read_csv(self.material_dict_path)
            # Use only the first max_materials molecules
            df = df.head(self.max_materials)
            materials = []
            for _, row in df.iterrows():
                materials.append((row['Material'].strip(), row['Grade'].strip()))
            logger.info(f"Loaded {len(materials)} material-grade combinations (using first {self.max_materials} molecules)")
            return materials
        except Exception as e:
            logger.error(f"Error loading materials: {e}")
            return []
    
    def _group_materials_by_polymer_type(self) -> Dict[str, List[Tuple[str, str]]]:
        """Group materials by polymer type."""
        polymer_groups = {}
        for material, grade in self.materials:
            if material not in polymer_groups:
                polymer_groups[material] = []
            polymer_groups[material].append((material, grade))
        return polymer_groups
    
    def _generate_comprehensive_combinations(self, max_combinations: int = 100, max_polymers: int = 2, full_exploration: bool = False) -> List[List[Tuple[str, str]]]:
        """Generate comprehensive blend combinations ensuring all polymer types are represented."""
        polymer_groups = self._group_materials_by_polymer_type()
        polymer_types = list(polymer_groups.keys())
        
        logger.info(f"Found {len(polymer_types)} unique polymer types: {polymer_types}")
        logger.info(f"Max polymers per blend: {max_polymers}")
        logger.info(f"Full material exploration: {full_exploration}")
        
        combinations_list = []
        
        # Generate combinations for each polymer count up to max_polymers
        for n_polymers in range(1, max_polymers + 1):
            if n_polymers == 1:
                # 1-polymer blends: test all materials
                for poly_type in polymer_types:
                    for material in polymer_groups[poly_type]:
                        if len(combinations_list) < max_combinations:
                            combinations_list.append([material])
            else:
                # Multi-polymer blends
                if full_exploration and n_polymers < 4:
                    # Full exploration: test all material combinations (but limit to 3-polymer blends)
                    combinations_list.extend(self._generate_full_exploration_combinations(
                        polymer_groups, polymer_types, n_polymers, max_combinations - len(combinations_list)
                    ))
                else:
                    # Limited exploration: one material per polymer type (forced for 4+ polymer blends)
                    if n_polymers >= 4:
                        logger.info(f"⚠️ Forcing limited exploration for {n_polymers}-polymer blends to prevent excessive computation")
                    combinations_list.extend(self._generate_limited_exploration_combinations(
                        polymer_groups, polymer_types, n_polymers, max_combinations - len(combinations_list)
                    ))
        
        # Limit to max_combinations
        combinations_list = combinations_list[:max_combinations]
        
        logger.info(f"Generated {len(combinations_list)} comprehensive blend combinations")
        return combinations_list
    
    def _generate_limited_exploration_combinations(self, polymer_groups: Dict, polymer_types: List[str], 
                                                 n_polymers: int, max_remaining: int) -> List[List[Tuple[str, str]]]:
        """Generate combinations with one material per polymer type (current approach)."""
        combinations = []
        
        # Generate all possible n-polymer combinations of polymer types
        from itertools import combinations as itertools_combinations
        polymer_type_combinations = list(itertools_combinations(polymer_types, n_polymers))
        
        for poly_type_combo in polymer_type_combinations:
            if len(combinations) >= max_remaining:
                break
            
            # Take the first material from each polymer type
            material_combo = [polymer_groups[poly_type][0] for poly_type in poly_type_combo]
            combinations.append(material_combo)
        
        return combinations
    
    def _generate_full_exploration_combinations(self, polymer_groups: Dict, polymer_types: List[str], 
                                              n_polymers: int, max_remaining: int) -> List[List[Tuple[str, str]]]:
        """Generate combinations with all materials per polymer type."""
        combinations = []
        
        # Generate all possible n-polymer combinations of polymer types
        from itertools import combinations as itertools_combinations
        polymer_type_combinations = list(itertools_combinations(polymer_types, n_polymers))
        
        for poly_type_combo in polymer_type_combinations:
            if len(combinations) >= max_remaining:
                break
            
            # Get all materials for each polymer type in this combination
            material_lists = [polymer_groups[poly_type] for poly_type in poly_type_combo]
            
            # Generate all possible material combinations
            from itertools import product
            material_combinations = list(product(*material_lists))
            
            for material_combo in material_combinations:
                if len(combinations) >= max_remaining:
                    break
                combinations.append(list(material_combo))
        
        return combinations
    
    def _create_blend_input_string(self, materials: List[Tuple[str, str]], 
                                  blend_type: str = 'equal') -> str:
        """Create input string for predict_unified_blend.py.
        
        For random blends, uses Dirichlet distribution to ensure:
        - All components have meaningful contributions (prevents misclassification)
        - Volume fractions naturally sum to 1.0
        - No component gets arbitrarily small
        """
        if blend_type == 'equal':
            # Equal volume fractions
            vol_fraction = 1.0 / len(materials)
            blend_parts = []
            for material, grade in materials:
                blend_parts.extend([material, grade, str(vol_fraction)])
        elif blend_type == 'random':
            # Generate volume fractions using Dirichlet distribution to ensure proper blend classification
            # alpha > 1 ensures no component gets too small, maintaining meaningful contributions
            # alpha = 2.0 gives balanced distributions, higher values = more uniform, lower = more concentrated
            alpha = 2.0  # concentration parameter - can be adjusted if needed
            fractions = np.random.dirichlet([alpha] * len(materials))
            
            blend_parts = []
            for i, (material, grade) in enumerate(materials):
                blend_parts.extend([material, grade, str(fractions[i])])
            
            # Validate that all components have meaningful contributions (prevents misclassification)
            min_contribution = 0.01  # 1% minimum contribution
            meaningful_components = sum(1 for f in fractions if f >= min_contribution)
            if meaningful_components != len(materials):
                logger.warning(f"Blend has {len(materials)} polymers but only {meaningful_components} meaningful components")
        else:
            raise ValueError(f"Unknown blend type: {blend_type}")
        
        return ', '.join(blend_parts)
    

    
    def generate_database(self, combinations: List[List[Tuple[str, str]]], random_samples: int, 
                         blend_types: List[str] = None) -> pd.DataFrame:
        """Generate comprehensive database in blenddatabase.csv format."""
        if blend_types is None:
            blend_types = ['random']
        
        results = []
        total_tests = len(combinations) * random_samples  # Only random samples now
        
        # Progress tracking setup
        logger.info(f"Starting database generation with {total_tests} total tests...")
        
        current_test = 0
        start_time = time.time()
        
        for materials in combinations:
            for blend_type in blend_types:
                if blend_type == 'random':
                    # Generate multiple random samples
                    for sample_idx in range(random_samples):
                        current_test += 1
                        
                        # Print progress every 1000 tests
                        if current_test % 1000 == 0:
                            elapsed_time = time.time() - start_time
                            avg_time_per_test = elapsed_time / current_test
                            print(f"Completed {current_test} tests (avg {avg_time_per_test:.2f}s per test)")
                        
                        # Create blend input
                        blend_input = self._create_blend_input_string(materials, blend_type)
                        
                        # Run prediction for ALL properties
                        all_properties_result = self._run_all_properties_prediction(blend_input)
                        
                        # Record results in blenddatabase.csv format
                        result = self._format_result_for_database(
                            materials, blend_input, blend_type, all_properties_result
                        )
                        
                        results.append(result)
                        
                        # Silent progress (no logging for individual tests)
        
        # Calculate final timing
        total_time = time.time() - start_time
        avg_time_per_test = total_time / total_tests if total_tests > 0 else 0
        logger.info(f"✅ Completed {total_tests} tests in {total_time:.1f}s (avg {avg_time_per_test:.1f}s per test)")
        
        # Convert to DataFrame with blenddatabase.csv format
        self.results_df = pd.DataFrame(results)
        
        return self.results_df
    
    def _run_all_properties_prediction(self, blend_input: str) -> Dict[str, Any]:
        """Run prediction for all properties (wvtr, ts, eab, cobb, compost)."""
        try:
            # Suppress output from modules
            import contextlib
            import io
            
            # Temporarily suppress logging and stdout
            original_log_level = logging.getLogger().level
            logging.getLogger().setLevel(logging.ERROR)  # Only show errors
            
            # Parse the blend input to get polymers and environmental parameters
            polymers, parsed_env_params = parse_polymer_input(blend_input, 'all')
            if polymers is None:
                return {
                    'success': False,
                    'error': 'Failed to parse blend input'
                }
            
            # Merge environmental parameters (fixed params take precedence)
            available_env_params = parsed_env_params.copy()
            available_env_params.update(self.fixed_env_params)
            
            # Convert to the format expected by the prediction engine
            env_params_dict = {}
            for key, value in available_env_params.items():
                if key.lower() == 'temperature':
                    env_params_dict['Temperature (C)'] = value
                elif key.lower() == 'rh':
                    env_params_dict['RH (%)'] = value
                elif key.lower() == 'thickness':
                    env_params_dict['Thickness (um)'] = value
            
            # Predict all properties with fixed environmental conditions
            all_results = {}
            
            # Get thickness from available_env_params (must be provided)
            thickness = available_env_params.get('thickness')
            if thickness is None:
                raise ValueError("❌ Thickness must be provided - no thickness defaults allowed!")
            
            # Standard properties
            for prop_type in ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']:
                # Set fixed environmental conditions for each property
                if prop_type == 'wvtr':
                    # WVTR: Fixed 38°C, 90% RH, user-specified thickness
                    prop_env_params = {
                        'Temperature (C)': 38,
                        'RH (%)': 90,
                        'Thickness (um)': thickness
                    }
                elif prop_type == 'otr':
                    # OTR: Fixed 23°C, 50% RH, user-specified thickness
                    prop_env_params = {
                        'Temperature (C)': 23,
                        'RH (%)': 50,
                        'Thickness (um)': thickness
                    }
                else:
                    # Other properties: Only thickness (no temperature/RH)
                    prop_env_params = {
                        'Thickness (um)': thickness
                    }
                
                result = predict_blend_property(
                    prop_type, 
                    polymers, 
                    prop_env_params, 
                    self.material_dict,
                    include_errors=False
                )
                
                
                
                if result and 'prediction' in result:
                    # For OTR and WVTR, use unnormalized prediction (actual value at thickness)
                    # For other properties, use the raw prediction
                    if prop_type in ['otr', 'wvtr'] and isinstance(result['prediction'], dict):
                        # Use unnormalized prediction for OTR/WVTR
                        prediction_value = result['prediction'].get('unnormalized_prediction', result['prediction']['prediction'])
                    else:
                        # Use raw prediction for other properties
                        prediction_value = result['prediction']
                    
                    all_results[prop_type] = {
                        'prediction': prediction_value,
                        'unit': result['unit']
                    }
                    # Preserve additional fields for specific properties
                    if prop_type == 'seal' and 'sealing_temp_pred' in result:
                        all_results[prop_type]['sealing_temp_pred'] = result['sealing_temp_pred']
                    # For compostability, also store t0_pred
                    if prop_type == 'compost' and 't0_pred' in result:
                        all_results[prop_type]['t0_pred'] = result['t0_pred']
                else:
                    all_results[prop_type] = {'prediction': None, 'unit': None}
            
            return {
                'success': True,
                'results': all_results,
                'polymers': polymers,
                'env_params': available_env_params
            }
            
            # Restore logging level
            logging.getLogger().setLevel(original_log_level)
                
        except Exception as e:
            # Restore logging level in case of exception
            logging.getLogger().setLevel(original_log_level)
            return {
                'success': False,
                'error': str(e)
            }
    
    def _format_result_for_database(self, materials: List[Tuple[str, str]], blend_input: str, 
                                   blend_type: str, all_properties_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single result to match blenddatabase.csv structure."""
        # Determine blend type
        n_polymers = len(materials)
        if n_polymers == 1:
            blend_type_name = 'homopolymer'
        elif n_polymers == 2:
            blend_type_name = 'binary'
        elif n_polymers == 3:
            blend_type_name = 'ternary'
        elif n_polymers == 4:
            blend_type_name = 'quaternary'
        else:
            blend_type_name = f'{n_polymers}-polymer'
        
        # Parse blend input to get volume fractions
        blend_parts = blend_input.split(', ')
        concentrations = []
        for i in range(2, len(blend_parts), 3):  # Start from index 2 (volume fraction)
            if i < len(blend_parts):
                concentrations.append(float(blend_parts[i]))
        
        # Initialize result with blend information
        result = {
            'blend_type': blend_type_name,
            'polymer1': materials[0][0] if len(materials) > 0 else None,
            'grade1': materials[0][1] if len(materials) > 0 else None,
            'concentration1': round(concentrations[0], 10) if len(concentrations) > 0 else None,
            'polymer2': materials[1][0] if len(materials) > 1 else None,
            'grade2': materials[1][1] if len(materials) > 1 else None,
            'concentration2': round(concentrations[1], 10) if len(concentrations) > 1 else None,
            'polymer3': materials[2][0] if len(materials) > 2 else None,
            'grade3': materials[2][1] if len(materials) > 2 else None,
            'concentration3': round(concentrations[2], 10) if len(concentrations) > 2 else None,
            'polymer4': materials[3][0] if len(materials) > 3 else None,
            'grade4': materials[3][1] if len(materials) > 3 else None,
            'concentration4': round(concentrations[3], 10) if len(concentrations) > 3 else None,
        }
        
        # Add environmental parameters - only thickness is user-specified
        # Temperature and humidity are fixed per property and not shown in output
        env_params = all_properties_result.get('env_params', {})
        result.update({
            'thickness_um': env_params.get('thickness', None),  # Must be provided
            'temperature_c': None,  # Not shown - fixed per property
            'humidity_percent': None  # Not shown - fixed per property
        })
        
        # Add property predictions
        if all_properties_result['success']:
            results = all_properties_result['results']
            
            # Helper function to extract prediction value
            def extract_prediction_value(prediction_obj, property_type=None):
                """Extract the actual prediction value from various formats."""
                if prediction_obj is None:
                    return None
                if isinstance(prediction_obj, (int, float)):
                    return float(prediction_obj)
                if isinstance(prediction_obj, dict) and 'prediction' in prediction_obj:
                    # For OTR and WVTR, prefer unnormalized prediction (actual value at thickness)
                    if property_type in ['otr', 'wvtr'] and 'unnormalized_prediction' in prediction_obj:
                        pred_val = prediction_obj['unnormalized_prediction']
                    else:
                        pred_val = prediction_obj['prediction']
                    
                    if isinstance(pred_val, (int, float)):
                        return float(pred_val)
                    elif hasattr(pred_val, 'item'):  # numpy scalar
                        return float(pred_val.item())
                    else:
                        return float(pred_val)
                if hasattr(prediction_obj, 'item'):  # numpy scalar
                    return float(prediction_obj.item())
                return float(prediction_obj)
            
            # WVTR
            wvtr_pred = extract_prediction_value(results.get('wvtr', {}).get('prediction'), 'wvtr')
            result.update({
                'wvtr_prediction': wvtr_pred,
                'wvtr_unit': results.get('wvtr', {}).get('unit', 'g/m²/day')
            })
            
            # Tensile Strength
            ts_pred = extract_prediction_value(results.get('ts', {}).get('prediction'), 'ts')
            result.update({
                'ts_prediction': ts_pred,
                'ts_unit': results.get('ts', {}).get('unit', 'MPa')
            })
            
            # Elongation at Break
            eab_pred = extract_prediction_value(results.get('eab', {}).get('prediction'), 'eab')
            result.update({
                'eab_prediction': eab_pred,
                'eab_unit': results.get('eab', {}).get('unit', '%')
            })
            
            # Cobb Value
            cobb_pred = extract_prediction_value(results.get('cobb', {}).get('prediction'), 'cobb')
            result.update({
                'cobb_prediction': cobb_pred,
                'cobb_unit': results.get('cobb', {}).get('unit', 'g/m²')
            })
            
            # OTR
            otr_pred = extract_prediction_value(results.get('otr', {}).get('prediction'), 'otr')
            result.update({
                'otr_prediction': otr_pred,
                'otr_unit': results.get('otr', {}).get('unit', 'cc/m²/day')
            })
            
            # Sealing Strength
            seal_result = results.get('seal', {})
            seal_pred = extract_prediction_value(seal_result.get('prediction'), 'seal')
            result.update({
                'seal_prediction': seal_pred,
                'seal_unit': seal_result.get('unit', 'N/15mm'),
                'sealing_temp_pred': extract_prediction_value(seal_result.get('sealing_temp_pred')),
                'sealing_temp_unit': '°C'
            })
            
            # Compostability (disintegration only - biodegradation commented out for now)
            compost = results.get('compost', {})
            max_L_pred = extract_prediction_value(compost.get('prediction'))  # max_L_pred
            t0_pred = extract_prediction_value(compost.get('t0_pred'))  # t0_pred (time to 50% disintegration)
            thickness = result.get('thickness_um', 100) / 1000.0  # Convert to mm
            
            # Generate curves if available (disintegration only)
            if CURVE_GENERATION_AVAILABLE and max_L_pred is not None and t0_pred is not None:
                try:
                    curve_result = generate_compostability_curves(
                        max_L_pred, t0_pred, thickness, 
                        save_csv=False, save_plot=False
                    )
                    
                    if curve_result:
                        disintegration_df = curve_result.get('disintegration_curve')
                        # biodegradation_df = curve_result.get('biodegradation_curve')  # Commented out
                        
                        # Extract specific day values
                        disintegration_day_30 = None
                        disintegration_day_90 = None
                        disintegration_day_180 = None
                        if disintegration_df is not None and not disintegration_df.empty:
                            disintegration_day_30 = float(disintegration_df[disintegration_df['day'] == 30]['disintegration'].iloc[0]) if 30 in disintegration_df['day'].values else None
                            disintegration_day_90 = float(disintegration_df[disintegration_df['day'] == 90]['disintegration'].iloc[0]) if 90 in disintegration_df['day'].values else None
                            disintegration_day_180 = float(disintegration_df[disintegration_df['day'] == 180]['disintegration'].iloc[0]) if 180 in disintegration_df['day'].values else None
                        
                        # biodegradation_day_30 = None  # Commented out
                        # biodegradation_day_90 = None
                        # biodegradation_day_180 = None
                        # if biodegradation_df is not None and not biodegradation_df.empty:
                        #     biodegradation_day_30 = float(biodegradation_df[biodegradation_df['day'] == 30]['biodegradation'].iloc[0]) if 30 in biodegradation_df['day'].values else None
                        #     biodegradation_day_90 = float(biodegradation_df[biodegradation_df['day'] == 90]['biodegradation'].iloc[0]) if 90 in biodegradation_df['day'].values else None
                        #     biodegradation_day_180 = float(biodegradation_df[biodegradation_df['day'] == 180]['biodegradation'].iloc[0]) if 180 in biodegradation_df['day'].values else None
                        
                        # Convert DataFrames to time profile dictionaries for JSON storage
                        # Sample every 15 days to reduce data size
                        disintegration_profile = {}
                        # biodegradation_profile = {}  # Commented out
                        
                        if disintegration_df is not None and not disintegration_df.empty:
                            for _, row in disintegration_df.iterrows():
                                day = int(row['day'])
                                # Only include every 15th day (0, 15, 30, 45, 60, etc.)
                                if day % 15 == 0:
                                    disintegration_profile[day] = float(row['disintegration'])
                        
                        # if biodegradation_df is not None and not biodegradation_df.empty:  # Commented out
                        #     for _, row in biodegradation_df.iterrows():
                        #         day = int(row['day'])
                        #         biodegradation_profile[day] = float(row['biodegradation'])
                        
                        result.update({
                            'disintegration_prediction': max_L_pred,
                            'disintegration_unit': compost.get('unit', '% disintegration'),
                            'disintegration_day_30': disintegration_day_30,
                            'disintegration_day_90': disintegration_day_90,
                            'disintegration_day_180': disintegration_day_180,
                            'disintegration_max': max_L_pred,
                            'disintegration_time_profile': json.dumps(disintegration_profile) if disintegration_profile else None,
                            # 'biodegradation_prediction': curve_result.get('max_biodegradation'),  # Commented out
                            # 'biodegradation_unit': '% biodegradation',
                            # 'biodegradation_day_30': biodegradation_day_30,
                            # 'biodegradation_day_90': biodegradation_day_90,
                            # 'biodegradation_day_180': biodegradation_day_180,
                            # 'biodegradation_max': curve_result.get('max_biodegradation'),
                            # 'biodegradation_time_profile': json.dumps(biodegradation_profile) if biodegradation_profile else None,
                        })
                    else:
                        # Fallback to basic values if curve generation fails
                        result.update({
                            'disintegration_prediction': max_L_pred,
                            'disintegration_unit': compost.get('unit', '% disintegration'),
                            'disintegration_day_30': None,
                            'disintegration_day_90': None,
                            'disintegration_day_180': None,
                            'disintegration_max': max_L_pred,
                            'disintegration_time_profile': None,
                            # 'biodegradation_prediction': None,  # Commented out
                            # 'biodegradation_unit': '% biodegradation',
                            # 'biodegradation_day_30': None,
                            # 'biodegradation_day_90': None,
                            # 'biodegradation_day_180': None,
                            # 'biodegradation_max': None,
                            # 'biodegradation_time_profile': None,
                        })
                except Exception as e:
                    logger.warning(f"Curve generation failed: {e}")
                    # Fallback to basic values
                    result.update({
                        'disintegration_prediction': max_L_pred,
                        'disintegration_unit': compost.get('unit', '% disintegration'),
                        'disintegration_day_30': None,
                        'disintegration_day_90': None,
                        'disintegration_day_180': None,
                        'disintegration_max': max_L_pred,
                        'disintegration_time_profile': None,
                        # 'biodegradation_prediction': None,  # Commented out
                        # 'biodegradation_unit': '% biodegradation',
                        # 'biodegradation_day_30': None,
                        # 'biodegradation_day_90': None,
                        # 'biodegradation_day_180': None,
                        # 'biodegradation_max': None,
                        # 'biodegradation_time_profile': None,
                    })
            else:
                # No curve generation available, use basic values
                result.update({
                    'disintegration_prediction': max_L_pred,
                    'disintegration_unit': compost.get('unit', '% disintegration'),
                    'disintegration_day_30': None,
                    'disintegration_day_90': None,
                    'disintegration_day_180': None,
                    'disintegration_max': max_L_pred,
                    'disintegration_time_profile': None,
                    # 'biodegradation_prediction': None,  # Commented out
                    # 'biodegradation_unit': '% biodegradation',
                    # 'biodegradation_day_30': None,
                    # 'biodegradation_day_90': None,
                    # 'biodegradation_day_180': None,
                    # 'biodegradation_max': None,
                    # 'biodegradation_time_profile': None,
                })
            

        else:
            # Set all predictions to None if failed
            result.update({
                'wvtr_prediction': None, 'wvtr_unit': 'g/m²/day',
                'ts_prediction': None, 'ts_unit': 'MPa',
                'eab_prediction': None, 'eab_unit': '%',
                'cobb_prediction': None, 'cobb_unit': 'g/m²',
                'otr_prediction': None, 'otr_unit': 'cc/m²/day',
                'seal_prediction': None, 'seal_unit': 'N/15mm',
                'sealing_temp_pred': None, 'sealing_temp_unit': '°C',
                'disintegration_prediction': None, 'disintegration_unit': '% disintegration',
                'disintegration_day_30': None, 'disintegration_day_90': None, 
                'disintegration_day_180': None, 'disintegration_max': None,
                'disintegration_time_profile': None,
                # 'biodegradation_prediction': None, 'biodegradation_unit': 'days',  # Commented out
                # 'biodegradation_day_30': None, 'biodegradation_day_90': None, 
                # 'biodegradation_day_180': None, 'biodegradation_max': None,
                # 'biodegradation_time_profile': None
            })
        
        return result
    

    

    


def main():
    """Main function to run the database generation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Polymer Blend Database Generator (blenddatabase.csv format)')
    parser.add_argument('--property', type=str, default='all', 
                       choices=['wvtr', 'ts', 'eab', 'cobb', 'otr', 'compost', 'all'],
                       help='Property to analyze (ignored - always predicts all properties)')
    parser.add_argument('--max_materials', type=int, default=20,
                       help='Maximum number of materials to use from dictionary (default: 20)')
    parser.add_argument('--n_polymers', type=int, action='append', choices=[1,2,3,4,5],
                       help='Number of polymers per blend to test. Can be specified multiple times. (default: 2)')
    parser.add_argument('--full_exploration', action='store_true',
                       help='Test all material combinations per polymer type (default: False)')
    parser.add_argument('--random_samples', type=int, default=1,
                       help='Number of random volume fraction samples per combination (default: 1)')
    parser.add_argument('--temperature', type=float, default=None,
                       help='Temperature for environmental parameters (default: property-specific)')
    parser.add_argument('--rh', type=float, default=None,
                       help='Relative humidity for environmental parameters (default: property-specific)')
    parser.add_argument('--thickness', type=float, action='append',
                       help='Thickness for environmental parameters. Can be specified multiple times. (default: property-specific)')
    args = parser.parse_args()
    
    if not args.n_polymers:
        args.n_polymers = [2]
    
    # Set default thickness if none provided
    if not args.thickness:
        args.thickness = [50]  # Default thickness
    
    print(f"Polymer Blend Database Generator")
    print("=" * 50)
    print(f"Generating database with all properties (wvtr, ts, eab, cobb, otr, seal, compost)")
    print(f"Using first {args.max_materials} molecules from materials dictionary")
    print(f"Testing for blend types: {args.n_polymers}-polymer blends")
    print(f"Testing thickness values: {args.thickness} μm")
    print(f"Full material exploration: {args.full_exploration}")
    print(f"Random samples per combination: {args.random_samples}")
    
    # Initialize database generator
    generator = PolymerBlendDatabaseGenerator(property_type=args.property, max_materials=args.max_materials)
    
    # Temperature and RH are ignored - use fixed conditions for WVTR/OTR
    # Only thickness is used from command line
    
    # Check that thickness is provided
    if not args.thickness:
        print("❌ ERROR: Thickness must be provided! Use --thickness <value>")
        return
    
    if len(generator.materials) == 0:
        print("❌ No materials loaded. Please check the material dictionary file.")
        return
    
    print(f"✅ Loaded {len(generator.materials)} material-grade combinations")
    env_params_str = ", ".join([f"{k}={v}" for k, v in generator.fixed_env_params.items()])
    print(f"🔧 Fixed environmental parameters: {env_params_str}")
    
    # Generate all combinations for the selected n_polymers
    all_combinations = []
    for n in args.n_polymers:
        if args.full_exploration and n < 4:
            # Full exploration: test all material combinations (but limit to 3-polymer blends)
            combos = generator._generate_full_exploration_combinations(
                generator._group_materials_by_polymer_type(),
                list(generator._group_materials_by_polymer_type().keys()),
                n, float('inf')
            )
        else:
            # Limited exploration: one material per polymer type (forced for 4+ polymer blends)
            if n >= 4:
                print(f"⚠️ Forcing limited exploration for {n}-polymer blends to prevent excessive computation")
            combos = generator._generate_limited_exploration_combinations(
                generator._group_materials_by_polymer_type(),
                list(generator._group_materials_by_polymer_type().keys()),
                n, float('inf')
            )
        all_combinations.extend(combos)
    
    # Calculate total tests including thickness variations
    num_combinations = len(all_combinations)
    num_thickness_values = len(args.thickness)
    num_tests = num_combinations * num_thickness_values * args.random_samples
    print(f"Total unique blends to test: {num_combinations}")
    print(f"Total thickness values: {num_thickness_values}")
    print(f"Total tests ({args.random_samples} random samples per blend per thickness): {num_tests}")
    
    # Generate database with multiple thickness values
    print("\n🚀 Starting database generation...")
    all_results = []
    overall_start_time = time.time()
    
    for thickness_idx, thickness in enumerate(args.thickness, 1):
        print(f"📏 Testing thickness: {thickness} μm ({thickness_idx}/{len(args.thickness)})")
        
        # Set thickness for this iteration
        generator.fixed_env_params['thickness'] = thickness
        
        # Generate database for this thickness (random samples only)
        results_df = generator.generate_database(
            combinations=all_combinations,
            random_samples=args.random_samples,
            blend_types=['random']  # Only random samples, no equal mixtures
        )
        
        if len(results_df) > 0:
            all_results.append(results_df)
    
    # Calculate total time
    total_overall_time = time.time() - overall_start_time
    print(f"⏱️  Total time: {total_overall_time:.1f}s")
    
    # Combine all results
    if all_results:
        final_results_df = pd.concat(all_results, ignore_index=True)
        
        # Save the combined results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_df.to_csv(f'qa_database_{timestamp}.csv', index=False)
        print(f"📊 Database saved: qa_database_{timestamp}.csv")
    else:
        final_results_df = pd.DataFrame()
    
    # Print database summary
    print("\n📊 Database Summary:")
    print("=" * 50)
    print(f"✅ Total samples generated: {len(final_results_df):,}")
    
    if len(final_results_df) > 0:
        # Count by blend type
        blend_counts = final_results_df['blend_type'].value_counts()
        for blend_type, count in blend_counts.items():
            print(f"  {blend_type.capitalize()} blends: {count:,}")
        
        # Count by thickness
        thickness_counts = final_results_df['thickness_um'].value_counts().sort_index()
        print(f"\n📏 Thickness distribution:")
        for thickness, count in thickness_counts.items():
            print(f"  {thickness} μm: {count:,} samples")
        
        # Calculate success rates for each property
        print(f"\n📈 Property Prediction Success Rates:")
        properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'disintegration']  # biodegradation commented out
        for prop in properties:
            pred_col = f'{prop}_prediction'
            if pred_col in final_results_df.columns:
                success_count = final_results_df[pred_col].notna().sum()
                success_rate = (success_count / len(final_results_df)) * 100
                print(f"  {prop.upper()}: {success_rate:.1f}% ({success_count:,}/{len(final_results_df):,})")
        
        # Show sample of successful predictions
        print(f"\n🔍 Sample of successful predictions:")
        successful_samples = final_results_df[
            final_results_df['wvtr_prediction'].notna() | 
            final_results_df['ts_prediction'].notna() | 
            final_results_df['eab_prediction'].notna() | 
            final_results_df['cobb_prediction'].notna() | 
            final_results_df['otr_prediction'].notna() |
            final_results_df['seal_prediction'].notna() |
            final_results_df['disintegration_prediction'].notna()
            # final_results_df['biodegradation_prediction'].notna()  # Commented out
        ].head(3)
        
        for idx, (_, row) in enumerate(successful_samples.iterrows(), 1):
            print(f"  Sample {idx}: {row['blend_type']} blend")
            if row['wvtr_prediction'] is not None:
                print(f"    WVTR: {row['wvtr_prediction']:.2f} {row['wvtr_unit']}")
            if row['ts_prediction'] is not None:
                print(f"    TS: {row['ts_prediction']:.2f} {row['ts_unit']}")
            if row['eab_prediction'] is not None:
                print(f"    EAB: {row['eab_prediction']:.2f} {row['eab_unit']}")
            if row['cobb_prediction'] is not None:
                print(f"    Cobb: {row['cobb_prediction']:.2f} {row['cobb_unit']}")
            if row['otr_prediction'] is not None:
                print(f"    OTR: {row['otr_prediction']:.2f} {row['otr_unit']}")
            if row['seal_prediction'] is not None:
                print(f"    Seal: {row['seal_prediction']:.2f} {row['seal_unit']}")
            if row['disintegration_prediction'] is not None:
                print(f"    Disintegration: {row['disintegration_prediction']:.2f} {row['disintegration_unit']}")
            # if row['biodegradation_prediction'] is not None:  # Commented out
            #     print(f"    Biodegradation: {row['biodegradation_prediction']:.2f} {row['biodegradation_unit']}")
            print()
    else:
        print("⚠️  No samples were generated. This might be due to:")
        print("   - Insufficient materials in the dictionary")
        print("   - Not enough different polymer types for the requested blend combinations")
        print("   - Issues with material loading or combination generation")
    
    print(f"\n✅ Database generation complete!")
    print(f"📊 Database saved to: qa_database_*.csv")
    print(f"📈 Database contains {len(final_results_df):,} samples with all property predictions")

if __name__ == "__main__":
    main() 