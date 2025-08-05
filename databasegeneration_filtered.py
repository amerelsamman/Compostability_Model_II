#!/usr/bin/env python3
"""
Modified Database Generation Script with Filtered Time Profiles
Filters time-degradation profiles to every 15 days instead of daily data points.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Dict, Any
from itertools import combinations

# Import the modular predictor
from modules_home.predictor import predict_compostability_core

# Import other prediction modules
from modules.prediction_engine import PredictionEngine
from modules.input_parser import InputParser
from modules.output_formatter import OutputFormatter

# Import old compostability function as fallback
try:
    from homecompost_modules.core_model import generate_csv_for_single_blend
    OLD_COMPOST_AVAILABLE = True
except ImportError:
    OLD_COMPOST_AVAILABLE = False

# Import new compostability function
try:
    from modules_home.predictor import predict_compostability_core
    NEW_COMPOST_AVAILABLE = True
except ImportError:
    NEW_COMPOST_AVAILABLE = False

class PolymerBlendDatabaseGenerator:
    def __init__(self, property_type='wvtr', material_dict_path='material-smiles-dictionary.csv', max_materials=22):
        """
        Initialize the database generator.
        
        Args:
            property_type: Type of property to predict (wvtr, ts, eab, cobb)
            material_dict_path: Path to the material SMILES dictionary
            max_materials: Maximum number of materials to use
        """
        self.property_type = property_type
        self.material_dict_path = material_dict_path
        self.max_materials = max_materials
        self.materials = self._load_materials()
        self.prediction_engine = PredictionEngine()
        self.input_parser = InputParser()
        self.output_formatter = OutputFormatter()
        
    def _load_materials(self) -> List[Tuple[str, str]]:
        """Load materials from the SMILES dictionary."""
        try:
            df = pd.read_csv(self.material_dict_path)
            materials = []
            
            for _, row in df.iterrows():
                commercial_name = row['commercialName'].strip()
                smiles = row['SMILES'].strip()
                
                # Skip if SMILES is invalid or empty
                if pd.isna(smiles) or smiles == '' or smiles == 'nan':
                    continue
                    
                materials.append((commercial_name, smiles))
            
            # Limit to max_materials
            if len(materials) > self.max_materials:
                materials = materials[:self.max_materials]
            
            print(f"Loaded {len(materials)} materials from {self.material_dict_path}")
            return materials
            
        except Exception as e:
            print(f"Error loading materials: {e}")
            return []
    
    def _group_materials_by_polymer_type(self) -> Dict[str, List[Tuple[str, str]]]:
        """Group materials by polymer type."""
        polymer_groups = {}
        
        for commercial_name, smiles in self.materials:
            # Extract polymer type from commercial name (simplified)
            if 'PLA' in commercial_name.upper():
                polymer_type = 'PLA'
            elif 'PBAT' in commercial_name.upper():
                polymer_type = 'PBAT'
            elif 'PBS' in commercial_name.upper():
                polymer_type = 'PBS'
            elif 'PCL' in commercial_name.upper():
                polymer_type = 'PCL'
            elif 'PGA' in commercial_name.upper():
                polymer_type = 'PGA'
            elif 'PHA' in commercial_name.upper() or 'PHB' in commercial_name.upper():
                polymer_type = 'PHA'
            elif 'Bio-PE' in commercial_name.upper() or 'STN' in commercial_name.upper():
                polymer_type = 'Bio-PE'
            else:
                polymer_type = 'Other'
            
            if polymer_type not in polymer_groups:
                polymer_groups[polymer_type] = []
            polymer_groups[polymer_type].append((commercial_name, smiles))
        
        return polymer_groups
    
    def _generate_comprehensive_combinations(self, max_combinations: int = 100, max_polymers: int = 2, full_exploration: bool = False) -> List[List[Tuple[str, str]]]:
        """Generate comprehensive material combinations."""
        polymer_groups = self._group_materials_by_polymer_type()
        polymer_types = list(polymer_groups.keys())
        
        all_combinations = []
        
        # Generate combinations for different numbers of polymers
        for n_polymers in range(2, max_polymers + 1):
            if full_exploration:
                combinations_list = self._generate_full_exploration_combinations(polymer_groups, polymer_types, n_polymers, max_combinations - len(all_combinations))
            else:
                combinations_list = self._generate_limited_exploration_combinations(polymer_groups, polymer_types, n_polymers, max_combinations - len(all_combinations))
            
            all_combinations.extend(combinations_list)
            
            if len(all_combinations) >= max_combinations:
                break
        
        return all_combinations[:max_combinations]
    
    def _generate_limited_exploration_combinations(self, polymer_groups: Dict, polymer_types: List[str], 
                                                 n_polymers: int, max_remaining: int) -> List[List[Tuple[str, str]]]:
        """Generate limited exploration combinations."""
        combinations_list = []
        
        # Generate combinations of polymer types
        for polymer_combo in combinations(polymer_types, n_polymers):
            # For each polymer type, select one material
            material_combinations = []
            
            for polymer_type in polymer_combo:
                if polymer_type in polymer_groups:
                    materials = polymer_groups[polymer_type]
                    # Select first material from each polymer type
                    if materials:
                        material_combinations.append(materials[0])
            
            if len(material_combinations) == n_polymers:
                combinations_list.append(material_combinations)
        
        return combinations_list[:max_remaining]
    
    def _generate_full_exploration_combinations(self, polymer_groups: Dict, polymer_types: List[str], 
                                              n_polymers: int, max_remaining: int) -> List[List[Tuple[str, str]]]:
        """Generate full exploration combinations."""
        combinations_list = []
        
        # Generate combinations of polymer types
        for polymer_combo in combinations(polymer_types, n_polymers):
            # For each polymer type, try all materials
            polymer_materials = []
            
            for polymer_type in polymer_combo:
                if polymer_type in polymer_groups:
                    polymer_materials.append(polymer_groups[polymer_type])
            
            # Generate all combinations of materials from different polymer types
            if polymer_materials:
                from itertools import product
                for material_combo in product(*polymer_materials):
                    combinations_list.append(list(material_combo))
        
        return combinations_list[:max_remaining]
    
    def _create_blend_input_string(self, materials: List[Tuple[str, str]], 
                                  blend_type: str = 'equal') -> str:
        """Create blend input string for prediction."""
        if not materials:
            return ""
        
        n_materials = len(materials)
        
        if blend_type == 'equal':
            # Equal concentration for all materials
            concentration = 1.0 / n_materials
            blend_parts = []
            
            for commercial_name, smiles in materials:
                blend_parts.append(f"{commercial_name}:{concentration:.3f}")
            
            return ";".join(blend_parts)
        
        elif blend_type == 'random':
            # Random concentrations that sum to 1.0
            concentrations = np.random.dirichlet(np.ones(n_materials))
            blend_parts = []
            
            for (commercial_name, smiles), concentration in zip(materials, concentrations):
                blend_parts.append(f"{commercial_name}:{concentration:.3f}")
            
            return ";".join(blend_parts)
        
        else:
            # Default to equal concentration
            return self._create_blend_input_string(materials, 'equal')
    
    def generate_database(self, combinations: List[List[Tuple[str, str]]], random_samples: int, 
                         blend_types: List[str] = None) -> pd.DataFrame:
        """Generate database with filtered time profiles."""
        if blend_types is None:
            blend_types = ['equal', 'random']
        
        results = []
        total_tests = len(combinations) * len(blend_types) * random_samples
        
        print(f"Generating database with {total_tests} total tests...")
        print(f"Combinations: {len(combinations)}")
        print(f"Blend types: {blend_types}")
        print(f"Random samples per combination: {random_samples}")
        
        test_count = 0
        
        for combo_idx, materials in enumerate(combinations):
            print(f"Processing combination {combo_idx + 1}/{len(combinations)}: {[m[0] for m in materials]}")
            
            for blend_type in blend_types:
                for sample in range(random_samples):
                    test_count += 1
                    
                    if test_count % 100 == 0:
                        print(f"Progress: {test_count}/{total_tests} tests completed")
                    
                    # Create blend input string
                    blend_input = self._create_blend_input_string(materials, blend_type)
                    
                    if blend_input:
                        # Run predictions for all properties
                        all_properties_result = self._run_all_properties_prediction(blend_input)
                        
                        # Format result for database
                        result = self._format_result_for_database(materials, blend_input, blend_type, all_properties_result)
                        results.append(result)
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        return self.results_df
    
    def _run_all_properties_prediction(self, blend_input: str) -> Dict[str, Any]:
        """Run predictions for all properties including filtered compostability."""
        all_results = {}
        
        # Parse blend input
        parsed_input = self.input_parser.parse_blend_string(blend_input)
        
        if not parsed_input:
            return all_results
        
        # Get environmental parameters
        available_env_params = parsed_input.get('environmental_parameters', {})
        
        # Predict WVTR
        try:
            wvtr_result = self.prediction_engine.predict_property('wvtr', blend_input, available_env_params)
            if wvtr_result:
                all_results['wvtr'] = wvtr_result
        except Exception as e:
            print(f"WVTR prediction error: {e}")
        
        # Predict TS
        try:
            ts_result = self.prediction_engine.predict_property('ts', blend_input, available_env_params)
            if ts_result:
                all_results['ts'] = ts_result
        except Exception as e:
            print(f"TS prediction error: {e}")
        
        # Predict EAB
        try:
            eab_result = self.prediction_engine.predict_property('eab', blend_input, available_env_params)
            if eab_result:
                all_results['eab'] = eab_result
        except Exception as e:
            print(f"EAB prediction error: {e}")
        
        # Predict Cobb
        try:
            cobb_result = self.prediction_engine.predict_property('cobb', blend_input, available_env_params)
            if cobb_result:
                all_results['cobb'] = cobb_result
        except Exception as e:
            print(f"Cobb prediction error: {e}")
        
        # Predict Compostability with filtered time profiles
        try:
            thickness = available_env_params.get('thickness', 100) / 1000.0  # Convert to mm
            
            # Use new modular predictor
            if NEW_COMPOST_AVAILABLE:
                compost_result = predict_compostability_core(
                    blend_input, 
                    actual_thickness=thickness,
                    suppress_output=True,
                    save_plots=False
                )
                
                if compost_result:
                    # Extract disintegration curve data
                    disintegration_df = compost_result['disintegration_curve']
                    biodegradation_df = compost_result['biodegradation_curve']
                    
                    # Convert DataFrames to filtered time profile dictionaries (every 15 days)
                    disintegration_profile = {}
                    biodegradation_profile = {}
                    
                    if not disintegration_df.empty:
                        for _, row in disintegration_df.iterrows():
                            day = int(row['day'])
                            # Only include every 15 days (0, 15, 30, 45, 60, ..., 195)
                            if day % 15 == 0 or day == 200:  # Include day 200 as the final point
                                disintegration_profile[day] = float(row['disintegration'])
                    
                    if not biodegradation_df.empty:
                        for _, row in biodegradation_df.iterrows():
                            day = int(row['day'])
                            # Only include every 15 days (0, 15, 30, 45, 60, ..., 390)
                            if day % 15 == 0 or day == 400:  # Include day 400 as the final point
                                biodegradation_profile[day] = float(row['biodegradation'])
                    
                    all_results['disintegration'] = {
                        'prediction': compost_result['max_disintegration'],
                        'unit': '% disintegration',
                        'day_30': disintegration_profile.get(30, 0),
                        'day_90': disintegration_profile.get(90, 0),
                        'day_180': disintegration_profile.get(180, 0),
                        'max': compost_result['max_disintegration'],
                        'full_time_profile': disintegration_profile,  # Filtered disintegration time profile
                    }
                    
                    all_results['biodegradation'] = {
                        'prediction': compost_result['max_biodegradation'],
                        'unit': '% biodegradation',
                        'day_30': biodegradation_profile.get(30, 0),
                        'day_90': biodegradation_profile.get(90, 0),
                        'day_180': biodegradation_profile.get(180, 0),
                        'max': compost_result['max_biodegradation'],
                        'full_time_profile': biodegradation_profile,  # Filtered biodegradation time profile
                    }
                else:
                    all_results['disintegration'] = {
                        'prediction': None, 
                        'unit': '% disintegration',
                        'day_30': None,
                        'day_90': None,
                        'day_180': None,
                        'max': None,
                        'full_time_profile': None,
                    }
                    
                    all_results['biodegradation'] = {
                        'prediction': None, 
                        'unit': '% biodegradation',
                        'day_30': None,
                        'day_90': None,
                        'day_180': None,
                        'max': None,
                        'full_time_profile': None,
                    }
            else:
                # Fallback to old compostability prediction
                if OLD_COMPOST_AVAILABLE:
                    compost_data = generate_csv_for_single_blend(blend_input, output_path=None, actual_thickness=thickness)
                    
                    if compost_data:
                        # Filter old compost data to every 15 days
                        filtered_compost_data = {}
                        for day, value in compost_data.items():
                            if day % 15 == 0 or day == 200:  # Include day 200 as the final point
                                filtered_compost_data[day] = value
                        
                        all_results['disintegration'] = {
                            'prediction': compost_data.get(180, 0),
                            'unit': '% disintegration',
                            'day_30': compost_data.get(30, 0),
                            'day_90': compost_data.get(90, 0),
                            'day_180': compost_data.get(180, 0),
                            'max': max(compost_data.values()) if compost_data else 0,
                            'full_time_profile': filtered_compost_data,  # Filtered compost data
                        }
                        
                        # Old model doesn't have biodegradation data
                        all_results['biodegradation'] = {
                            'prediction': None, 
                            'unit': '% biodegradation',
                            'day_30': None,
                            'day_90': None,
                            'day_180': None,
                            'max': None,
                            'full_time_profile': None,
                        }
                    else:
                        all_results['disintegration'] = {
                            'prediction': None, 
                            'unit': '% disintegration',
                            'day_30': None,
                            'day_90': None,
                            'day_180': None,
                            'max': None,
                            'full_time_profile': None,
                        }
                        
                        all_results['biodegradation'] = {
                            'prediction': None, 
                            'unit': '% biodegradation',
                            'day_30': None,
                            'day_90': None,
                            'day_180': None,
                            'max': None,
                            'full_time_profile': None,
                        }
                else:
                    all_results['disintegration'] = {
                        'prediction': None, 
                        'unit': '% disintegration',
                        'day_30': None,
                        'day_90': None,
                        'day_180': None,
                        'max': None,
                        'full_time_profile': None,
                    }
                    
                    all_results['biodegradation'] = {
                        'prediction': None, 
                        'unit': '% biodegradation',
                        'day_30': None,
                        'day_90': None,
                        'day_180': None,
                        'max': None,
                        'full_time_profile': None,
                    }
        except Exception as e:
            print(f"Compostability prediction error: {e}")
            all_results['disintegration'] = {
                'prediction': None, 
                'unit': '% disintegration',
                'day_30': None,
                'day_90': None,
                'day_180': None,
                'max': None,
                'full_time_profile': None,
            }
            
            all_results['biodegradation'] = {
                'prediction': None, 
                'unit': '% biodegradation',
                'day_30': None,
                'day_90': None,
                'day_180': None,
                'max': None,
                'full_time_profile': None,
            }
        
        return all_results
    
    def _format_result_for_database(self, materials: List[Tuple[str, str]], blend_input: str, 
                                   blend_type: str, all_properties_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format result for database with filtered time profiles."""
        result = {
            'blend_input': blend_input,
            'blend_type': blend_type,
            'n_materials': len(materials),
            'materials': [m[0] for m in materials],
            'smiles': [m[1] for m in materials]
        }
        
        # Add material-specific columns
        for i, (commercial_name, smiles) in enumerate(materials, 1):
            result[f'material{i}'] = commercial_name
            result[f'smiles{i}'] = smiles
        
        # Fill remaining material columns with None
        for i in range(len(materials) + 1, 5):
            result[f'material{i}'] = None
            result[f'smiles{i}'] = None
        
        # Add environmental parameters
        parsed_input = self.input_parser.parse_blend_string(blend_input)
        if parsed_input:
            env_params = parsed_input.get('environmental_parameters', {})
            result['temperature'] = env_params.get('temperature', 25)
            result['rh'] = env_params.get('rh', 50)
            result['thickness_um'] = env_params.get('thickness', 100)
        
        # Add property predictions
        properties = ['wvtr', 'ts', 'eab', 'cobb']
        for prop in properties:
            prop_data = all_properties_result.get(prop, {})
            result[f'{prop}_prediction'] = prop_data.get('prediction')
            result[f'{prop}_unit'] = prop_data.get('unit', '')
        
        # Add compostability predictions with filtered time profiles
        disintegration = all_properties_result.get('disintegration', {})
        biodegradation = all_properties_result.get('biodegradation', {})
        
        # Convert filtered time profile dictionaries to JSON strings for CSV storage
        disintegration_profile = disintegration.get('full_time_profile')
        biodegradation_profile = biodegradation.get('full_time_profile')
        
        if disintegration_profile is not None:
            # Convert numpy values to regular Python types for JSON serialization
            disintegration_profile_clean = {}
            for day, value in disintegration_profile.items():
                if isinstance(value, np.floating):
                    disintegration_profile_clean[day] = float(value)
                else:
                    disintegration_profile_clean[day] = value
            disintegration_profile_json = json.dumps(disintegration_profile_clean)
        else:
            disintegration_profile_json = None
        
        if biodegradation_profile is not None:
            # Convert numpy values to regular Python types for JSON serialization
            biodegradation_profile_clean = {}
            for day, value in biodegradation_profile.items():
                if isinstance(value, np.floating):
                    biodegradation_profile_clean[day] = float(value)
                else:
                    biodegradation_profile_clean[day] = value
            biodegradation_profile_json = json.dumps(biodegradation_profile_clean)
        else:
            biodegradation_profile_json = None
        
        result.update({
            'disintegration_prediction': disintegration.get('prediction'),
            'disintegration_unit': disintegration.get('unit', '% disintegration'),
            'disintegration_day_30': disintegration.get('day_30'),
            'disintegration_day_90': disintegration.get('day_90'),
            'disintegration_day_180': disintegration.get('day_180'),
            'disintegration_max': disintegration.get('max'),
            'disintegration_time_profile': disintegration_profile_json,  # Filtered disintegration time profile as JSON
            'biodegradation_prediction': biodegradation.get('prediction'),
            'biodegradation_unit': biodegradation.get('unit', '% biodegradation'),
            'biodegradation_day_30': biodegradation.get('day_30'),
            'biodegradation_day_90': biodegradation.get('day_90'),
            'biodegradation_day_180': biodegradation.get('day_180'),
            'biodegradation_max': biodegradation.get('max'),
            'biodegradation_time_profile': biodegradation_profile_json,  # Filtered biodegradation time profile as JSON
        })
        
        return result

def main():
    """Main function to run the database generation."""
    parser = argparse.ArgumentParser(description='Generate polymer blend database with filtered time profiles')
    parser.add_argument('--max_materials', type=int, default=22, help='Maximum number of materials to use')
    parser.add_argument('--n_polymers', type=int, nargs='+', default=[2], help='Number of polymers per blend')
    parser.add_argument('--full_exploration', action='store_true', help='Use full exploration mode')
    parser.add_argument('--random_samples', type=int, default=1, help='Number of random samples per combination')
    parser.add_argument('--thickness', type=int, nargs='+', default=[100], help='Thickness values in μm')
    parser.add_argument('--temperature', type=int, nargs='+', default=[25], help='Temperature values in °C')
    parser.add_argument('--rh', type=int, nargs='+', default=[50], help='Relative humidity values in %')
    parser.add_argument('--output', type=str, default='blends_database_filtered.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    # Create database generator
    generator = PolymerBlendDatabaseGenerator(max_materials=args.max_materials)
    
    # Generate combinations
    max_combinations = 1000  # Adjust as needed
    combinations = generator._generate_comprehensive_combinations(
        max_combinations=max_combinations,
        max_polymers=max(args.n_polymers),
        full_exploration=args.full_exploration
    )
    
    print(f"Generated {len(combinations)} material combinations")
    
    # Generate database
    results_df = generator.generate_database(combinations, args.random_samples)
    
    # Save results
    results_df.to_csv(args.output, index=False)
    
    # Print file size
    file_size = os.path.getsize(args.output)
    file_size_mb = file_size / (1024 * 1024)
    print(f"\nDatabase saved to: {args.output}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total rows: {len(results_df)}")
    
    # Print sample of time profile sizes
    disintegration_profiles = results_df['disintegration_time_profile'].dropna()
    biodegradation_profiles = results_df['biodegradation_time_profile'].dropna()
    
    if not disintegration_profiles.empty:
        sample_profile = json.loads(disintegration_profiles.iloc[0])
        print(f"Sample disintegration profile has {len(sample_profile)} data points (filtered from ~200)")
    
    if not biodegradation_profiles.empty:
        sample_profile = json.loads(biodegradation_profiles.iloc[0])
        print(f"Sample biodegradation profile has {len(sample_profile)} data points (filtered from ~400)")

if __name__ == "__main__":
    main() 