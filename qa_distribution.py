#!/usr/bin/env python3
"""
QA Script for Polymer Blend Property Distribution Analysis
Tests predict_unified_blend.py with various polymer blend combinations
and analyzes the distribution of property predictions.

Supports all properties: wvtr, ts, eab, cobb, compost, all
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations, product
import logging
from typing import List, Tuple, Dict, Any
from datetime import datetime
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Property configurations
PROPERTY_CONFIGS = {
    'wvtr': {
        'name': 'WVTR',
        'unit': 'g/m²/day',
        'env_params': ['temperature', 'rh', 'thickness'],
        'default_env': {'temperature': 38, 'rh': 90, 'thickness': 100},
        'parse_pattern': 'Predicted WVTR:',
        'log_scale': True
    },
    'ts': {
        'name': 'Tensile Strength',
        'unit': 'MPa',
        'env_params': ['thickness'],
        'default_env': {'thickness': 100},
        'parse_pattern': 'Predicted Tensile Strength:',
        'log_scale': True
    },
    'eab': {
        'name': 'Elongation at Break',
        'unit': '%',
        'env_params': ['thickness'],
        'default_env': {'thickness': 100},
        'parse_pattern': 'Predicted Elongation at Break:',
        'log_scale': True
    },
    'cobb': {
        'name': 'Cobb Value',
        'unit': 'g/m²',
        'env_params': [],
        'default_env': {},
        'parse_pattern': 'Predicted Cobb Value:',
        'log_scale': True
    },
    'compost': {
        'name': 'Home Compostability',
        'unit': '% disintegration',
        'env_params': ['thickness'],
        'default_env': {'thickness': 100},
        'parse_pattern': 'Max Disintegration:',
        'log_scale': False
    },
    'all': {
        'name': 'All Properties',
        'unit': 'mixed',
        'env_params': ['temperature', 'rh', 'thickness'],
        'default_env': {'temperature': 38, 'rh': 90, 'thickness': 100},
        'parse_pattern': None,  # Special handling for all properties
        'log_scale': False
    }
}

class PropertyQAAnalyzer:
    def __init__(self, property_type='wvtr', material_dict_path='material-smiles-dictionary.csv', max_materials=22):
        """Initialize the QA analyzer."""
        self.property_type = property_type.lower()
        if self.property_type not in PROPERTY_CONFIGS:
            raise ValueError(f"Unsupported property type: {property_type}. Supported: {list(PROPERTY_CONFIGS.keys())}")
        
        self.config = PROPERTY_CONFIGS[self.property_type]
        self.material_dict_path = material_dict_path
        self.max_materials = max_materials
        self.materials = self._load_materials()
        self.results = []
        self.fixed_env_params = self.config['default_env'].copy()
        
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
                if full_exploration:
                    # Full exploration: test all material combinations
                    combinations_list.extend(self._generate_full_exploration_combinations(
                        polymer_groups, polymer_types, n_polymers, max_combinations - len(combinations_list)
                    ))
                else:
                    # Limited exploration: one material per polymer type
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
        """Create input string for predict_unified_blend.py."""
        if blend_type == 'equal':
            # Equal volume fractions
            vol_fraction = 1.0 / len(materials)
            blend_parts = []
            for material, grade in materials:
                blend_parts.extend([material, grade, str(vol_fraction)])
        elif blend_type == 'random':
            # Random volume fractions that sum to 1.0
            fractions = np.random.dirichlet(np.ones(len(materials)))
            blend_parts = []
            for (material, grade), fraction in zip(materials, fractions):
                blend_parts.extend([material, grade, str(fraction)])
        else:
            raise ValueError(f"Unknown blend type: {blend_type}")
        
        return ', '.join(blend_parts)
    
    def _parse_property_value(self, output_lines: List[str]) -> float:
        """Parse property value from predict_unified_blend.py output."""
        if self.property_type == 'all':
            # For 'all' mode, we need to parse multiple properties
            # This is a simplified approach - you might want to enhance this
            for line in output_lines:
                if 'Predicted WVTR:' in line and 'g/m²/day' in line:
                    try:
                        parts = line.split(':')[1].strip().split()
                        return float(parts[0])
                    except (ValueError, IndexError):
                        continue
            return None
        else:
            # For single property mode
            pattern = self.config['parse_pattern']
            for line in output_lines:
                if pattern in line:
                    try:
                        # Extract numeric value
                        parts = line.split(':')[1].strip().split()
                        return float(parts[0])
                    except (ValueError, IndexError):
                        continue
            return None
    
    def _run_prediction(self, blend_input: str) -> Dict[str, Any]:
        """Run a single prediction using predict_unified_blend.py."""
        try:
            # Construct command
            cmd = [sys.executable, 'predict_unified_blend.py', self.property_type, blend_input]
            
            # Add environmental parameters
            for param, value in self.fixed_env_params.items():
                cmd.append(f"{param}={value}")
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Parse the output to extract property prediction
                output_lines = result.stdout.strip().split('\n')
                property_value = self._parse_property_value(output_lines)
                
                if property_value is not None:
                    return {
                        'success': True,
                        'property_value': property_value,
                        'output': result.stdout,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'property_value': None,
                        'output': result.stdout,
                        'error': f'Could not parse {self.config["name"]} value from output'
                    }
            else:
                return {
                    'success': False,
                    'property_value': None,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'property_value': None,
                'output': '',
                'error': 'Prediction timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'property_value': None,
                'output': '',
                'error': str(e)
            }
    
    def run_qa_tests(self, combinations: List[List[Tuple[str, str]]], random_samples: int, 
                    blend_types: List[str] = None) -> pd.DataFrame:
        """Run comprehensive QA tests."""
        if blend_types is None:
            blend_types = ['equal', 'random']
        
        logger.info("Starting QA tests...")
        logger.info(f"Property: {self.config['name']}")
        logger.info(f"Fixed environmental parameters: {self.fixed_env_params}")
        logger.info(f"Random samples per combination: {random_samples}")
        
        results = []
        total_tests = len(combinations) * len(blend_types)
        if 'random' in blend_types:
            total_tests += len(combinations) * (random_samples - 1)  # Additional random samples
        
        current_test = 0
        
        for materials in combinations:
            for blend_type in blend_types:
                if blend_type == 'random':
                    # Generate multiple random samples
                    for sample_idx in range(random_samples):
                        current_test += 1
                        logger.info(f"Test {current_test}/{total_tests}: {len(materials)} polymers, {blend_type} blend (sample {sample_idx + 1}/{random_samples})")
                        
                        # Create blend input
                        blend_input = self._create_blend_input_string(materials, blend_type)
                        
                        # Run prediction
                        prediction_result = self._run_prediction(blend_input)
                        
                        # Record results
                        result = {
                            'test_id': current_test,
                            'n_polymers': len(materials),
                            'blend_type': f"{blend_type}_sample_{sample_idx + 1}",
                            'materials': [f"{m[0]} {m[1]}" for m in materials],
                            'polymer_types': [m[0] for m in materials],
                            'blend_input': blend_input,
                            'success': prediction_result['success'],
                            'property_value': prediction_result['property_value'],
                            'error': prediction_result['error'],
                            **self.fixed_env_params
                        }
                        
                        results.append(result)
                        
                        # Log progress
                        if prediction_result['success']:
                            logger.info(f"  ✓ {self.config['name']}: {prediction_result['property_value']:.4f} {self.config['unit']}")
                        else:
                            logger.warning(f"  ✗ Failed: {prediction_result['error']}")
                else:
                    # Single test for equal blends
                    current_test += 1
                    logger.info(f"Test {current_test}/{total_tests}: {len(materials)} polymers, {blend_type} blend")
                    
                    # Create blend input
                    blend_input = self._create_blend_input_string(materials, blend_type)
                    
                    # Run prediction
                    prediction_result = self._run_prediction(blend_input)
                    
                    # Record results
                    result = {
                        'test_id': current_test,
                        'n_polymers': len(materials),
                        'blend_type': blend_type,
                        'materials': [f"{m[0]} {m[1]}" for m in materials],
                        'polymer_types': [m[0] for m in materials],
                        'blend_input': blend_input,
                        'success': prediction_result['success'],
                        'property_value': prediction_result['property_value'],
                        'error': prediction_result['error'],
                        **self.fixed_env_params
                    }
                    
                    results.append(result)
                    
                    # Log progress
                    if prediction_result['success']:
                        logger.info(f"  ✓ {self.config['name']}: {prediction_result['property_value']:.4f} {self.config['unit']}")
                    else:
                        logger.warning(f"  ✗ Failed: {prediction_result['error']}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_df.to_csv(f'qa_{self.property_type}_results_{timestamp}.csv', index=False)
        logger.info(f"Results saved to qa_{self.property_type}_results_{timestamp}.csv")
        
        return self.results_df
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze the property distributions."""
        if self.results_df is None or len(self.results_df) == 0:
            logger.error("No results to analyze")
            return {}
        
        # Filter successful predictions
        successful_results = self.results_df[self.results_df['success'] == True].copy()
        
        if len(successful_results) == 0:
            logger.warning("No successful predictions to analyze")
            return {}
        
        # Basic statistics
        stats = {
            'total_tests': len(self.results_df),
            'successful_tests': len(successful_results),
            'success_rate': len(successful_results) / len(self.results_df),
            'property_stats': {
                'mean': successful_results['property_value'].mean(),
                'std': successful_results['property_value'].std(),
                'min': successful_results['property_value'].min(),
                'max': successful_results['property_value'].max(),
                'median': successful_results['property_value'].median(),
                'q25': successful_results['property_value'].quantile(0.25),
                'q75': successful_results['property_value'].quantile(0.75)
            }
        }
        
        # Statistics by number of polymers
        polymer_stats = {}
        for n in successful_results['n_polymers'].unique():
            subset = successful_results[successful_results['n_polymers'] == n]
            polymer_stats[n] = {
                'count': len(subset),
                'mean': subset['property_value'].mean(),
                'std': subset['property_value'].std(),
                'min': subset['property_value'].min(),
                'max': subset['property_value'].max(),
                'median': subset['property_value'].median()
            }
        stats['by_polymer_count'] = polymer_stats
        
        # Statistics by blend type
        blend_stats = {}
        for blend_type in successful_results['blend_type'].unique():
            subset = successful_results[successful_results['blend_type'] == blend_type]
            blend_stats[blend_type] = {
                'count': len(subset),
                'mean': subset['property_value'].mean(),
                'std': subset['property_value'].std(),
                'min': subset['property_value'].min(),
                'max': subset['property_value'].max(),
                'median': subset['property_value'].median()
            }
        stats['by_blend_type'] = blend_stats
        
        # Check polymer type coverage
        all_polymer_types = set()
        for polymer_types in successful_results['polymer_types']:
            all_polymer_types.update(polymer_types)
        
        stats['polymer_type_coverage'] = {
            'total_unique_types': len(all_polymer_types),
            'covered_types': list(all_polymer_types)
        }
        
        return stats
    
    def create_plots(self, save_dir: str = 'qa_plots'):
        """Create distribution plots."""
        if self.results_df is None or len(self.results_df) == 0:
            logger.error("No results to plot")
            return
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Filter successful predictions
        successful_results = self.results_df[self.results_df['success'] == True].copy()
        
        if len(successful_results) == 0:
            logger.warning("No successful predictions to plot")
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create property-specific plot title
        env_params_str = ", ".join([f"{k}={v}" for k, v in self.fixed_env_params.items()])
        plot_title = f'{self.config["name"]} Distribution\n({env_params_str})'
        
        # 1. Overall property distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(successful_results['property_value'], bins=30, kde=True, ax=ax)
        ax.set_xlabel(f'{self.config["name"]} ({self.config["unit"]})')
        ax.set_ylabel('Frequency')
        ax.set_title(plot_title)
        ax.axvline(successful_results['property_value'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {successful_results["property_value"].mean():.2f}')
        ax.axvline(successful_results['property_value'].median(), color='green', linestyle='--', 
                  label=f'Median: {successful_results["property_value"].median():.2f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_{self.property_type}_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Property distribution by number of polymers
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, n_polymers in enumerate(sorted(successful_results['n_polymers'].unique())):
            if i >= len(axes):
                break
            subset = successful_results[successful_results['n_polymers'] == n_polymers]
            ax = axes[i]
            
            sns.histplot(subset['property_value'], bins=20, kde=True, ax=ax)
            ax.set_xlabel(f'{self.config["name"]} ({self.config["unit"]})')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{n_polymers}-Polymer Blends (n={len(subset)})')
            ax.axvline(subset['property_value'].mean(), color='red', linestyle='--', 
                      label=f'Mean: {subset["property_value"].mean():.2f}')
            ax.axvline(subset['property_value'].median(), color='green', linestyle='--', 
                      label=f'Median: {subset["property_value"].median():.2f}')
            ax.legend()
        
        # Hide unused subplots
        for i in range(len(successful_results['n_polymers'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.property_type}_by_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plot by number of polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=successful_results, x='n_polymers', y='property_value', ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel(f'{self.config["name"]} ({self.config["unit"]})')
        ax.set_title(f'{self.config["name"]} Distribution by Number of Polymers\n({env_params_str})')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.property_type}_boxplot_by_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Violin plot by blend type
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(data=successful_results, x='blend_type', y='property_value', ax=ax)
        ax.set_xlabel('Blend Type')
        ax.set_ylabel(f'{self.config["name"]} ({self.config["unit"]})')
        ax.set_title(f'{self.config["name"]} Distribution by Blend Type\n({env_params_str})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.property_type}_violin_by_blend_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Scatter plot: Property vs Number of Polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=successful_results, x='n_polymers', y='property_value', 
                       hue='blend_type', alpha=0.6, ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel(f'{self.config["name"]} ({self.config["unit"]})')
        ax.set_title(f'{self.config["name"]} vs Number of Polymers\n({env_params_str})')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.property_type}_vs_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}/")
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive QA report."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'qa_{self.property_type}_report_{timestamp}.txt'
        
        # Analyze distributions
        stats = self.analyze_distributions()
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{self.config['name'].upper()} QA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Configuration:\n")
            f.write(f"  - Property: {self.config['name']}\n")
            f.write(f"  - Unit: {self.config['unit']}\n")
            for param, value in self.fixed_env_params.items():
                f.write(f"  - Fixed {param}: {value}\n")
            f.write(f"  - Total Tests: {stats.get('total_tests', 0)}\n")
            f.write(f"  - Successful Tests: {stats.get('successful_tests', 0)}\n")
            f.write(f"  - Success Rate: {stats.get('success_rate', 0):.2%}\n\n")
            
            # Add prominent property range summary section
            if 'by_polymer_count' in stats:
                f.write("=" * 80 + "\n")
                f.write(f"{self.config['name'].upper()} RANGE SUMMARY BY POLYMER BLEND TYPE\n")
                f.write("=" * 80 + "\n\n")
                
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"🔬 {n_polymers}-POLYMER BLENDS (n={poly_stats['count']}):\n")
                    f.write(f"   📊 {self.config['name']} Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} {self.config['unit']}\n")
                    f.write(f"   📈 Mean {self.config['name']}: {poly_stats['mean']:.2f} {self.config['unit']}\n")
                    f.write(f"   📉 Median {self.config['name']}: {poly_stats['median']:.2f} {self.config['unit']}\n")
                    f.write(f"   📋 Min {self.config['name']}: {poly_stats['min']:.2f} {self.config['unit']}\n")
                    f.write(f"   📋 Max {self.config['name']}: {poly_stats['max']:.2f} {self.config['unit']}\n\n")
                
                f.write("=" * 80 + "\n\n")
            
            if 'property_stats' in stats:
                f.write(f"Overall {self.config['name']} Statistics:\n")
                property_stats = stats['property_stats']
                f.write(f"  - Mean: {property_stats['mean']:.2f} {self.config['unit']}\n")
                f.write(f"  - Median: {property_stats['median']:.2f} {self.config['unit']}\n")
                f.write(f"  - Std Dev: {property_stats['std']:.2f} {self.config['unit']}\n")
                f.write(f"  - Min: {property_stats['min']:.2f} {self.config['unit']}\n")
                f.write(f"  - Max: {property_stats['max']:.2f} {self.config['unit']}\n")
                f.write(f"  - Q25: {property_stats['q25']:.2f} {self.config['unit']}\n")
                f.write(f"  - Q75: {property_stats['q75']:.2f} {self.config['unit']}\n\n")
            
            if 'by_polymer_count' in stats:
                f.write(f"Detailed {self.config['name']} Statistics by Number of Polymers:\n")
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):\n")
                    f.write(f"    - Mean: {poly_stats['mean']:.2f} {self.config['unit']}\n")
                    f.write(f"    - Median: {poly_stats['median']:.2f} {self.config['unit']}\n")
                    f.write(f"    - Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} {self.config['unit']}\n\n")
            
            if 'by_blend_type' in stats:
                f.write(f"{self.config['name']} Statistics by Blend Type:\n")
                for blend_type, blend_stats in stats['by_blend_type'].items():
                    f.write(f"  {blend_type.capitalize()} Blends (n={blend_stats['count']}):\n")
                    f.write(f"    - Mean: {blend_stats['mean']:.2f} {self.config['unit']}\n")
                    f.write(f"    - Median: {blend_stats['median']:.2f} {self.config['unit']}\n")
                    f.write(f"    - Range: {blend_stats['min']:.2f} - {blend_stats['max']:.2f} {self.config['unit']}\n\n")
            
            if 'polymer_type_coverage' in stats:
                f.write("Polymer Type Coverage:\n")
                coverage = stats['polymer_type_coverage']
                f.write(f"  - Total Unique Polymer Types: {coverage['total_unique_types']}\n")
                f.write(f"  - Covered Types: {', '.join(sorted(coverage['covered_types']))}\n\n")
            
            # Add error analysis if any
            if self.results_df is not None:
                failed_tests = self.results_df[self.results_df['success'] == False]
                if len(failed_tests) > 0:
                    f.write("Error Analysis:\n")
                    error_counts = failed_tests['error'].value_counts()
                    for error, count in error_counts.items():
                        f.write(f"  - {error}: {count} occurrences\n")
        
        logger.info(f"Report saved to {save_path}")
        return save_path

def main():
    """Main function to run the QA analysis."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Polymer Blend Property QA Distribution Analysis')
    parser.add_argument('--property', type=str, default='wvtr', 
                       choices=['wvtr', 'ts', 'eab', 'cobb', 'compost', 'all'],
                       help='Property to analyze (default: wvtr)')
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
    parser.add_argument('--thickness', type=float, default=None,
                       help='Thickness for environmental parameters (default: property-specific)')
    args = parser.parse_args()
    
    if not args.n_polymers:
        args.n_polymers = [2]
    
    print(f"{PROPERTY_CONFIGS[args.property]['name']} QA Distribution Analysis")
    print("=" * 50)
    print(f"Property: {PROPERTY_CONFIGS[args.property]['name']} ({PROPERTY_CONFIGS[args.property]['unit']})")
    print(f"Using first {args.max_materials} molecules from materials dictionary")
    print(f"Testing for blend types: {args.n_polymers}-polymer blends")
    print(f"Full material exploration: {args.full_exploration}")
    print(f"Random samples per combination: {args.random_samples}")
    
    # Initialize analyzer
    analyzer = PropertyQAAnalyzer(property_type=args.property, max_materials=args.max_materials)
    
    # Override environmental parameters if provided
    if args.temperature is not None:
        analyzer.fixed_env_params['temperature'] = args.temperature
    if args.rh is not None:
        analyzer.fixed_env_params['rh'] = args.rh
    if args.thickness is not None:
        analyzer.fixed_env_params['thickness'] = args.thickness
    
    if len(analyzer.materials) == 0:
        print("❌ No materials loaded. Please check the material dictionary file.")
        return
    
    print(f"✅ Loaded {len(analyzer.materials)} material-grade combinations")
    env_params_str = ", ".join([f"{k}={v}" for k, v in analyzer.fixed_env_params.items()])
    print(f"🔧 Fixed environmental parameters: {env_params_str}")
    
    # Generate all combinations for the selected n_polymers
    all_combinations = []
    for n in args.n_polymers:
        if args.full_exploration:
            combos = analyzer._generate_full_exploration_combinations(
                analyzer._group_materials_by_polymer_type(),
                list(analyzer._group_materials_by_polymer_type().keys()),
                n, float('inf')
            )
        else:
            combos = analyzer._generate_limited_exploration_combinations(
                analyzer._group_materials_by_polymer_type(),
                list(analyzer._group_materials_by_polymer_type().keys()),
                n, float('inf')
            )
        all_combinations.extend(combos)
    num_combinations = len(all_combinations)
    num_tests = num_combinations * (1 + args.random_samples)
    print(f"Total unique blends to test: {num_combinations}")
    print(f"Total tests (including {args.random_samples} randoms + 1 equal per blend): {num_tests}")
    
    # Run QA tests
    print("\n🚀 Starting QA tests...")
    results_df = analyzer.run_qa_tests(
        combinations=all_combinations,
        random_samples=args.random_samples,
        blend_types=['equal', 'random']
    )
    
    # Analyze results
    print("\n📊 Analyzing distributions...")
    stats = analyzer.analyze_distributions()
    
    if stats:
        print(f"✅ Success rate: {stats['success_rate']:.2%}")
        print(f"📈 Overall {analyzer.config['name']} range: {stats['property_stats']['min']:.2f} - {stats['property_stats']['max']:.2f} {analyzer.config['unit']}")
        print(f"📊 Overall {analyzer.config['name']} mean: {stats['property_stats']['mean']:.2f} {analyzer.config['unit']}")
        print(f"🎯 Polymer types covered: {stats['polymer_type_coverage']['total_unique_types']}")
        
        # Display property ranges by polymer blend type
        if 'by_polymer_count' in stats:
            print(f"\n🔬 {analyzer.config['name'].upper()} RANGES BY POLYMER BLEND TYPE:")
            print("=" * 50)
            for n_polymers, poly_stats in stats['by_polymer_count'].items():
                print(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):")
                print(f"    📊 Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} {analyzer.config['unit']}")
                print(f"    📈 Mean: {poly_stats['mean']:.2f} {analyzer.config['unit']}")
                print(f"    📉 Median: {poly_stats['median']:.2f} {analyzer.config['unit']}")
                print()
    else:
        print("❌ No successful predictions to analyze")
        return
    
    # Create plots
    print("\n📈 Creating plots...")
    analyzer.create_plots()
    
    # Generate report
    print("\n📝 Generating report...")
    report_path = analyzer.generate_report()
    
    print(f"\n✅ QA analysis complete!")
    print(f"📊 Results saved to: qa_{args.property}_results_*.csv")
    print(f"📈 Plots saved to: qa_plots/")
    print(f"📝 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 