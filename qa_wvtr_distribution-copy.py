#!/usr/bin/env python3
"""
QA Script for TS Distribution Analysis
Tests predict_unified_blend.py with various polymer blend combinations
and analyzes the distribution of TS (Tensile Strength) predictions.
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

class TSQAAnalyzer:
    def __init__(self, material_dict_path='material-smiles-dictionary.csv', max_materials=22):
        """Initialize the QA analyzer."""
        self.material_dict_path = material_dict_path
        self.max_materials = max_materials
        self.materials = self._load_materials()
        self.results = []
        self.fixed_env_params = {
            'thickness': 100
        }
        
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
    
    def _run_prediction(self, blend_input: str) -> Dict[str, Any]:
        """Run a single prediction using predict_unified_blend.py."""
        try:
            # Construct command
            cmd = [
                sys.executable, 'predict_unified_blend.py',
                'ts',
                blend_input,
                f"thickness={self.fixed_env_params['thickness']}"
            ]
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Parse the output to extract TS prediction
                output_lines = result.stdout.strip().split('\n')
                ts_value = None
                
                for line in output_lines:
                    if 'Predicted Tensile Strength:' in line and 'MPa' in line:
                        # Extract numeric value from "Predicted Tensile Strength: X.XX MPa"
                        try:
                            # Split by colon and then by space to get the number
                            parts = line.split(':')[1].strip().split()
                            ts_value = float(parts[0])
                            break
                        except (ValueError, IndexError):
                            continue
                
                if ts_value is not None:
                    return {
                        'success': True,
                        'ts': ts_value,
                        'output': result.stdout,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'ts': None,
                        'output': result.stdout,
                        'error': 'Could not parse TS value from output'
                    }
            else:
                return {
                    'success': False,
                    'ts': None,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'ts': None,
                'output': '',
                'error': 'Prediction timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'ts': None,
                'output': '',
                'error': str(e)
            }
    
    def run_qa_tests(self, combinations: List[List[Tuple[str, str]]], random_samples: int, 
                    blend_types: List[str] = None) -> pd.DataFrame:
        """Run comprehensive QA tests."""
        if blend_types is None:
            blend_types = ['equal', 'random']
        
        logger.info("Starting QA tests...")
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
                            'ts': prediction_result['ts'],
                            'error': prediction_result['error'],
                            'thickness': self.fixed_env_params['thickness']
                        }
                        
                        results.append(result)
                        
                        # Log progress
                        if prediction_result['success']:
                            logger.info(f"  ✓ TS: {prediction_result['ts']:.4f} MPa")
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
                        'ts': prediction_result['ts'],
                        'error': prediction_result['error'],
                        'thickness': self.fixed_env_params['thickness']
                    }
                    
                    results.append(result)
                    
                    # Log progress
                    if prediction_result['success']:
                        logger.info(f"  ✓ TS: {prediction_result['ts']:.4f} MPa")
                    else:
                        logger.warning(f"  ✗ Failed: {prediction_result['error']}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_df.to_csv(f'qa_ts_results_{timestamp}.csv', index=False)
        logger.info(f"Results saved to qa_ts_results_{timestamp}.csv")
        
        return self.results_df
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze the TS distributions."""
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
            'ts_stats': {
                'mean': successful_results['ts'].mean(),
                'std': successful_results['ts'].std(),
                'min': successful_results['ts'].min(),
                'max': successful_results['ts'].max(),
                'median': successful_results['ts'].median(),
                'q25': successful_results['ts'].quantile(0.25),
                'q75': successful_results['ts'].quantile(0.75)
            }
        }
        
        # Statistics by number of polymers
        polymer_stats = {}
        for n in successful_results['n_polymers'].unique():
            subset = successful_results[successful_results['n_polymers'] == n]
            polymer_stats[n] = {
                'count': len(subset),
                'mean': subset['ts'].mean(),
                'std': subset['ts'].std(),
                'min': subset['ts'].min(),
                'max': subset['ts'].max(),
                'median': subset['ts'].median()
            }
        stats['by_polymer_count'] = polymer_stats
        
        # Statistics by blend type
        blend_stats = {}
        for blend_type in successful_results['blend_type'].unique():
            subset = successful_results[successful_results['blend_type'] == blend_type]
            blend_stats[blend_type] = {
                'count': len(subset),
                'mean': subset['ts'].mean(),
                'std': subset['ts'].std(),
                'min': subset['ts'].min(),
                'max': subset['ts'].max(),
                'median': subset['ts'].median()
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
        
        # 1. Overall TS distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(successful_results['ts'], bins=30, kde=True, ax=ax)
        ax.set_xlabel('Tensile Strength (MPa)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Overall TS Distribution\n(Thickness={self.fixed_env_params["thickness"]}μm)')
        ax.axvline(successful_results['ts'].mean(), color='red', linestyle='--', label=f'Mean: {successful_results["ts"].mean():.2f}')
        ax.axvline(successful_results['ts'].median(), color='green', linestyle='--', label=f'Median: {successful_results["ts"].median():.2f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_ts_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. TS distribution by number of polymers
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, n_polymers in enumerate(sorted(successful_results['n_polymers'].unique())):
            if i >= len(axes):
                break
            subset = successful_results[successful_results['n_polymers'] == n_polymers]
            ax = axes[i]
            
            sns.histplot(subset['ts'], bins=20, kde=True, ax=ax)
            ax.set_xlabel('Tensile Strength (MPa)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{n_polymers}-Polymer Blends\n(n={len(subset)})')
            ax.axvline(subset['ts'].mean(), color='red', linestyle='--', alpha=0.7)
        
        # Hide unused subplots
        for i in range(len(successful_results['n_polymers'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ts_by_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plot by number of polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=successful_results, x='n_polymers', y='ts', ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel('Tensile Strength (MPa)')
        ax.set_title(f'TS Distribution by Number of Polymers\n(Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ts_boxplot_by_polymers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Box plot by blend type
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=successful_results, x='blend_type', y='ts', ax=ax)
        ax.set_xlabel('Blend Type')
        ax.set_ylabel('Tensile Strength (MPa)')
        ax.set_title(f'TS Distribution by Blend Type\n(Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ts_boxplot_by_blend_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Scatter plot: TS vs Number of Polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=successful_results, x='n_polymers', y='ts', 
                       hue='blend_type', alpha=0.6, ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel('Tensile Strength (MPa)')
        ax.set_title(f'TS vs Number of Polymers\n(Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ts_vs_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}/")
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive QA report."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'qa_ts_report_{timestamp}.txt'
        
        # Analyze distributions
        stats = self.analyze_distributions()
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TS QA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Configuration:\n")
            f.write(f"  - Fixed Thickness: {self.fixed_env_params['thickness']}μm\n")
            f.write(f"  - Total Tests: {stats.get('total_tests', 0)}\n")
            f.write(f"  - Successful Tests: {stats.get('successful_tests', 0)}\n")
            f.write(f"  - Success Rate: {stats.get('success_rate', 0):.2%}\n\n")
            
            # Add prominent TS range summary section
            if 'by_polymer_count' in stats:
                f.write("=" * 80 + "\n")
                f.write("TS RANGE SUMMARY BY POLYMER BLEND TYPE\n")
                f.write("=" * 80 + "\n\n")
                
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"🔬 {n_polymers}-POLYMER BLENDS (n={poly_stats['count']}):\n")
                    f.write(f"   📊 TS Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} MPa\n")
                    f.write(f"   📈 Mean TS: {poly_stats['mean']:.2f} MPa\n")
                    f.write(f"   📉 Median TS: {poly_stats['median']:.2f} MPa\n")
                    f.write(f"   📋 Min TS: {poly_stats['min']:.2f} MPa\n")
                    f.write(f"   📋 Max TS: {poly_stats['max']:.2f} MPa\n\n")
                
                f.write("=" * 80 + "\n\n")
            
            if 'ts_stats' in stats:
                f.write("Overall TS Statistics:\n")
                ts_stats = stats['ts_stats']
                f.write(f"  - Mean: {ts_stats['mean']:.2f} MPa\n")
                f.write(f"  - Median: {ts_stats['median']:.2f} MPa\n")
                f.write(f"  - Std Dev: {ts_stats['std']:.2f} MPa\n")
                f.write(f"  - Min: {ts_stats['min']:.2f} MPa\n")
                f.write(f"  - Max: {ts_stats['max']:.2f} MPa\n")
                f.write(f"  - Q25: {ts_stats['q25']:.2f} MPa\n")
                f.write(f"  - Q75: {ts_stats['q75']:.2f} MPa\n\n")
            
            if 'by_polymer_count' in stats:
                f.write("Detailed TS Statistics by Number of Polymers:\n")
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):\n")
                    f.write(f"    - Mean: {poly_stats['mean']:.2f} MPa\n")
                    f.write(f"    - Median: {poly_stats['median']:.2f} MPa\n")
                    f.write(f"    - Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} MPa\n\n")
            
            if 'by_blend_type' in stats:
                f.write("TS Statistics by Blend Type:\n")
                for blend_type, blend_stats in stats['by_blend_type'].items():
                    f.write(f"  {blend_type.capitalize()} Blends (n={blend_stats['count']}):\n")
                    f.write(f"    - Mean: {blend_stats['mean']:.2f} MPa\n")
                    f.write(f"    - Median: {blend_stats['median']:.2f} MPa\n")
                    f.write(f"    - Range: {blend_stats['min']:.2f} - {blend_stats['max']:.2f} MPa\n\n")
            
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
    parser = argparse.ArgumentParser(description='TS QA Distribution Analysis')
    parser.add_argument('--max_materials', type=int, default=20,
                       help='Maximum number of materials to use from dictionary (default: 20)')
    parser.add_argument('--n_polymers', type=int, action='append', choices=[1,2,3,4,5],
                       help='Number of polymers per blend to test. Can be specified multiple times. (default: 2)')
    parser.add_argument('--full_exploration', action='store_true',
                       help='Test all material combinations per polymer type (default: False)')
    parser.add_argument('--random_samples', type=int, default=1,
                       help='Number of random volume fraction samples per combination (default: 1)')
    args = parser.parse_args()
    
    if not args.n_polymers:
        args.n_polymers = [2]
    
    print("TS QA Distribution Analysis")
    print("=" * 50)
    print(f"Using first {args.max_materials} molecules from materials dictionary")
    print(f"Testing for blend types: {args.n_polymers}-polymer blends")
    print(f"Full material exploration: {args.full_exploration}")
    print(f"Random samples per combination: {args.random_samples}")
    
    # Initialize analyzer
    analyzer = TSQAAnalyzer(max_materials=args.max_materials)
    
    if len(analyzer.materials) == 0:
        print("❌ No materials loaded. Please check the material dictionary file.")
        return
    
    print(f"✅ Loaded {len(analyzer.materials)} material-grade combinations")
    print(f"🔧 Fixed environmental parameters: Thickness={analyzer.fixed_env_params['thickness']}μm")
    
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
        print(f"📈 Overall TS range: {stats['ts_stats']['min']:.2f} - {stats['ts_stats']['max']:.2f} MPa")
        print(f"📊 Overall TS mean: {stats['ts_stats']['mean']:.2f} MPa")
        print(f"🎯 Polymer types covered: {stats['polymer_type_coverage']['total_unique_types']}")
        
        # Display TS ranges by polymer blend type
        if 'by_polymer_count' in stats:
            print("\n🔬 TS RANGES BY POLYMER BLEND TYPE:")
            print("=" * 50)
            for n_polymers, poly_stats in stats['by_polymer_count'].items():
                print(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):")
                print(f"    📊 Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} MPa")
                print(f"    📈 Mean: {poly_stats['mean']:.2f} MPa")
                print(f"    📉 Median: {poly_stats['median']:.2f} MPa")
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
    print(f"📊 Results saved to: qa_ts_results_*.csv")
    print(f"📈 Plots saved to: qa_plots/")
    print(f"📝 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 