#!/usr/bin/env python3
"""
QA Script for WVTR Distribution Analysis
Tests predict_unified_blend.py with various polymer blend combinations
and analyzes the distribution of WVTR predictions.
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class WVTRQAAnalyzer:
    def __init__(self, material_dict_path='material-smiles-dictionary.csv', max_materials=22):
        """Initialize the QA analyzer."""
        self.material_dict_path = material_dict_path
        self.max_materials = max_materials
        self.materials = self._load_materials()
        self.results = []
        self.fixed_env_params = {
            'temperature': 38,
            'rh': 90,
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
    
    def _generate_comprehensive_combinations(self, max_combinations: int = 100) -> List[List[Tuple[str, str]]]:
        """Generate comprehensive blend combinations ensuring all polymer types are represented."""
        polymer_groups = self._group_materials_by_polymer_type()
        polymer_types = list(polymer_groups.keys())
        
        logger.info(f"Found {len(polymer_types)} unique polymer types: {polymer_types}")
        
        combinations_list = []
        
        # 1. Add single polymer tests for each type (ensures all polymers are represented)
        for poly_type in polymer_types:
            for material in polymer_groups[poly_type]:
                combinations_list.append([material])
        
        # 2. Add 2-polymer blends covering different polymer type combinations
        for i, poly_type1 in enumerate(polymer_types):
            for poly_type2 in polymer_types[i+1:]:
                if len(combinations_list) < max_combinations:
                    material1 = polymer_groups[poly_type1][0]
                    material2 = polymer_groups[poly_type2][0]
                    combinations_list.append([material1, material2])
        
        # 3. Add 3-polymer blends
        for i, poly_type1 in enumerate(polymer_types):
            for j, poly_type2 in enumerate(polymer_types[i+1:], i+1):
                for poly_type3 in polymer_types[j+1:]:
                    if len(combinations_list) < max_combinations:
                        material1 = polymer_groups[poly_type1][0]
                        material2 = polymer_groups[poly_type2][0]
                        material3 = polymer_groups[poly_type3][0]
                        combinations_list.append([material1, material2, material3])
        
        # 4. Add 4-polymer blends if space allows
        for i, poly_type1 in enumerate(polymer_types):
            for j, poly_type2 in enumerate(polymer_types[i+1:], i+1):
                for k, poly_type3 in enumerate(polymer_types[j+1:], j+1):
                    for poly_type4 in polymer_types[k+1:]:
                        if len(combinations_list) < max_combinations:
                            material1 = polymer_groups[poly_type1][0]
                            material2 = polymer_groups[poly_type2][0]
                            material3 = polymer_groups[poly_type3][0]
                            material4 = polymer_groups[poly_type4][0]
                            combinations_list.append([material1, material2, material3, material4])
        
        # 5. Add 5-polymer blends if space allows
        for i, poly_type1 in enumerate(polymer_types):
            for j, poly_type2 in enumerate(polymer_types[i+1:], i+1):
                for k, poly_type3 in enumerate(polymer_types[j+1:], j+1):
                    for l, poly_type4 in enumerate(polymer_types[k+1:], k+1):
                        for poly_type5 in polymer_types[l+1:]:
                            if len(combinations_list) < max_combinations:
                                material1 = polymer_groups[poly_type1][0]
                                material2 = polymer_groups[poly_type2][0]
                                material3 = polymer_groups[poly_type3][0]
                                material4 = polymer_groups[poly_type4][0]
                                material5 = polymer_groups[poly_type5][0]
                                combinations_list.append([material1, material2, material3, material4, material5])
        
        # Limit to max_combinations
        combinations_list = combinations_list[:max_combinations]
        
        logger.info(f"Generated {len(combinations_list)} comprehensive blend combinations")
        return combinations_list
    
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
                'wvtr',
                blend_input,
                f"temperature={self.fixed_env_params['temperature']}",
                f"rh={self.fixed_env_params['rh']}",
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
                # Parse the output to extract WVTR prediction
                output_lines = result.stdout.strip().split('\n')
                wvtr_value = None
                
                for line in output_lines:
                    if 'Predicted WVTR:' in line and 'g/m²/day' in line:
                        # Extract numeric value from "Predicted WVTR: X.XX g/m²/day"
                        try:
                            # Split by colon and then by space to get the number
                            parts = line.split(':')[1].strip().split()
                            wvtr_value = float(parts[0])
                            break
                        except (ValueError, IndexError):
                            continue
                
                if wvtr_value is not None:
                    return {
                        'success': True,
                        'wvtr': wvtr_value,
                        'output': result.stdout,
                        'error': None
                    }
                else:
                    return {
                        'success': False,
                        'wvtr': None,
                        'output': result.stdout,
                        'error': 'Could not parse WVTR value from output'
                    }
            else:
                return {
                    'success': False,
                    'wvtr': None,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'wvtr': None,
                'output': '',
                'error': 'Prediction timed out'
            }
        except Exception as e:
            return {
                'success': False,
                'wvtr': None,
                'output': '',
                'error': str(e)
            }
    
    def run_qa_tests(self, max_combinations: int = 100, blend_types: List[str] = None) -> pd.DataFrame:
        """Run comprehensive QA tests."""
        if blend_types is None:
            blend_types = ['equal', 'random']
        
        logger.info("Starting QA tests...")
        logger.info(f"Fixed environmental parameters: {self.fixed_env_params}")
        
        # Generate comprehensive combinations
        material_combinations = self._generate_comprehensive_combinations(max_combinations)
        
        results = []
        total_tests = len(material_combinations) * len(blend_types)
        current_test = 0
        
        for materials in material_combinations:
            for blend_type in blend_types:
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
                    'wvtr': prediction_result['wvtr'],
                    'error': prediction_result['error'],
                    'temperature': self.fixed_env_params['temperature'],
                    'rh': self.fixed_env_params['rh'],
                    'thickness': self.fixed_env_params['thickness']
                }
                
                results.append(result)
                
                # Log progress
                if prediction_result['success']:
                    logger.info(f"  ✓ WVTR: {prediction_result['wvtr']:.4f} g/m²/day")
                else:
                    logger.warning(f"  ✗ Failed: {prediction_result['error']}")
        
        # Convert to DataFrame
        self.results_df = pd.DataFrame(results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_df.to_csv(f'qa_wvtr_results_{timestamp}.csv', index=False)
        logger.info(f"Results saved to qa_wvtr_results_{timestamp}.csv")
        
        return self.results_df
    
    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze the WVTR distributions."""
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
            'wvtr_stats': {
                'mean': successful_results['wvtr'].mean(),
                'std': successful_results['wvtr'].std(),
                'min': successful_results['wvtr'].min(),
                'max': successful_results['wvtr'].max(),
                'median': successful_results['wvtr'].median(),
                'q25': successful_results['wvtr'].quantile(0.25),
                'q75': successful_results['wvtr'].quantile(0.75)
            }
        }
        
        # Statistics by number of polymers
        polymer_stats = {}
        for n in successful_results['n_polymers'].unique():
            subset = successful_results[successful_results['n_polymers'] == n]
            polymer_stats[n] = {
                'count': len(subset),
                'mean': subset['wvtr'].mean(),
                'std': subset['wvtr'].std(),
                'min': subset['wvtr'].min(),
                'max': subset['wvtr'].max(),
                'median': subset['wvtr'].median()
            }
        stats['by_polymer_count'] = polymer_stats
        
        # Statistics by blend type
        blend_stats = {}
        for blend_type in successful_results['blend_type'].unique():
            subset = successful_results[successful_results['blend_type'] == blend_type]
            blend_stats[blend_type] = {
                'count': len(subset),
                'mean': subset['wvtr'].mean(),
                'std': subset['wvtr'].std(),
                'min': subset['wvtr'].min(),
                'max': subset['wvtr'].max(),
                'median': subset['wvtr'].median()
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
        
        # 1. Overall WVTR distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.histplot(successful_results['wvtr'], bins=30, kde=True, ax=ax)
        ax.set_xlabel('WVTR (g/m²/day)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Overall WVTR Distribution\n(T={self.fixed_env_params["temperature"]}°C, RH={self.fixed_env_params["rh"]}%, Thickness={self.fixed_env_params["thickness"]}μm)')
        ax.axvline(successful_results['wvtr'].mean(), color='red', linestyle='--', label=f'Mean: {successful_results["wvtr"].mean():.2f}')
        ax.axvline(successful_results['wvtr'].median(), color='green', linestyle='--', label=f'Median: {successful_results["wvtr"].median():.2f}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'{save_dir}/overall_wvtr_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. WVTR distribution by number of polymers
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, n_polymers in enumerate(sorted(successful_results['n_polymers'].unique())):
            if i >= len(axes):
                break
            subset = successful_results[successful_results['n_polymers'] == n_polymers]
            ax = axes[i]
            
            sns.histplot(subset['wvtr'], bins=20, kde=True, ax=ax)
            ax.set_xlabel('WVTR (g/m²/day)')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{n_polymers}-Polymer Blends\n(n={len(subset)})')
            ax.axvline(subset['wvtr'].mean(), color='red', linestyle='--', alpha=0.7)
        
        # Hide unused subplots
        for i in range(len(successful_results['n_polymers'].unique()), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/wvtr_by_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box plot by number of polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.boxplot(data=successful_results, x='n_polymers', y='wvtr', ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel('WVTR (g/m²/day)')
        ax.set_title(f'WVTR Distribution by Number of Polymers\n(T={self.fixed_env_params["temperature"]}°C, RH={self.fixed_env_params["rh"]}%, Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/wvtr_boxplot_by_polymers.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Box plot by blend type
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=successful_results, x='blend_type', y='wvtr', ax=ax)
        ax.set_xlabel('Blend Type')
        ax.set_ylabel('WVTR (g/m²/day)')
        ax.set_title(f'WVTR Distribution by Blend Type\n(T={self.fixed_env_params["temperature"]}°C, RH={self.fixed_env_params["rh"]}%, Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/wvtr_boxplot_by_blend_type.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Scatter plot: WVTR vs Number of Polymers
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(data=successful_results, x='n_polymers', y='wvtr', 
                       hue='blend_type', alpha=0.6, ax=ax)
        ax.set_xlabel('Number of Polymers')
        ax.set_ylabel('WVTR (g/m²/day)')
        ax.set_title(f'WVTR vs Number of Polymers\n(T={self.fixed_env_params["temperature"]}°C, RH={self.fixed_env_params["rh"]}%, Thickness={self.fixed_env_params["thickness"]}μm)')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/wvtr_vs_polymer_count.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_dir}/")
    
    def generate_report(self, save_path: str = None) -> str:
        """Generate a comprehensive QA report."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f'qa_wvtr_report_{timestamp}.txt'
        
        # Analyze distributions
        stats = self.analyze_distributions()
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("WVTR QA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Test Configuration:\n")
            f.write(f"  - Fixed Temperature: {self.fixed_env_params['temperature']}°C\n")
            f.write(f"  - Fixed RH: {self.fixed_env_params['rh']}%\n")
            f.write(f"  - Fixed Thickness: {self.fixed_env_params['thickness']}μm\n")
            f.write(f"  - Total Tests: {stats.get('total_tests', 0)}\n")
            f.write(f"  - Successful Tests: {stats.get('successful_tests', 0)}\n")
            f.write(f"  - Success Rate: {stats.get('success_rate', 0):.2%}\n\n")
            
            # Add prominent WVTR range summary section
            if 'by_polymer_count' in stats:
                f.write("=" * 80 + "\n")
                f.write("WVTR RANGE SUMMARY BY POLYMER BLEND TYPE\n")
                f.write("=" * 80 + "\n\n")
                
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"🔬 {n_polymers}-POLYMER BLENDS (n={poly_stats['count']}):\n")
                    f.write(f"   📊 WVTR Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} g/m²/day\n")
                    f.write(f"   📈 Mean WVTR: {poly_stats['mean']:.2f} g/m²/day\n")
                    f.write(f"   📉 Median WVTR: {poly_stats['median']:.2f} g/m²/day\n")
                    f.write(f"   📋 Min WVTR: {poly_stats['min']:.2f} g/m²/day\n")
                    f.write(f"   📋 Max WVTR: {poly_stats['max']:.2f} g/m²/day\n\n")
                
                f.write("=" * 80 + "\n\n")
            
            if 'wvtr_stats' in stats:
                f.write("Overall WVTR Statistics:\n")
                wvtr_stats = stats['wvtr_stats']
                f.write(f"  - Mean: {wvtr_stats['mean']:.2f} g/m²/day\n")
                f.write(f"  - Median: {wvtr_stats['median']:.2f} g/m²/day\n")
                f.write(f"  - Std Dev: {wvtr_stats['std']:.2f} g/m²/day\n")
                f.write(f"  - Min: {wvtr_stats['min']:.2f} g/m²/day\n")
                f.write(f"  - Max: {wvtr_stats['max']:.2f} g/m²/day\n")
                f.write(f"  - Q25: {wvtr_stats['q25']:.2f} g/m²/day\n")
                f.write(f"  - Q75: {wvtr_stats['q75']:.2f} g/m²/day\n\n")
            
            if 'by_polymer_count' in stats:
                f.write("Detailed WVTR Statistics by Number of Polymers:\n")
                for n_polymers, poly_stats in stats['by_polymer_count'].items():
                    f.write(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):\n")
                    f.write(f"    - Mean: {poly_stats['mean']:.2f} g/m²/day\n")
                    f.write(f"    - Median: {poly_stats['median']:.2f} g/m²/day\n")
                    f.write(f"    - Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} g/m²/day\n\n")
            
            if 'by_blend_type' in stats:
                f.write("WVTR Statistics by Blend Type:\n")
                for blend_type, blend_stats in stats['by_blend_type'].items():
                    f.write(f"  {blend_type.capitalize()} Blends (n={blend_stats['count']}):\n")
                    f.write(f"    - Mean: {blend_stats['mean']:.2f} g/m²/day\n")
                    f.write(f"    - Median: {blend_stats['median']:.2f} g/m²/day\n")
                    f.write(f"    - Range: {blend_stats['min']:.2f} - {blend_stats['max']:.2f} g/m²/day\n\n")
            
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
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WVTR QA Distribution Analysis')
    parser.add_argument('--max_combinations', type=int, default=100,
                       help='Maximum number of blend combinations to test (default: 100)')
    parser.add_argument('--max_materials', type=int, default=22,
                       help='Maximum number of materials to use from dictionary (default: 22)')
    args = parser.parse_args()
    
    print("WVTR QA Distribution Analysis")
    print("=" * 50)
    print(f"Using first {args.max_materials} molecules from materials dictionary")
    print(f"Testing up to {args.max_combinations} blend combinations")
    
    # Initialize analyzer
    analyzer = WVTRQAAnalyzer(max_materials=args.max_materials)
    
    if len(analyzer.materials) == 0:
        print("❌ No materials loaded. Please check the material dictionary file.")
        return
    
    print(f"✅ Loaded {len(analyzer.materials)} material-grade combinations")
    print(f"🔧 Fixed environmental parameters: T={analyzer.fixed_env_params['temperature']}°C, RH={analyzer.fixed_env_params['rh']}%, Thickness={analyzer.fixed_env_params['thickness']}μm")
    
    # Run QA tests
    print("\n🚀 Starting QA tests...")
    results_df = analyzer.run_qa_tests(
        max_combinations=args.max_combinations,  # Test user-specified combinations
        blend_types=['equal', 'random']
    )
    
    # Analyze results
    print("\n📊 Analyzing distributions...")
    stats = analyzer.analyze_distributions()
    
    if stats:
        print(f"✅ Success rate: {stats['success_rate']:.2%}")
        print(f"📈 Overall WVTR range: {stats['wvtr_stats']['min']:.2f} - {stats['wvtr_stats']['max']:.2f} g/m²/day")
        print(f"📊 Overall WVTR mean: {stats['wvtr_stats']['mean']:.2f} g/m²/day")
        print(f"🎯 Polymer types covered: {stats['polymer_type_coverage']['total_unique_types']}")
        
        # Display WVTR ranges by polymer blend type
        if 'by_polymer_count' in stats:
            print("\n🔬 WVTR RANGES BY POLYMER BLEND TYPE:")
            print("=" * 50)
            for n_polymers, poly_stats in stats['by_polymer_count'].items():
                print(f"  {n_polymers}-Polymer Blends (n={poly_stats['count']}):")
                print(f"    📊 Range: {poly_stats['min']:.2f} - {poly_stats['max']:.2f} g/m²/day")
                print(f"    📈 Mean: {poly_stats['mean']:.2f} g/m²/day")
                print(f"    📉 Median: {poly_stats['median']:.2f} g/m²/day")
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
    print(f"📊 Results saved to: qa_wvtr_results_*.csv")
    print(f"📈 Plots saved to: qa_plots/")
    print(f"📝 Report saved to: {report_path}")

if __name__ == "__main__":
    main() 