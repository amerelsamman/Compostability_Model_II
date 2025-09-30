#!/usr/bin/env python3
"""
Additive Testing Framework for Polymer Blend Property Prediction
Tests the effect of ADR4300 additive on binary polymer blends
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from typing import List, Dict, Tuple, Any
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Add the train directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from train.modules.prediction_engine import predict_blend_property
from train.modules.prediction_utils import load_material_dictionary

def get_polymer_families():
    """
    Get representative polymer grades from each family for binary blend testing.
    Returns a dictionary mapping polymer families to representative grades.
    Focus on biopolymers only (exclude petroleum-based polymers).
    """
    return {
        'PLA': 'Ingeo 4032D',
        'PBAT': 'ecoflex® F Blend C1200', 
        'PHB': 'ENMAT™ Y1000P',
        'PHA': 'PHACT A1000P',
        'PBS': 'BioPBS™ FZ91',
        'PCL': 'Capa™ 6500',
        'Bio-PE': 'I\'m green™ STN7006'
        # Excluded petroleum-based polymers: LDPE, PP, PET, PVDC, PA, EVOH
    }

def generate_binary_blend_combinations(polymer_families: Dict[str, str]) -> List[Tuple[str, str]]:
    """
    Generate all possible binary blend combinations from polymer families.
    """
    families = list(polymer_families.keys())
    combinations_list = []
    
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            combinations_list.append((families[i], families[j]))
    
    return combinations_list

def create_test_blends(polymer_families: Dict[str, str], 
                      additive_grade: str = 'ADR4300',
                      additive_concentrations: List[float] = [0.0, 0.05, 0.10, 0.15]) -> List[Dict[str, Any]]:
    """
    Create test blends with and without additive for each binary combination.
    
    Args:
        polymer_families: Dictionary mapping families to representative grades
        additive_grade: Grade of the additive to test
        additive_concentrations: List of additive concentrations to test
    
    Returns:
        List of blend configurations for testing
    """
    binary_combinations = generate_binary_blend_combinations(polymer_families)
    test_blends = []
    
    for family1, family2 in binary_combinations:
        grade1 = polymer_families[family1]
        grade2 = polymer_families[family2]
        
        # Create base binary blend (50-50 split)
        base_polymers = [
            (family1, grade1, 0.5),
            (family2, grade2, 0.5)
        ]
        
        # Test without additive
        test_blends.append({
            'blend_id': f"{family1}_{grade1}_vs_{family2}_{grade2}_no_additive",
            'family1': family1,
            'family2': family2,
            'grade1': grade1,
            'grade2': grade2,
            'polymers': base_polymers,
            'additive_concentration': 0.0,
            'has_additive': False
        })
        
        # Test with different additive concentrations
        for add_conc in additive_concentrations[1:]:  # Skip 0.0 (already tested)
            # Adjust polymer concentrations to make room for additive
            polymer1_conc = 0.5 * (1 - add_conc)
            polymer2_conc = 0.5 * (1 - add_conc)
            
            polymers_with_additive = [
                (family1, grade1, polymer1_conc),
                (family2, grade2, polymer2_conc),
                ('ADR4300', additive_grade, add_conc)
            ]
            
            test_blends.append({
                'blend_id': f"{family1}_{grade1}_vs_{family2}_{grade2}_ADR4300_{add_conc:.0%}",
                'family1': family1,
                'family2': family2,
                'grade1': grade1,
                'grade2': grade2,
                'polymers': polymers_with_additive,
                'additive_concentration': add_conc,
                'has_additive': True
            })
    
    return test_blends

def predict_properties_for_blends(test_blends: List[Dict[str, Any]], 
                                material_dict: Dict[Tuple[str, str], str],
                                properties: List[str] = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']) -> pd.DataFrame:
    """
    Predict properties for all test blends.
    
    Args:
        test_blends: List of blend configurations
        material_dict: Material dictionary for SMILES lookup
        properties: List of properties to predict
    
    Returns:
        DataFrame with predictions for all blends and properties
    """
    results = []
    
    for blend in test_blends:
        print(f"Testing blend: {blend['blend_id']}")
        
        # Set up environmental parameters (use standard conditions)
        env_params = {
            'Temperature (C)': 25.0,
            'RH (%)': 60.0,
            'Thickness (um)': 100.0
        }
        
        blend_results = {
            'blend_id': blend['blend_id'],
            'family1': blend['family1'],
            'family2': blend['family2'],
            'grade1': blend['grade1'],
            'grade2': blend['grade2'],
            'additive_concentration': blend['additive_concentration'],
            'has_additive': blend['has_additive']
        }
        
        # Predict each property
        for prop in properties:
            try:
                result = predict_blend_property(
                    property_type=prop,
                    polymers=blend['polymers'],
                    available_env_params=env_params,
                    material_dict=material_dict,
                    include_errors=True
                )
                
                if result and 'prediction' in result:
                    if isinstance(result['prediction'], dict):
                        # Handle dual properties (like TS, EAB, compost)
                        for key, value in result['prediction'].items():
                            blend_results[f"{prop}_{key}"] = value
                    else:
                        # Single property
                        blend_results[f"{prop}_prediction"] = result['prediction']
                    
                    # Add error bounds if available
                    if 'error_bounds' in result:
                        error_bounds = result['error_bounds']
                        blend_results[f"{prop}_upper_bound"] = error_bounds.get('upper_bound', np.nan)
                        blend_results[f"{prop}_lower_bound"] = error_bounds.get('lower_bound', np.nan)
                        blend_results[f"{prop}_model_error"] = error_bounds.get('model_error', np.nan)
                
                else:
                    print(f"  Warning: Failed to predict {prop} for {blend['blend_id']}")
                    blend_results[f"{prop}_prediction"] = np.nan
                    
            except Exception as e:
                print(f"  Error predicting {prop} for {blend['blend_id']}: {e}")
                blend_results[f"{prop}_prediction"] = np.nan
        
        results.append(blend_results)
    
    return pd.DataFrame(results)

def plot_additive_effects(results_df: pd.DataFrame, output_dir: str = "test_results/additive_effects"):
    """
    Create plots showing additive effects on different properties.
    Separate figures for different property types, focusing on biopolymers only.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. BARRIER PROPERTIES (WVTR, OTR, COBB, SEAL)
    barrier_props = ['wvtr', 'otr', 'cobb', 'seal']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, prop in enumerate(barrier_props):
        ax = axes[i]
        
        # Filter data for this property
        prop_data = results_df[results_df[f'{prop}_prediction'].notna()].copy()
        
        if len(prop_data) == 0:
            ax.text(0.5, 0.5, f'No data for {prop.upper()}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Create comparison plot
        no_additive = prop_data[prop_data['has_additive'] == False]
        with_additive = prop_data[prop_data['has_additive'] == True]
        
        # Plot base blends (no additive)
        ax.scatter(range(len(no_additive)), no_additive[f'{prop}_prediction'], 
                  label='No Additive', alpha=0.7, s=50, color='blue')
        
        # Plot blends with additive
        for conc in with_additive['additive_concentration'].unique():
            if conc > 0:
                add_data = with_additive[with_additive['additive_concentration'] == conc]
                ax.scatter(range(len(add_data)), add_data[f'{prop}_prediction'], 
                          label=f'ADR4300 {conc:.0%}', alpha=0.7, s=50)
        
        ax.set_title(f'{prop.upper()} Prediction Comparison (Biopolymers Only)')
        ax.set_xlabel('Blend Index')
        ax.set_ylabel(f'{prop.upper()} Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/barrier_properties_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. MECHANICAL PROPERTIES (TS, EAB) with TD/MD
    mechanical_props = ['ts', 'eab']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, prop in enumerate(mechanical_props):
        # Original property
        ax_orig = axes[i, 0]
        prop_data = results_df[results_df[f'{prop}_prediction'].notna()].copy()
        
        if len(prop_data) > 0:
            no_additive = prop_data[prop_data['has_additive'] == False]
            with_additive = prop_data[prop_data['has_additive'] == True]
            
            ax_orig.scatter(range(len(no_additive)), no_additive[f'{prop}_prediction'], 
                          label='No Additive', alpha=0.7, s=50, color='blue')
            
            for conc in with_additive['additive_concentration'].unique():
                if conc > 0:
                    add_data = with_additive[with_additive['additive_concentration'] == conc]
                    ax_orig.scatter(range(len(add_data)), add_data[f'{prop}_prediction'], 
                                  label=f'ADR4300 {conc:.0%}', alpha=0.7, s=50)
            
            ax_orig.set_title(f'{prop.upper()} Prediction Comparison (Biopolymers Only)')
            ax_orig.set_xlabel('Blend Index')
            ax_orig.set_ylabel(f'{prop.upper()} Prediction')
            ax_orig.legend()
            ax_orig.grid(True, alpha=0.3)
        
        # TD/MD ratio (mock calculation for now)
        ax_ratio = axes[i, 1]
        if len(prop_data) > 0:
            # Calculate TD/MD ratio (assuming we have both values)
            # For now, we'll use the prediction as TD and create a mock MD
            prop_data['td_md_ratio'] = prop_data[f'{prop}_prediction'] / (prop_data[f'{prop}_prediction'] * 0.8)  # Mock MD = 0.8 * TD
            
            no_additive_ratio = prop_data[prop_data['has_additive'] == False]
            with_additive_ratio = prop_data[prop_data['has_additive'] == True]
            
            ax_ratio.scatter(range(len(no_additive_ratio)), no_additive_ratio['td_md_ratio'], 
                           label='No Additive', alpha=0.7, s=50, color='blue')
            
            for conc in with_additive_ratio['additive_concentration'].unique():
                if conc > 0:
                    add_data = with_additive_ratio[with_additive_ratio['additive_concentration'] == conc]
                    ax_ratio.scatter(range(len(add_data)), add_data['td_md_ratio'], 
                                   label=f'ADR4300 {conc:.0%}', alpha=0.7, s=50)
            
            ax_ratio.set_title(f'{prop.upper()} TD/MD Ratio Comparison (Biopolymers Only)')
            ax_ratio.set_xlabel('Blend Index')
            ax_ratio.set_ylabel(f'{prop.upper()} TD/MD Ratio')
            ax_ratio.legend()
            ax_ratio.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/mechanical_properties_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. COMPOSTABILITY PROPERTIES
    compost_props = ['compost']
    
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    
    prop = 'compost'
    prop_data = results_df[results_df[f'{prop}_prediction'].notna()].copy()
    
    if len(prop_data) > 0:
        no_additive = prop_data[prop_data['has_additive'] == False]
        with_additive = prop_data[prop_data['has_additive'] == True]
        
        axes.scatter(range(len(no_additive)), no_additive[f'{prop}_prediction'], 
                    label='No Additive', alpha=0.7, s=50, color='blue')
        
        for conc in with_additive['additive_concentration'].unique():
            if conc > 0:
                add_data = with_additive[with_additive['additive_concentration'] == conc]
                axes.scatter(range(len(add_data)), add_data[f'{prop}_prediction'], 
                            label=f'ADR4300 {conc:.0%}', alpha=0.7, s=50)
        
        axes.set_title(f'{prop.upper()} Prediction Comparison (Biopolymers Only)')
        axes.set_xlabel('Blend Index')
        axes.set_ylabel(f'{prop.upper()} Prediction')
        axes.legend()
        axes.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/compostability_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. CONCENTRATION EFFECTS for all properties
    all_props = ['wvtr', 'otr', 'cobb', 'seal', 'ts', 'eab', 'compost']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    for i, prop in enumerate(all_props):
        if i >= len(axes):
            break
            
        ax = axes[i]
        prop_data = results_df[results_df[f'{prop}_prediction'].notna()].copy()
        
        if len(prop_data) == 0:
            ax.text(0.5, 0.5, f'No data for {prop.upper()}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Group by blend type and plot concentration effects
        blend_types = prop_data.groupby(['family1', 'family2'])
        
        for (f1, f2), group in blend_types:
            if len(group) > 1:  # Only plot if we have multiple concentrations
                concentrations = group['additive_concentration'].values
                predictions = group[f'{prop}_prediction'].values
                
                # Sort by concentration
                sort_idx = np.argsort(concentrations)
                concentrations = concentrations[sort_idx]
                predictions = predictions[sort_idx]
                
                ax.plot(concentrations, predictions, 'o-', 
                       label=f'{f1}-{f2}', alpha=0.7, linewidth=2, markersize=6)
        
        ax.set_title(f'{prop.upper()} vs Additive Concentration (Biopolymers Only)')
        ax.set_xlabel('ADR4300 Concentration')
        ax.set_ylabel(f'{prop.upper()} Prediction')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(all_props), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/concentration_effects.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(results_df: pd.DataFrame, output_dir: str = "test_results/additive_effects"):
    """
    Create summary statistics and save to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate summary statistics
    summary_stats = []
    
    for prop in ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']:
        prop_col = f'{prop}_prediction'
        if prop_col in results_df.columns:
            no_additive = results_df[results_df['has_additive'] == False][prop_col].dropna()
            with_additive = results_df[results_df['has_additive'] == True][prop_col].dropna()
            
            if len(no_additive) > 0 and len(with_additive) > 0:
                summary_stats.append({
                    'property': prop.upper(),
                    'no_additive_mean': no_additive.mean(),
                    'no_additive_std': no_additive.std(),
                    'with_additive_mean': with_additive.mean(),
                    'with_additive_std': with_additive.std(),
                    'mean_difference': with_additive.mean() - no_additive.mean(),
                    'percent_change': ((with_additive.mean() - no_additive.mean()) / no_additive.mean()) * 100,
                    'n_no_additive': len(no_additive),
                    'n_with_additive': len(with_additive)
                })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f'{output_dir}/additive_effects_summary.csv', index=False)
    
    # Save detailed results
    results_df.to_csv(f'{output_dir}/detailed_additive_test_results.csv', index=False)
    
    return summary_df

def main():
    """
    Main function to run the additive testing framework.
    """
    print("🧪 Starting Additive Testing Framework for ADR4300")
    print("=" * 60)
    
    # Load material dictionary
    print("Loading material dictionary...")
    material_dict = load_material_dictionary('material-smiles-dictionary.csv')
    if not material_dict:
        print("❌ Failed to load material dictionary")
        return None, None
    
    print(f"✅ Loaded {len(material_dict)} materials")
    
    # Get polymer families and create test blends
    print("Creating test blends...")
    polymer_families = get_polymer_families()
    test_blends = create_test_blends(polymer_families)
    
    print(f"✅ Created {len(test_blends)} test blends")
    print(f"   - {len(polymer_families)} polymer families")
    print(f"   - {len(generate_binary_blend_combinations(polymer_families))} binary combinations")
    print(f"   - Testing ADR4300 at concentrations: 0%, 5%, 10%, 15%")
    
    # Predict properties for all blends
    print("\n🔮 Predicting properties for all blends...")
    properties = ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']
    results_df = predict_properties_for_blends(test_blends, material_dict, properties)
    
    print(f"✅ Completed predictions for {len(results_df)} blends")
    
    # Create visualizations and summary
    print("\n📊 Creating analysis and visualizations...")
    summary_df = create_summary_statistics(results_df)
    plot_additive_effects(results_df)
    
    print("✅ Analysis complete!")
    print(f"\n📁 Results saved to: test_results/additive_effects/")
    print(f"   - detailed_additive_test_results.csv")
    print(f"   - additive_effects_summary.csv") 
    print(f"   - additive_effects_comparison.png")
    print(f"   - concentration_effects.png")
    
    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    return results_df, summary_df

if __name__ == "__main__":
    results_df, summary_df = main()
