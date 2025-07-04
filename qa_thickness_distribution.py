#!/usr/bin/env python3
"""
QA Script for WVTR vs. Thickness Distribution Analysis
Tests predict_unified_blend.py with several fixed blends (random ratios) over a range of thicknesses
and analyzes the distribution of WVTR predictions as a function of thickness.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

MATERIAL_DICT_PATH = 'material-smiles-dictionary.csv'
PREDICT_SCRIPT = 'predict_unified_blend.py'
TEMPERATURE = 38
RH = 90
THICKNESS_RANGE = list(range(20, 201, 20))  # 20μm to 200μm in 20μm steps (10 datapoints)
MAX_MATERIALS = 20  # Use only first 20 molecules from dictionary
BLEND_SIZES = [2]  # Only 2-polymer blends

np.random.seed(42)  # For reproducibility

def load_materials():
    df = pd.read_csv(MATERIAL_DICT_PATH)
    # Use only the first 22 molecules
    df = df.head(MAX_MATERIALS)
    return [(row['Material'].strip(), row['Grade'].strip()) for _, row in df.iterrows()]

def pick_random_blends(materials, n_blends, blend_sizes):
    blends = []
    used = set()
    
    # Ensure we have enough unique polymer types
    unique_materials = {}
    for m, g in materials:
        if m not in unique_materials:
            unique_materials[m] = g
    
    print(f"Available polymer types: {list(unique_materials.keys())}")
    
    for i in range(n_blends):
        # Pick a random blend size from the available sizes
        size = np.random.choice(blend_sizes)
        
        if len(unique_materials) < size:
            print(f"Warning: Only {len(unique_materials)} unique polymer types available, but blend size {size} requested")
            continue
            
        # Pick unique polymer types
        chosen_types = np.random.choice(list(unique_materials.keys()), size=size, replace=False)
        blend = [(m, unique_materials[m]) for m in chosen_types]
        
        # Generate random ratios
        ratios = np.random.dirichlet(np.ones(size))
        blend_info = {
            'materials': blend,
            'ratios': ratios
        }
        
        # Avoid duplicates
        blend_key = tuple(sorted([m for m, _ in blend]))
        if blend_key not in used:
            blends.append(blend_info)
            used.add(blend_key)
        else:
            # Try again with a different combination
            i -= 1
            continue
            
        if len(blends) >= n_blends:
            break
    
    return blends

def create_blend_input_string(blend):
    parts = []
    for (material, grade), ratio in zip(blend['materials'], blend['ratios']):
        parts.extend([material, grade, str(ratio)])
    return ', '.join(parts)

def run_prediction(blend_input, thickness, property_type='wvtr'):
    cmd = [
        sys.executable, PREDICT_SCRIPT, property_type, blend_input,
        f"temperature={TEMPERATURE}",
        f"rh={RH}",
        f"thickness={thickness}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if property_type == 'wvtr' and 'Predicted WVTR:' in line and 'g/m²/day' in line:
                    try:
                        value = float(line.split(':')[1].strip().split()[0])
                        return value, None
                    except Exception:
                        continue
                elif property_type == 'ts' and 'Predicted Tensile Strength:' in line and 'MPa' in line:
                    try:
                        value = float(line.split(':')[1].strip().split()[0])
                        return value, None
                    except Exception:
                        continue
            return None, f'Could not parse {property_type.upper()} value'
        else:
            return None, result.stderr
    except subprocess.TimeoutExpired:
        return None, 'Prediction timed out'
    except Exception as e:
        return None, str(e)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='WVTR/TS QA Thickness Distribution Analysis')
    parser.add_argument('--n_blends', type=int, default=10, 
                       help='Number of blends to test (default: 10)')
    parser.add_argument('--property', type=str, default='wvtr', choices=['wvtr', 'ts'],
                       help='Property to test (default: wvtr)')
    args = parser.parse_args()
    
    print(f"{args.property.upper()} QA Thickness Distribution Analysis")
    print("=" * 50)
    print(f"Using first {MAX_MATERIALS} molecules from materials dictionary")
    print(f"Testing {args.n_blends} blends with random ratios")
    
    materials = load_materials()
    if not materials:
        print("❌ No materials loaded. Please check the material dictionary file.")
        return
    print(f"✅ Loaded {len(materials)} material-grade combinations")
    blends = pick_random_blends(materials, args.n_blends, BLEND_SIZES)
    print(f"🔬 Testing {len(blends)} blends (random ratios, fixed composition)")
    results = []
    for blend_idx, blend in enumerate(blends):
        blend_input = create_blend_input_string(blend)
        blend_desc = ' | '.join([f"{m} {g} ({r:.3f})" for (m, g), r in zip(blend['materials'], blend['ratios'])])
        print(f"\nBlend {blend_idx+1}: {blend_desc}")
        for thickness in THICKNESS_RANGE:
            value, error = run_prediction(blend_input, thickness, args.property)
            results.append({
                'blend_id': blend_idx+1,
                'blend_materials': [m for m, _ in blend['materials']],
                'blend_grades': [g for _, g in blend['materials']],
                'blend_ratios': blend['ratios'].tolist(),
                'blend_input': blend_input,
                'thickness': thickness,
                args.property: value,
                'error': error,
                'temperature': TEMPERATURE,
                'rh': RH
            })
            if value is not None:
                unit = 'g/m²/day' if args.property == 'wvtr' else 'MPa'
                print(f"  Thickness {thickness}μm: {args.property.upper()} = {value:.2f} {unit}")
            else:
                print(f"  Thickness {thickness}μm: ERROR: {error}")
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df = pd.DataFrame(results)
    results_path = f'qa_{args.property}_thickness_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved to: {results_path}")
    # Plotting
    os.makedirs('qa_plots', exist_ok=True)
    plt.figure(figsize=(12, 8))
    for blend_idx, blend in enumerate(blends):
        subset = results_df[results_df['blend_id'] == blend_idx+1]
        
        # Create concise legend label with grade names and volume fractions
        grades = [g for _, g in blend['materials']]
        ratios = blend['ratios']
        # Format as "Grade1/Grade2 70/30" or "Grade1/Grade2/Grade3 50/30/20"
        legend_label = '/'.join(grades) + ' ' + '/'.join([f"{int(r*100)}" for r in ratios])
        
        plt.plot(subset['thickness'], subset[args.property], marker='o', label=legend_label)
    
    plt.xlabel('Thickness (μm)')
    unit = 'g/m²/day' if args.property == 'wvtr' else 'MPa'
    plt.ylabel(f'{args.property.upper()} ({unit})')
    plt.title(f'{args.property.upper()} vs. Thickness for Different Blends')
    plt.legend()
    plt.tight_layout()
    plot_path = f'qa_plots/{args.property}_vs_thickness.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"📈 Plot saved to: {plot_path}")

if __name__ == "__main__":
    main() 