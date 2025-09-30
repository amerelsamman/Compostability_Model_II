#!/usr/bin/env python3
"""
Simple QA script for simulation database analysis
Shows property ranges by polymer and general distribution
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

def load_material_dictionary():
    """Load material dictionary for family mapping"""
    material_dict = {}
    with open('material-smiles-dictionary.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            material_dict[row['Material Grade']] = row['Material Family']
    return material_dict

def analyze_polymer_property_ranges(df, property_name, output_dir):
    """Analyze property ranges for each polymer"""
    print(f"\n=== POLYMER PROPERTY RANGES ===")
    
    # Find property columns
    property_cols = [col for col in df.columns if col.startswith('property')]
    if not property_cols:
        print("No property columns found!")
        return {}
    
    # Get polymer columns
    polymer_cols = [col for col in df.columns if col.startswith('Polymer') and 'Grade' in col]
    
    polymer_ranges = {}
    
    for prop_col in property_cols:
        print(f"\n{prop_col}:")
        
        # Check if thickness column exists (for WVTR/OTR)
        thickness_col = None
        if property_name.lower() == 'wvtr' and 'Thickness (um)' in df.columns:
            thickness_col = 'Thickness (um)'
            print("  (Separated by thickness)")
        elif property_name.lower() == 'otr' and 'Thickness (um)' in df.columns:
            thickness_col = 'Thickness (um)'
            print("  (Separated by thickness)")
        
        if thickness_col:
            # Analyze by thickness ranges
            thickness_ranges = [(0, 50), (50, 100), (100, 200), (200, 500), (500, float('inf'))]
            
            for thickness_min, thickness_max in thickness_ranges:
                if thickness_max == float('inf'):
                    thickness_mask = df[thickness_col] >= thickness_min
                    thickness_label = f"≥{thickness_min}μm"
                else:
                    thickness_mask = (df[thickness_col] >= thickness_min) & (df[thickness_col] < thickness_max)
                    thickness_label = f"{thickness_min}-{thickness_max}μm"
                
                thickness_df = df[thickness_mask]
                if len(thickness_df) == 0:
                    continue
                
                print(f"\n  Thickness {thickness_label}:")
                
                # Get all unique polymers and their property ranges for this thickness
                polymer_stats = {}
                
                for poly_col in polymer_cols:
                    for polymer in thickness_df[poly_col].dropna().unique():
                        if pd.notna(polymer):
                            # Get rows where this polymer appears in this thickness range
                            mask = (thickness_df[poly_col] == polymer)
                            values = thickness_df[mask][prop_col].dropna()
                            
                            if len(values) > 0:
                                if polymer not in polymer_stats:
                                    polymer_stats[polymer] = []
                                polymer_stats[polymer].extend(values.tolist())
                
                # Calculate ranges for each polymer in this thickness range
                for polymer, values in polymer_stats.items():
                    if values:
                        min_val = min(values)
                        max_val = max(values)
                        mean_val = np.mean(values)
                        count = len(values)
                        
                        print(f"    {polymer}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f} (n={count})")
                        
                        if polymer not in polymer_ranges:
                            polymer_ranges[polymer] = {}
                        if prop_col not in polymer_ranges[polymer]:
                            polymer_ranges[polymer][prop_col] = {}
                        
                        polymer_ranges[polymer][prop_col][thickness_label] = {
                            'min': min_val,
                            'max': max_val,
                            'mean': mean_val,
                            'count': count
                        }
        else:
            # Original analysis without thickness separation
            # Get all unique polymers and their property ranges
            polymer_stats = {}
            
            for poly_col in polymer_cols:
                for polymer in df[poly_col].dropna().unique():
                    if pd.notna(polymer):
                        # Get rows where this polymer appears
                        mask = df[poly_col] == polymer
                        values = df[mask][prop_col].dropna()
                        
                        if len(values) > 0:
                            if polymer not in polymer_stats:
                                polymer_stats[polymer] = []
                            polymer_stats[polymer].extend(values.tolist())
            
            # Calculate ranges for each polymer
            for polymer, values in polymer_stats.items():
                if values:
                    min_val = min(values)
                    max_val = max(values)
                    mean_val = np.mean(values)
                    count = len(values)
                    
                    print(f"  {polymer}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f} (n={count})")
                    
                    if polymer not in polymer_ranges:
                        polymer_ranges[polymer] = {}
                    polymer_ranges[polymer][prop_col] = {
                        'min': min_val,
                        'max': max_val,
                        'mean': mean_val,
                        'count': count
                    }
    
    return polymer_ranges

def analyze_property_distribution(df, property_name, output_dir):
    """Analyze general property distribution"""
    print(f"\n=== PROPERTY DISTRIBUTION ===")
    
    # Find property columns
    property_cols = [col for col in df.columns if col.startswith('property')]
    if not property_cols:
        print("No property columns found!")
        return {}
    
    stats = {}
    
    for prop_col in property_cols:
        values = df[prop_col].dropna()
        if len(values) > 0:
            print(f"\n{prop_col}:")
            print(f"  Count: {len(values):,}")
            print(f"  Mean: {values.mean():.2f}")
            print(f"  Std: {values.std():.2f}")
            print(f"  Min: {values.min():.2f}")
            print(f"  Max: {values.max():.2f}")
            print(f"  Median: {values.median():.2f}")
            print(f"  25th percentile: {values.quantile(0.25):.2f}")
            print(f"  75th percentile: {values.quantile(0.75):.2f}")
            
            stats[prop_col] = {
                'count': len(values),
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'median': values.median(),
                'q25': values.quantile(0.25),
                'q75': values.quantile(0.75)
            }
    
    # Create simple distribution plot
    fig, axes = plt.subplots(1, len(property_cols), figsize=(6*len(property_cols), 5))
    if len(property_cols) == 1:
        axes = [axes]
    
    for i, prop_col in enumerate(property_cols):
        values = df[prop_col].dropna()
        if len(values) > 0:
            axes[i].hist(values, bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{prop_col} Distribution')
            axes[i].set_xlabel('Property Value')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(values.mean(), color='red', linestyle='--', label=f'Mean: {values.mean():.1f}')
            axes[i].axvline(values.median(), color='green', linestyle='--', label=f'Median: {values.median():.1f}')
            axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{property_name}_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats

def generate_simple_report(property_name, output_dir, polymer_ranges, distribution_stats):
    """Generate simple QA report"""
    report_path = os.path.join(output_dir, f'{property_name}_qa_report.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"SIMPLE SIMULATION QA REPORT\n")
        f.write(f"Property: {property_name.upper()}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"=" * 50 + "\n\n")
        
        f.write(f"PROPERTY DISTRIBUTION\n")
        f.write(f"-" * 20 + "\n")
        for prop, stats in distribution_stats.items():
            f.write(f"\n{prop}:\n")
            f.write(f"  Count: {stats['count']:,}\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Std: {stats['std']:.2f}\n")
            f.write(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n")
            f.write(f"  Median: {stats['median']:.2f}\n")
            f.write(f"  IQR: [{stats['q25']:.2f}, {stats['q75']:.2f}]\n")
        
        f.write(f"\nPOLYMER PROPERTY RANGES\n")
        f.write(f"-" * 25 + "\n")
        for polymer, ranges in polymer_ranges.items():
            f.write(f"\n{polymer}:\n")
            for prop, stats in ranges.items():
                if isinstance(stats, dict) and any(isinstance(v, dict) for v in stats.values()):
                    # Thickness-separated data
                    for thickness, thickness_stats in stats.items():
                        f.write(f"  {prop} ({thickness}): min={thickness_stats['min']:.2f}, max={thickness_stats['max']:.2f}, mean={thickness_stats['mean']:.2f} (n={thickness_stats['count']})\n")
                else:
                    # Regular data
                    f.write(f"  {prop}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f} (n={stats['count']})\n")
    
    print(f"QA report saved to: {report_path}")

def detect_and_remove_outliers(df, property_name, output_dir):
    """Detect and remove outliers from the dataset"""
    print(f"\n=== OUTLIER DETECTION ===")
    
    # Find property columns
    property_cols = [col for col in df.columns if col.startswith('property')]
    if not property_cols:
        print("No property columns found!")
        return df
    
    original_count = len(df)
    outlier_count = 0
    
    for prop_col in property_cols:
        values = df[prop_col].dropna()
        if len(values) > 0:
            # Use IQR method for outlier detection
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = (values < lower_bound) | (values > upper_bound)
            outlier_count += outliers.sum()
            
            print(f"{prop_col}:")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Outliers: {outliers.sum():,} ({outliers.sum()/len(values)*100:.1f}%)")
            
            # Remove outliers
            df = df[~((df[prop_col] < lower_bound) | (df[prop_col] > upper_bound))]
    
    removed_count = original_count - len(df)
    print(f"\nOutlier removal summary:")
    print(f"  Original blends: {original_count:,}")
    print(f"  Removed outliers: {removed_count:,}")
    print(f"  Remaining blends: {len(df):,}")
    print(f"  Removal rate: {removed_count/original_count*100:.1f}%")
    
    if removed_count > 0:
        # Save cleaned dataset
        cleaned_path = data_path.replace('.csv', '_cleaned.csv')
        df.to_csv(cleaned_path, index=False)
        print(f"  Cleaned dataset saved to: {cleaned_path}")
    
    return df

def main(property_name, data_path, output_dir):
    """Main QA analysis function"""
    print(f"Starting simple QA analysis for {property_name}...")
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} blends from {data_path}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Detect and remove outliers
    df_cleaned = detect_and_remove_outliers(df, property_name, output_dir)
    
    # Run analyses on cleaned data
    polymer_ranges = analyze_polymer_property_ranges(df_cleaned, property_name, output_dir)
    distribution_stats = analyze_property_distribution(df_cleaned, property_name, output_dir)
    
    # Generate report
    generate_simple_report(property_name, output_dir, polymer_ranges, distribution_stats)
    
    print(f"\n✅ Simple QA analysis complete!")
    print(f"Files generated in {output_dir}:")
    print(f"  - {property_name}_qa_report.txt")
    print(f"  - {property_name}_distribution.png")
    if len(df_cleaned) < len(df):
        print(f"  - {property_name}_cleaned.csv (outlier-removed dataset)")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python qa_simulation_database.py <property_name> <data_path> <output_dir>")
        print("Example: python qa_simulation_database.py wvtr train/data/wvtr/polymerblends_for_ml.csv train/data/wvtr/")
        sys.exit(1)
    
    property_name = sys.argv[1]
    data_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    main(property_name, data_path, output_dir)
