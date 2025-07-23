#!/usr/bin/env python3
"""
Working Database Generator for Polymer Blends

This script generates a polymer blend database with all property predictions
including disintegration dictionaries for compostability.
"""

import sys
import os
import pandas as pd
import numpy as np
import itertools
import random
import time
import json
from datetime import datetime
import logging
from typing import List, Tuple, Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import prediction modules
try:
    from polymer_blend_predictor import predict_property
except ImportError:
    print("❌ polymer_blend_predictor not found. Please ensure it's in the current directory.")
    sys.exit(1)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_materials():
    """Load material data from CSV file"""
    try:
        materials_df = pd.read_csv('material-smiles-dictionary.csv')
        # Filter out rows with 'Unknown' grade
        materials_df = materials_df[materials_df['Grade'] != 'Unknown']
        return materials_df
    except FileNotFoundError:
        print("❌ material-smiles-dictionary.csv not found!")
        return None

def get_polymer_groups():
    """Get polymer materials grouped by type"""
    materials_df = load_materials()
    if materials_df is None:
        return {}
    
    # Group by polymer type
    polymer_groups = {}
    for _, row in materials_df.iterrows():
        polymer_type = row['Material']
        grade = row['Grade']
        
        if polymer_type not in polymer_groups:
            polymer_groups[polymer_type] = []
        polymer_groups[polymer_type].append((polymer_type, grade))
    
    return polymer_groups

def generate_blend_samples(num_samples=10000):
    """Generate diverse polymer blend samples"""
    print(f"🔬 Generating {num_samples} polymer blend samples...")
    
    polymer_groups = get_polymer_groups()
    if not polymer_groups:
        return []
    
    unique_polymers = list(polymer_groups.keys())
    print(f"📊 Available polymers: {unique_polymers}")
    print(f"📊 Total unique polymers: {len(unique_polymers)}")
    
    samples = []
    
    # Calculate samples per blend type
    samples_per_type = {
        'binary': int(num_samples * 0.4),      # 40%
        'ternary': int(num_samples * 0.4),     # 40%
        'quaternary': int(num_samples * 0.2)   # 20%
    }
    
    print(f"\nSample distribution:")
    print(f"Binary blends: {samples_per_type['binary']:,}")
    print(f"Ternary blends: {samples_per_type['ternary']:,}")
    print(f"Quaternary blends: {samples_per_type['quaternary']:,}")
    
    # Volume fraction options for binary blends
    volume_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Volume distributions for ternary blends
    ternary_distributions = [
        (0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.3, 0.3, 0.4),
        (0.6, 0.25, 0.15), (0.25, 0.5, 0.25), (0.2, 0.4, 0.4),
        (0.7, 0.2, 0.1), (0.2, 0.6, 0.2), (0.1, 0.3, 0.6)
    ]
    
    # Volume distributions for quaternary blends
    quaternary_distributions = [
        (0.4, 0.3, 0.2, 0.1), (0.3, 0.3, 0.2, 0.2), (0.25, 0.25, 0.25, 0.25),
        (0.5, 0.25, 0.15, 0.1), (0.35, 0.35, 0.2, 0.1), (0.2, 0.3, 0.3, 0.2)
    ]
    
    # Thickness options
    thicknesses = [10, 50, 250]  # μm
    
    # Generate Binary Blends
    binary_combinations = list(itertools.combinations(unique_polymers, 2))
    samples_per_combo = max(1, samples_per_type['binary'] // len(binary_combinations))
    
    print(f"\nGenerating binary blends...")
    for polymer1, polymer2 in binary_combinations[:samples_per_type['binary']]:
        for _ in range(min(samples_per_combo, samples_per_type['binary'] - len(samples))):
            if len(samples) >= samples_per_type['binary']:
                break
                
            # Randomly select grades
            grade1 = random.choice(polymer_groups[polymer1])
            grade2 = random.choice(polymer_groups[polymer2])
            
            # Randomly select volume fractions
            vol_frac1 = random.choice(volume_fractions)
            vol_frac2 = 1.0 - vol_frac1
            
            # Randomly select thickness
            thickness = random.choice(thicknesses)
            
            samples.append({
                'materials': [grade1, grade2],
                'concentrations': [vol_frac1, vol_frac2],
                'blend_type': 'binary',
                'thickness_um': thickness,
                'temperature_c': 38,  # Changed to 38
                'humidity_percent': 90  # Changed to 90
            })
    
    # Generate Ternary Blends
    ternary_combinations = list(itertools.combinations(unique_polymers, 3))
    samples_per_combo = max(1, samples_per_type['ternary'] // len(ternary_combinations))
    
    print(f"Generating ternary blends...")
    current_ternary = len([s for s in samples if s['blend_type'] == 'ternary'])
    
    for polymer1, polymer2, polymer3 in ternary_combinations[:samples_per_type['ternary']]:
        for _ in range(min(samples_per_combo, samples_per_type['ternary'] - current_ternary)):
            if current_ternary >= samples_per_type['ternary']:
                break
                
            # Randomly select grades
            grade1 = random.choice(polymer_groups[polymer1])
            grade2 = random.choice(polymer_groups[polymer2])
            grade3 = random.choice(polymer_groups[polymer3])
            
            # Randomly select volume distribution
            vol_dist = random.choice(ternary_distributions)
            
            # Randomly select thickness
            thickness = random.choice(thicknesses)
            
            samples.append({
                'materials': [grade1, grade2, grade3],
                'concentrations': list(vol_dist),
                'blend_type': 'ternary',
                'thickness_um': thickness,
                'temperature_c': 38,  # Changed to 38
                'humidity_percent': 90  # Changed to 90
            })
            current_ternary += 1
    
    # Generate Quaternary Blends
    quaternary_combinations = list(itertools.combinations(unique_polymers, 4))
    samples_per_combo = max(1, samples_per_type['quaternary'] // len(quaternary_combinations))
    
    print(f"Generating quaternary blends...")
    current_quaternary = len([s for s in samples if s['blend_type'] == 'quaternary'])
    
    for polymer1, polymer2, polymer3, polymer4 in quaternary_combinations[:samples_per_type['quaternary']]:
        for _ in range(min(samples_per_combo, samples_per_type['quaternary'] - current_quaternary)):
            if current_quaternary >= samples_per_type['quaternary']:
                break
                
            # Randomly select grades
            grade1 = random.choice(polymer_groups[polymer1])
            grade2 = random.choice(polymer_groups[polymer2])
            grade3 = random.choice(polymer_groups[polymer3])
            grade4 = random.choice(polymer_groups[polymer4])
            
            # Randomly select volume distribution
            vol_dist = random.choice(quaternary_distributions)
            
            # Randomly select thickness
            thickness = random.choice(thicknesses)
            
            samples.append({
                'materials': [grade1, grade2, grade3, grade4],
                'concentrations': list(vol_dist),
                'blend_type': 'quaternary',
                'thickness_um': thickness,
                'temperature_c': 38,  # Changed to 38
                'humidity_percent': 90  # Changed to 90
            })
            current_quaternary += 1
    
    print(f"✅ Generated {len(samples)} samples")
    return samples

def predict_properties_for_sample(sample, sample_num, total_samples):
    """Predict all properties for a single sample"""
    # Create base results
    results = {
        'blend_type': sample['blend_type'],
        'thickness_um': sample['thickness_um'],
        'temperature_c': sample['temperature_c'],
        'humidity_percent': sample['humidity_percent']
    }
    
    # Add material information
    for i, (material, grade) in enumerate(sample['materials']):
        results[f'polymer{i+1}'] = material
        results[f'grade{i+1}'] = grade
        results[f'concentration{i+1}'] = sample['concentrations'][i]
    
    # Fill in empty slots for quaternary blends
    for i in range(len(sample['materials']), 4):
        results[f'polymer{i+1}'] = None
        results[f'grade{i+1}'] = None
        results[f'concentration{i+1}'] = None
    
    # Create polymers list in the correct format for predict_property
    polymers = []
    for (material, grade), concentration in zip(sample['materials'], sample['concentrations']):
        polymers.append((material, grade, concentration))
    
    # Initialize compost-specific columns
    results['compost_day_30'] = None
    results['compost_day_90'] = None
    results['compost_day_180'] = None
    results['compost_max'] = None
    
    # Predict each property
    properties = ['wvtr', 'ts', 'eab', 'cobb', 'compost']
    
    for prop in properties:
        try:
            if prop == 'compost':
                # Use direct homecompost functionality
                try:
                    from homecompost_modules.blend_generator import generate_csv_for_single_blend
                    
                    # Create blend string in the format expected by homecompost
                    blend_parts = []
                    for (material, grade), concentration in zip(sample['materials'], sample['concentrations']):
                        blend_parts.extend([material, grade, str(concentration)])
                    blend_str = ",".join(blend_parts)
                    
                    # Convert thickness from um to mm
                    thickness_mm = sample['thickness_um'] / 1000.0
                    
                    # Generate disintegration profile
                    time_prediction_dict = generate_csv_for_single_blend(
                        blend_str, 
                        output_path=None, 
                        actual_thickness=thickness_mm
                    )
                    
                    if time_prediction_dict:
                        # Get final disintegration at day 180 (standard composting period)
                        final_disintegration = time_prediction_dict.get(180, 0.0)
                        
                        results[f'{prop}_prediction'] = final_disintegration
                        results[f'{prop}_unit'] = '% disintegration'
                        
                        # Extract specific day values
                        results['compost_day_30'] = time_prediction_dict.get(30, 0.0)
                        results['compost_day_90'] = time_prediction_dict.get(90, 0.0)
                        results['compost_day_180'] = time_prediction_dict.get(180, 0.0)
                        
                        # Calculate max disintegration
                        results['compost_max'] = max(time_prediction_dict.values()) if time_prediction_dict else 0.0
                    else:
                        results[f'{prop}_prediction'] = None
                        results[f'{prop}_unit'] = None
                        
                except ImportError:
                    logger.warning("Homecompost modules not available, using fallback compost prediction")
                    # Fallback to existing compost prediction
                    prediction_result = predict_property(
                        polymers=polymers,
                        property_name=prop,
                        thickness=sample['thickness_um']
                    )
                    
                    if prediction_result.get('success', False):
                        results[f'{prop}_prediction'] = prediction_result.get('prediction', None)
                        results[f'{prop}_unit'] = prediction_result.get('unit', '% disintegration')
                        
                        # Extract time series if available
                        time_series = None
                        if 'full_time_series' in prediction_result:
                            time_series = prediction_result['full_time_series']
                        elif 'time_series' in prediction_result:
                            time_series = prediction_result['time_series']
                        
                        if time_series and isinstance(time_series, dict):
                            results['compost_day_30'] = time_series.get(30, 0.0)
                            results['compost_day_90'] = time_series.get(90, 0.0)
                            results['compost_day_180'] = time_series.get(180, 0.0)
                            results['compost_max'] = max(time_series.values()) if time_series else 0.0
                    else:
                        results[f'{prop}_prediction'] = None
                        results[f'{prop}_unit'] = None
                    
            elif prop == 'wvtr':
                # WVTR needs temperature, RH, and thickness
                prediction_result = predict_property(
                    polymers=polymers,
                    property_name=prop,
                    temperature=sample['temperature_c'],
                    rh=sample['humidity_percent'],
                    thickness=sample['thickness_um']
                )
                
                if prediction_result.get('success', False):
                    results[f'{prop}_prediction'] = prediction_result.get('prediction', None)
                    results[f'{prop}_unit'] = prediction_result.get('unit', 'g/m²/day')
                else:
                    results[f'{prop}_prediction'] = None
                    results[f'{prop}_unit'] = None
                    
            elif prop in ['ts', 'eab']:
                # TS and EAB need thickness
                prediction_result = predict_property(
                    polymers=polymers,
                    property_name=prop,
                    thickness=sample['thickness_um']
                )
                
                if prediction_result.get('success', False):
                    results[f'{prop}_prediction'] = prediction_result.get('prediction', None)
                    results[f'{prop}_unit'] = prediction_result.get('unit', 'MPa' if prop == 'ts' else '%')
                else:
                    results[f'{prop}_prediction'] = None
                    results[f'{prop}_unit'] = None
                    
            elif prop == 'cobb':
                # COBB doesn't need environmental parameters
                prediction_result = predict_property(
                    polymers=polymers,
                    property_name=prop
                )
                
                if prediction_result.get('success', False):
                    results[f'{prop}_prediction'] = prediction_result.get('prediction', None)
                    results[f'{prop}_unit'] = prediction_result.get('unit', 'g/m²')
                else:
                    results[f'{prop}_prediction'] = None
                    results[f'{prop}_unit'] = None
                    
        except Exception as e:
            logger.warning(f"Failed to predict {prop} for sample {sample_num}: {str(e)}")
            results[f'{prop}_prediction'] = None
            results[f'{prop}_unit'] = None
    
    return results

def generate_database(num_samples=10000, batch_size=50):
    """Generate complete database with all predictions"""
    print(f"🚀 Starting database generation for {num_samples} samples...")
    print("=" * 60)
    
    # Generate samples
    samples = generate_blend_samples(num_samples)
    if not samples:
        print("❌ No samples generated!")
        return
    
    # Process samples in batches
    all_results = []
    total_batches = (len(samples) + batch_size - 1) // batch_size
    
    start_time = time.time()
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(samples))
        batch_samples = samples[start_idx:end_idx]
        
        print(f"\n📦 Processing batch {batch_num + 1}/{total_batches} (samples {start_idx + 1}-{end_idx})")
        
        batch_results = []
        for i, sample in enumerate(batch_samples):
            sample_num = start_idx + i + 1
            result = predict_properties_for_sample(sample, sample_num, len(samples))
            batch_results.append(result)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (len(samples) - sample_num) * (elapsed / sample_num) if sample_num > 0 else 0
                print(f"  Sample {sample_num}/{len(samples)}")
                print(f"  ⏱️  Elapsed: {elapsed:.1f}s, Est. remaining: {remaining:.1f}s")
        
        all_results.extend(batch_results)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Randomize the order of the database for better diversity
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Reorder columns according to specified structure
    column_order = [
        'blend_type',
        'polymer1', 'grade1', 'concentration1',
        'polymer2', 'grade2', 'concentration2',
        'polymer3', 'grade3', 'concentration3',
        'polymer4', 'grade4', 'concentration4',
        'thickness_um', 'temperature_c', 'humidity_percent',
        'wvtr_prediction', 'wvtr_unit',
        'ts_prediction', 'ts_unit',
        'eab_prediction', 'eab_unit',
        'cobb_prediction', 'cobb_unit',
        'compost_prediction', 'compost_unit',
        'compost_day_30', 'compost_day_90', 'compost_day_180', 'compost_max'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Save database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"polymer_database_{timestamp}.csv"
    
    print(f"\n💾 Saving database...")
    df.to_csv(filename, index=False)
    print(f"✅ Saved database to: {filename}")
    
    # Print summary
    print(f"\n📊 Database Summary:")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"Binary blends: {len(df[df['blend_type'] == 'binary']):,}")
    print(f"Ternary blends: {len(df[df['blend_type'] == 'ternary']):,}")
    print(f"Quaternary blends: {len(df[df['blend_type'] == 'quaternary']):,}")
    
    # Calculate success rates
    print(f"\n📈 Prediction Success Rates:")
    for prop in ['wvtr', 'ts', 'eab', 'cobb', 'compost']:
        if f'{prop}_prediction' in df.columns:
            success_count = df[f'{prop}_prediction'].notna().sum()
            success_rate = (success_count / len(df)) * 100
            print(f"{prop.upper()}: {success_rate:.1f}% ({success_count:,}/{len(df):,})")
    
    # Compost-specific statistics
    if 'compost_day_30' in df.columns:
        print(f"\n📊 Compost Statistics:")
        print(f"Day 30 - Mean: {df['compost_day_30'].mean():.1f}%, Max: {df['compost_day_30'].max():.1f}%")
        print(f"Day 90 - Mean: {df['compost_day_90'].mean():.1f}%, Max: {df['compost_day_90'].max():.1f}%")
        print(f"Day 180 - Mean: {df['compost_day_180'].mean():.1f}%, Max: {df['compost_day_180'].max():.1f}%")
        print(f"Max Disintegration - Mean: {df['compost_max'].mean():.1f}%, Max: {df['compost_max'].max():.1f}%")
    
    total_time = time.time() - start_time
    print(f"\n🎉 Database Generation Finished!")
    print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"📁 Output file: {filename}")
    print(f"📊 Database contains {len(df):,} samples with all property predictions")
    
    # Show sample of successful predictions
    print(f"\n🔍 Sample of successful predictions:")
    successful_samples = df[df['wvtr_prediction'].notna() | df['ts_prediction'].notna() | 
                           df['eab_prediction'].notna() | df['cobb_prediction'].notna() | 
                           df['compost_prediction'].notna()].head(3)
    
    for idx, row in successful_samples.iterrows():
        print(f"\nSample {idx + 1}:")
        print(f"  Blend: {row['polymer1']} ({row['concentration1']:.2f}) + {row['polymer2']} ({row['concentration2']:.2f})")
        if pd.notna(row.get('polymer3')):
            print(f"         + {row['polymer3']} ({row['concentration3']:.2f})")
        if pd.notna(row.get('polymer4')):
            print(f"         + {row['polymer4']} ({row['concentration4']:.2f})")
        
        print(f"  Thickness: {row['thickness_um']} μm")
        
        for prop in ['wvtr', 'ts', 'eab', 'cobb', 'compost']:
            if f'{prop}_prediction' in row and pd.notna(row[f'{prop}_prediction']):
                unit = row.get(f'{prop}_unit', '')
                print(f"  {prop.upper()}: {row[f'{prop}_prediction']:.2f} {unit}")
        
        if 'compost_day_30' in row and pd.notna(row['compost_day_30']):
            print(f"  Compost Day 30/90/180: {row['compost_day_30']:.1f}% / {row['compost_day_90']:.1f}% / {row['compost_day_180']:.1f}%")

if __name__ == "__main__":
    # Parse command line arguments
    num_samples = 10000
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print("❌ Invalid number of samples. Using default 10,000.")
    
    generate_database(num_samples) 