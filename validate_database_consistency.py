#!/usr/bin/env python3
"""
Validate database consistency by comparing database predictions with command line predictions.
Uses the first 100 blends from database.csv and runs them through predict_blend_properties.py.
"""

import pandas as pd
import subprocess
import sys
import os
import re
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_database(database_path: str = 'database.csv') -> pd.DataFrame:
    """Load the generated database."""
    try:
        df = pd.read_csv(database_path)
        logger.info(f"✅ Loaded database with {len(df)} blends")
        return df
    except Exception as e:
        logger.error(f"❌ Error loading database: {e}")
        return None

def convert_blend_to_command_line(row: pd.Series) -> str:
    """Convert a database row to command line format for predict_blend_properties.py."""
    blend_parts = []
    
    # Add polymers and their concentrations
    for i in range(1, 5):  # polymer1-4, grade1-4, concentration1-4
        polymer = row.get(f'polymer{i}')
        grade = row.get(f'grade{i}')
        concentration = row.get(f'concentration{i}')
        
        if pd.notna(polymer) and pd.notna(grade) and pd.notna(concentration):
            blend_parts.extend([polymer, grade, str(concentration)])
    
    return ', '.join(blend_parts)

def run_command_line_prediction(blend_string: str, thickness: float) -> Dict[str, float]:
    """Run predict_blend_properties.py on a blend and parse the output."""
    try:
        properties = {}
        
        # Run separate commands for WVTR and OTR with their specific environmental conditions
        # Other properties don't need environmental conditions
        
        # 1. WVTR with 38°C, 90% RH
        wvtr_cmd = [
            'python', 'predict_blend_properties.py', 
            'wvtr',
            blend_string,
            f'thickness={thickness}',
            'temperature=38',
            'rh=90'
        ]
        
        wvtr_result = subprocess.run(wvtr_cmd, capture_output=True, text=True, timeout=30)
        if wvtr_result.returncode == 0:
            wvtr_match = re.search(r'• WVTR \(Actual at [\d.]+μm\) - ([\d.]+) g/m²/day', wvtr_result.stdout)
            if wvtr_match:
                properties['wvtr'] = float(wvtr_match.group(1))
            else:
                properties['wvtr'] = None
        else:
            properties['wvtr'] = None
        
        # 2. OTR with 23°C, 50% RH
        otr_cmd = [
            'python', 'predict_blend_properties.py', 
            'otr',
            blend_string,
            f'thickness={thickness}',
            'temperature=23',
            'rh=50'
        ]
        
        otr_result = subprocess.run(otr_cmd, capture_output=True, text=True, timeout=30)
        if otr_result.returncode == 0:
            otr_match = re.search(r'• Oxygen Transmission Rate \(Actual at [\d.]+μm\) - ([\d.]+) cc/m²/day', otr_result.stdout)
            if otr_match:
                properties['otr'] = float(otr_match.group(1))
            else:
                properties['otr'] = None
        else:
            properties['otr'] = None
        
        # 3. Other properties (no environmental conditions needed)
        other_cmd = [
            'python', 'predict_blend_properties.py', 
            'all',
            blend_string,
            f'thickness={thickness}'
        ]
        
        other_result = subprocess.run(other_cmd, capture_output=True, text=True, timeout=30)
        if other_result.returncode == 0:
            output = other_result.stdout
            
            # Extract other property predictions
            other_patterns = {
                'ts': r'• Tensile Strength - ([\d.]+) MPa',
                'eab': r'• Elongation at Break - ([\d.]+) %',
                'cobb': r'• Cobb Value - ([\d.]+) g/m²',
                'seal': r'• Max Seal Strength - ([\d.]+) N/15mm',
                'compost': r'• Max Disintegration - ([\d.]+)%'
            }
            
            for prop, pattern in other_patterns.items():
                match = re.search(pattern, output)
                if match:
                    properties[prop] = float(match.group(1))
                else:
                    properties[prop] = None
        else:
            # Set other properties to None if command failed
            for prop in ['ts', 'eab', 'cobb', 'seal', 'compost']:
                properties[prop] = None
        
        return properties
        
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Command timed out for blend: {blend_string}")
        return None
    except Exception as e:
        logger.error(f"❌ Error running command: {e}")
        return None

def compare_predictions(db_row: pd.Series, cli_predictions: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Compare database predictions with command line predictions."""
    comparison = {}
    
    # Property mappings
    property_mappings = {
        'wvtr': 'wvtr_prediction',
        'otr': 'otr_prediction', 
        'ts': 'ts_prediction',
        'eab': 'eab_prediction',
        'cobb': 'cobb_prediction',
        'seal': 'seal_prediction',
        'compost': 'disintegration_prediction'
    }
    
    for prop, db_col in property_mappings.items():
        db_value = db_row.get(db_col)
        cli_value = cli_predictions.get(prop)
        
        if pd.notna(db_value) and cli_value is not None:
            diff = abs(db_value - cli_value)
            
            comparison[prop] = {
                'database': db_value,
                'command_line': cli_value,
                'absolute_diff': diff
            }
        else:
            comparison[prop] = {
                'database': db_value,
                'command_line': cli_value,
                'absolute_diff': None,
            }
    
    return comparison

def validate_consistency(database_path: str = 'database.csv', max_blends: int = 100):
    """Main validation function."""
    logger.info("🔍 Starting database consistency validation...")
    
    # Load database
    df = load_database(database_path)
    if df is None:
        return
    
    # Take first max_blends
    test_df = df.head(max_blends)
    logger.info(f"📊 Testing first {len(test_df)} blends")
    
    
    results = []
    successful_comparisons = 0
    
    for idx, row in test_df.iterrows():
        logger.info(f"🔄 Processing blend {idx + 1}/{len(test_df)}")
        
        # Convert to command line format
        blend_string = convert_blend_to_command_line(row)
        thickness = row['thickness_um']
        
        logger.info(f"   Blend: {blend_string}")
        logger.info(f"   Thickness: {thickness} μm")
        
        # Run command line prediction
        cli_predictions = run_command_line_prediction(blend_string, thickness)
        
        if cli_predictions is None:
            logger.error(f"   ❌ Failed to get command line predictions")
            continue
        
        # Compare predictions
        comparison = compare_predictions(row, cli_predictions)
        
        # Check if comparison was successful
        valid_comparisons = sum(1 for prop in comparison.values() 
                              if prop['absolute_diff'] is not None)
        
        if valid_comparisons > 0:
            successful_comparisons += 1
            logger.info(f"   ✅ Successfully compared {valid_comparisons} properties")
        else:
            logger.error(f"   ❌ No valid comparisons")
        
        # Store results
        result = {
            'blend_index': idx,
            'blend_string': blend_string,
            'thickness_um': thickness,
            'comparison': comparison,
            'valid_properties': valid_comparisons
        }
        results.append(result)
    
    # Summary statistics
    logger.info("\n📊 VALIDATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"✅ Successful comparisons: {successful_comparisons}/{len(test_df)}")
    
    if successful_comparisons > 0:
        # Calculate average differences
        all_diffs = []
        for result in results:
            if result['valid_properties'] > 0:
                for prop, comp in result['comparison'].items():
                    if comp['absolute_diff'] is not None:
                        all_diffs.append(comp['absolute_diff'])
        
        if all_diffs:
            avg_diff = sum(all_diffs) / len(all_diffs)
            max_diff = max(all_diffs)
            min_diff = min(all_diffs)
            
            logger.info(f"📈 Average absolute difference: {avg_diff:.6f}")
            logger.info(f"📈 Maximum absolute difference: {max_diff:.6f}")
            logger.info(f"📈 Minimum absolute difference: {min_diff:.6f}")
            
            # Check for significant differences (>0.1)
            significant_diffs = [d for d in all_diffs if d > 0.1]
            if significant_diffs:
                logger.warning(f"⚠️ {len(significant_diffs)} comparisons show >0.1 absolute difference")
            else:
                logger.info("✅ All comparisons show <0.1 absolute difference - GOOD CONSISTENCY")
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_df.to_csv('validation_results.csv', index=False)
    logger.info("💾 Detailed results saved to validation_results.csv")
    
    return results

if __name__ == "__main__":
    database_path = sys.argv[1] if len(sys.argv) > 1 else 'database.csv'
    max_blends = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    validate_consistency(database_path, max_blends)
