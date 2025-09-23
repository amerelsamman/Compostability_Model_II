#!/usr/bin/env python3
"""
Clean, Simple Polymer Blend Prediction Script
Two clear sections:
1. Model Prediction (using train/modules/ - same as other properties)
2. Curve Generation (using train/modules_home/ - will become modules_home_curve/)

Usage:
  python predict_blend_properties_new.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  python predict_blend_properties_new.py compost "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  python predict_blend_properties_new.py wvtr "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
"""

import sys
import logging
import os
import numpy as np
import pandas as pd

# =============================================================================
# SECTION 1: MODEL PREDICTION (using train/modules/ - same as other properties)
# =============================================================================
from train.modules.input_parser import validate_input, load_and_validate_material_dictionary, parse_polymer_input
from train.modules.output_formatter import print_clean_summary
from train.modules.prediction_engine import predict_blend_property
from train.modules.prediction_utils import PROPERTY_CONFIGS

# =============================================================================
# SECTION 2: CURVE GENERATION (using train/modules_home/ - will become modules_home_curve/)
# =============================================================================
try:
    from train.modules_home.curve_generator import generate_compostability_curves
    CURVE_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Compostability curve generation not available: {e}")
    CURVE_GENERATION_AVAILABLE = False

try:
    from train.modules_sealing.curve_generator import generate_sealing_profile
    SEALING_CURVE_GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Sealing curve generation not available: {e}")
    SEALING_CURVE_GENERATION_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# This function is no longer needed - compostability now uses predict_single_property from prediction_engine.py

def generate_compostability_curves(compost_result):
    """
    SECTION 2: Curve generation using train/modules_home/ (will become modules_home_curve/).
    This is the ADDITIONAL functionality on top of the standard prediction.
    """
    if not CURVE_GENERATION_AVAILABLE:
        logger.warning("⚠️ Curve generation not available, returning basic prediction")
        return compost_result
    
    try:
        # Extract data from the basic prediction result
        max_L_pred = compost_result['max_L_pred']
        t0_pred = compost_result['t0_pred']
        thickness = compost_result['thickness']
        
        # Call the dedicated curve generation function
        from train.modules_home.curve_generator import generate_compostability_curves as generate_curves
        
        curve_results = generate_curves(
            max_L_pred, t0_pred, thickness,
            output_dir="test_results/predict_blend_properties",
            save_csv=True, save_plot=True
        )
        
        if curve_results is None:
            return compost_result
        
        # Enhance the result with curve data
        enhanced_result = compost_result.copy()
        enhanced_result.update(curve_results)
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"❌ Curve generation failed: {e}")
        return compost_result

def generate_sealing_profile_curves(adhesion_result, polymers, compositions, material_dict):
    """
    SECTION 2: Sealing profile generation using train/modules_sealing/.
    This is the ADDITIONAL functionality on top of the standard adhesion prediction.
    """
    if not SEALING_CURVE_GENERATION_AVAILABLE:
        logger.warning("⚠️ Sealing curve generation not available, returning basic prediction")
        return adhesion_result
    
    try:
        # Extract predicted adhesion strength from the basic prediction result
        predicted_adhesion_strength = adhesion_result['prediction']
        
        # Load seal masterdata to get real polymer properties
        import pandas as pd
        masterdata_path = 'train/data/seal/masterdata.csv'
        if not os.path.exists(masterdata_path):
            logger.warning("⚠️ Seal masterdata.csv not found, using placeholder values")
            return adhesion_result
            
        masterdata_df = pd.read_csv(masterdata_path)
        
        # Convert polymers from tuples to dictionaries for sealing profile generation
        # polymers is list of tuples: [(Material, Grade, vol_fraction), ...]
        polymer_dicts = []
        for i, (material, grade, vol_fraction) in enumerate(polymers):
            # Look up the polymer in masterdata
            polymer_data = masterdata_df[
                (masterdata_df['Materials'] == material) & 
                (masterdata_df['Polymer Grade 1'] == grade)
            ]
            
            if polymer_data.empty:
                logger.warning(f"⚠️ Polymer {material} {grade} not found in masterdata, using placeholder values")
                polymer_dict = {
                    'material': material,
                    'grade': grade,
                    'vol_fraction': vol_fraction,
                    'melt temperature': 150.0,  # Placeholder
                    'property': 10.0,  # Placeholder
                    'degradation temperature': 250.0  # Placeholder
                }
            else:
                # Use actual data from masterdata
                row = polymer_data.iloc[0]
                polymer_dict = {
                    'material': material,
                    'grade': grade,
                    'vol_fraction': vol_fraction,
                    'melt temperature': row['melt temperature'],
                    'property': row['property'],
                    'degradation temperature': row['degradation temperature']
                }
            
            polymer_dicts.append(polymer_dict)
        
        # Create blend name from polymer grades
        blend_name = "_".join([p[1] for p in polymers])  # p[1] is the grade
        
        # Call the dedicated sealing profile generation function
        curve_results = generate_sealing_profile(
            polymers=polymer_dicts,
            compositions=compositions,
            predicted_adhesion_strength=predicted_adhesion_strength,
            temperature_range=(0, 300),
            num_points=100,
            save_csv=True,
            save_plot=True,
            save_dir="test_results/predict_blend_properties",
            blend_name=blend_name
        )
        
        if curve_results is None or not curve_results.get('is_valid', False):
            logger.warning("⚠️ Sealing profile generation failed or invalid curve")
            return adhesion_result
        
        # Enhance the result with curve data
        enhanced_result = adhesion_result.copy()
        enhanced_result.update({
            'sealing_profile': curve_results,
            'curve_data': curve_results['curve_data'],
            'boundary_points': curve_results['boundary_points']
        })
        
        return enhanced_result
        
    except Exception as e:
        logger.error(f"❌ Sealing profile generation failed: {e}")
        return adhesion_result

def main():
    """Main function to run the clean blend prediction."""
    # Check for --no-errors flag
    include_errors = True
    if '--no-errors' in sys.argv:
        include_errors = False
        sys.argv.remove('--no-errors')
    
    # Validate input
    mode, polymer_input, available_env_params = validate_input()
    if mode is None:
        sys.exit(1)
    
    # Load material dictionary
    material_dict = load_and_validate_material_dictionary()
    if material_dict is None:
        return
    
    # Parse polymer input
    polymers, parsed_env_params = parse_polymer_input(polymer_input, mode)
    if polymers is None:
        return
    
    # Merge environmental parameters (command line takes precedence)
    if parsed_env_params:
        available_env_params.update(parsed_env_params)
    
    if mode == 'all':
        # Predict all properties
        results = []
        
        # All properties (including compostability) now use the same predict_blend_property function
        for prop_type in ['wvtr', 'ts', 'eab', 'cobb', 'otr', 'seal', 'compost']:
            result = predict_blend_property(prop_type, polymers, available_env_params, material_dict, include_errors=include_errors)
            if result:
                # Add curve generation for special properties
                if prop_type == 'compost':
                    enhanced_result = generate_compostability_curves(result)
                    results.append(enhanced_result)
                elif prop_type == 'seal':
                    # Add sealing profile generation for seal
                    compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                    enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                    results.append(enhanced_result)
                else:
                    results.append(result)
        
        # Print clean summary
        print_clean_summary(results)
        
        return results
        
    else:
        # Single property mode (using train/modules/ - same as other properties)
        result = predict_blend_property(mode, polymers, available_env_params, material_dict, include_errors=include_errors)
        
        if result:
            # Add curve generation for compostability if that's what was requested
            if mode == 'compost':
                enhanced_result = generate_compostability_curves(result)
                
                if 'max_biodegradation' in enhanced_result:
                    # Enhanced results with curves
                    print(f"• Max Disintegration - {enhanced_result['prediction']:.1f}%")
                    print(f"• Max Biodegradation - {enhanced_result['max_biodegradation']:.1f}%")
                    print(f"• Time to 50% (t0) - {enhanced_result['t0_pred']:.1f} days")
                    print(f"• k0 (Disintegration) - {enhanced_result['k0_disintegration']:.4f}")
                    print(f"• k0 (Biodegradation) - {enhanced_result['k0_biodegradation']:.4f}")
                else:
                    # Basic results without curves
                    print(f"• Max Disintegration - {enhanced_result['prediction']:.1f}%")
                
                return enhanced_result
            elif mode == 'seal':
                # Add sealing profile generation for seal
                compositions = [p[2] for p in polymers]  # p[2] is vol_fraction
                enhanced_result = generate_sealing_profile_curves(result, polymers, compositions, material_dict)
                
                # Print basic seal result (with temperature if available)
                config = PROPERTY_CONFIGS[enhanced_result['property_type']]
                if 'sealing_temp_pred' in enhanced_result:
                    # Dual property: seal strength + sealing temperature
                    print(f"• Max Seal Strength - {enhanced_result['prediction']:.2f} {enhanced_result['unit']}")
                    print(f"• Max Sealing Temperature - {enhanced_result['sealing_temp_pred']:.1f}°C")
                else:
                    print(f"• {enhanced_result['name']} - {enhanced_result['prediction']:.2f} {enhanced_result['unit']}")
                
                # Print sealing profile information
                if 'sealing_profile' in enhanced_result:
                    boundary_points = enhanced_result['boundary_points']
                    print(f"\n📊 Sealing Profile Generated:")
                    print(f"  • Initial sealing: {boundary_points['initial_sealing'][0]:.0f}°C, {boundary_points['initial_sealing'][1]:.1f} N/15mm")
                    print(f"  • First polymer max: {boundary_points['first_polymer_max'][0]:.0f}°C, {boundary_points['first_polymer_max'][1]:.1f} N/15mm")
                    print(f"  • Blend predicted: {boundary_points['blend_predicted'][0]:.0f}°C, {boundary_points['blend_predicted'][1]:.1f} N/15mm")
                    print(f"  • Degradation: {boundary_points['degradation'][0]:.0f}°C, {boundary_points['degradation'][1]:.1f} N/15mm")
                    print(f"  • Curve data saved to: test_results/predict_blend_properties/")
                
                return enhanced_result
            else:
                # Standard property results
                config = PROPERTY_CONFIGS[result['property_type']]
                print(f"• {config['name']} - {result['prediction']:.2f} {config['unit']}")
                return result['prediction']
        else:
            return None

if __name__ == "__main__":
    main()
