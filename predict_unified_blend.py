#!/usr/bin/env python3
"""
Unified Polymer Blend Prediction Script
Predicts multiple properties for polymer blends using transfer learning models.

Usage:
  # All properties (including compostability)
  python predict_unified_blend.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  
  # Single property
  python predict_unified_blend.py wvtr "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  python predict_unified_blend.py compost "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5"
  
  # With environmental parameters
  python predict_unified_blend.py all "PLA, 4032D, 0.5, PBAT, Ecoworld, 0.5" temperature=25 rh=60 thickness=100

Format:
  All/WVTR: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Temperature, RH, Thickness"
  TS/EAB: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..., Thickness"
  Cobb/Compost: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..."
"""

import sys
import logging
import os
from modules.input_parser import validate_input, load_and_validate_material_dictionary, parse_polymer_input
from modules.output_formatter import (
    print_header, print_all_properties_summary, print_input_summary,
    print_single_property_header, print_single_property_results, print_all_properties_header
)
from modules.prediction_engine import predict_single_property

# Import home-compost modules
try:
    from homecompost_modules.blend_generator import generate_blend
    from homecompost_modules.plotting import generate_custom_blend_curves
    HOMECOMPOST_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Home-compost modules not available: {e}")
    HOMECOMPOST_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def predict_compostability(polymers, available_env_params):
    """
    Predict compostability using the home-compost model.
    
    Args:
        polymers: list of polymer tuples (material, grade, vol_fraction)
        available_env_params: available environmental parameters
    
    Returns:
        compostability result dict or None if failed
    """
    if not HOMECOMPOST_AVAILABLE:
        logger.error("❌ Home-compost modules not available")
        return None
    
    try:
        # Convert polymers to blend string format
        blend_parts = []
        for material, grade, vol_fraction in polymers:
            blend_parts.extend([material, grade, str(vol_fraction)])
        blend_str = ",".join(blend_parts)
        
        # Get thickness from environmental parameters (default 50 μm)
        thickness = available_env_params.get('Thickness (um)', 50) / 1000.0  # Convert to mm
        
        # Generate blend
        material_info, blend_curve = generate_blend(blend_str, actual_thickness=thickness)
        
        if not material_info or len(blend_curve) == 0:
            logger.error("❌ Failed to generate compostability curve")
            return None
        
        # Calculate key metrics
        max_disintegration = max(blend_curve)
        day_90_disintegration = blend_curve[89] if len(blend_curve) > 89 else blend_curve[-1]
        
        # Determine home-compostable status
        is_home_compostable = max_disintegration >= 90.0
        
        # Create blend label
        blend_label = " + ".join([f"{mat['polymer']} {mat['grade']} ({mat['vol_frac']:.1%})" for mat in material_info])
        
        result = {
            'property_type': 'compost',
            'name': 'Home Compostability',
            'unit': '% disintegration',
            'prediction': max_disintegration,
            'env_params': available_env_params,
            'day_90_disintegration': day_90_disintegration,
            'is_home_compostable': is_home_compostable,
            'blend_label': blend_label,
            'material_info': material_info,
            'blend_curve': blend_curve
        }
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Compostability prediction failed: {e}")
        return None

def main():
    """Main function to run the unified blend prediction."""
    # Check for --no-errors flag
    include_errors = True
    if '--no-errors' in sys.argv:
        include_errors = False
        sys.argv.remove('--no-errors')
    
    # Validate input
    mode, polymer_input, available_env_params = validate_input()
    if mode is None:
        sys.exit(1)
    
    # Print header
    print_header(mode, polymer_input)
    
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
        print_all_properties_header()
        results = []
        
        # Standard properties
        for prop_type in ['wvtr', 'ts', 'eab', 'cobb']:
            result = predict_single_property(prop_type, polymers, available_env_params, material_dict, include_errors=include_errors)
            if result:
                results.append(result)
        
        # Compostability
        if HOMECOMPOST_AVAILABLE:
            compost_result = predict_compostability(polymers, available_env_params)
            if compost_result:
                results.append(compost_result)
        else:
            logger.warning("⚠️ Skipping compostability prediction (modules not available)")
        
        # Print summary
        print_all_properties_summary(results)
        print_input_summary(polymers, available_env_params)
        
        # Generate compostability plot and CSV if available
        if HOMECOMPOST_AVAILABLE and any(r['property_type'] == 'compost' for r in results):
            try:
                # Convert polymers to blend string for plotting
                blend_parts = []
                for material, grade, vol_fraction in polymers:
                    blend_parts.extend([material, grade, str(vol_fraction)])
                blend_str = ",".join(blend_parts)
                
                # Get thickness
                thickness = available_env_params.get('Thickness (um)', 50) / 1000.0
                
                # Generate plot and CSV
                generate_custom_blend_curves([blend_str], 'blend_curve.png', actual_thickness=thickness)
                from homecompost_modules.blend_generator import generate_csv_for_single_blend
                generate_csv_for_single_blend(blend_str, 'blend_data.csv', actual_thickness=thickness)
                
                print(f"\n🌱 Compostability files generated:")
                print(f"   📊 Plot: blend_curve.png")
                print(f"   📄 Data: blend_data.csv")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate compostability files: {e}")
        
        return results
        
    elif mode == 'compost':
        # Single compostability mode
        if not HOMECOMPOST_AVAILABLE:
            logger.error("❌ Home-compost modules not available")
            return None
        
        print_single_property_header(mode)
        
        result = predict_compostability(polymers, available_env_params)
        if result:
            print(f"\n=== COMPOSTABILITY RESULTS ===")
            print(f"Blend: {result['blend_label']}")
            print(f"Max Disintegration: {result['prediction']:.1f}%")
            print(f"90-Day Disintegration: {result['day_90_disintegration']:.1f}%")
            print(f"Home Compostable: {'✅ Yes' if result['is_home_compostable'] else '❌ No'}")
            
            # Generate plot and CSV
            try:
                blend_parts = []
                for material, grade, vol_fraction in polymers:
                    blend_parts.extend([material, grade, str(vol_fraction)])
                blend_str = ",".join(blend_parts)
                thickness = available_env_params.get('Thickness (um)', 50) / 1000.0
                
                generate_custom_blend_curves([blend_str], 'blend_curve.png', actual_thickness=thickness)
                from homecompost_modules.blend_generator import generate_csv_for_single_blend
                generate_csv_for_single_blend(blend_str, 'blend_data.csv', actual_thickness=thickness)
                
                print(f"\n🌱 Files generated:")
                print(f"   📊 Plot: blend_curve.png")
                print(f"   📄 Data: blend_data.csv")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate files: {e}")
        
        return result
        
    else:
        # Single property mode
        print_single_property_header(mode)
        
        result = predict_single_property(mode, polymers, available_env_params, material_dict, include_errors=include_errors)
        print_single_property_results(result, polymers)
        
        if result:
            return result['prediction']
        else:
            return None

if __name__ == "__main__":
    main() 