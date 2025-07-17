#!/usr/bin/env python3
"""
Systematic QA Script for Polymer Blend Model Testing
Calls predict_unified_blend.py to test various blends and neat polymers,
then populates qa.csv with the results.

This script follows a systematic approach:
1. Configurable number of blend mixtures (binary and ternary)
2. For each blend: equal vol fractions + 2 specific uneven variations
3. Environmental variations (T, RH, thickness) only on equal blends
4. Test neat polymers for each component
5. Uses materials from material-smiles-dictionary-qa.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import time
import itertools
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qa.csv')
MATERIAL_DICT_PATH = os.path.join(ROOT_DIR, 'material-smiles-dictionary-qa.csv')

# Add root directory to path for imports
sys.path.append(ROOT_DIR)

# Default environmental conditions
DEFAULT_THICKNESS = 50  # μm
DEFAULT_TEMPERATURE = 25  # °C
DEFAULT_RH = 50  # %

# Variation values
THICKNESS_VALUES = [50, 20]  # μm
TEMPERATURE_VALUES = [37, 50]  # °C
RH_VALUES = [70, 90]  # %

def load_material_dictionary() -> List[Tuple[str, str]]:
    """Load all materials from the QA dictionary."""
    try:
        df = pd.read_csv(MATERIAL_DICT_PATH)
        materials = []
        for _, row in df.iterrows():
            materials.append((row['Material'].strip(), row['Grade'].strip()))
        logger.info(f"✅ Loaded {len(materials)} materials from QA dictionary")
        return materials
    except Exception as e:
        logger.error(f"❌ Error loading material dictionary: {e}")
        return []

def create_blend_mixtures(materials: List[Tuple[str, str]], num_binary: int = 1, num_ternary: int = 1, num_quaternary: int = 0) -> List[Dict[str, Any]]:
    """
    Create systematic blend mixtures with reproducible results.
    Uses fixed seed to ensure same blends are generated each time.
    Randomly selects materials from the dictionary.
    Organizes by type: all binary first, then ternary, then quaternary.
    """
    # Set seed for reproducible blend generation
    np.random.seed(42)
    
    blend_mixtures = []
    used_material_indices = set()
    
    # Binary blends (all first)
    for i in range(num_binary):
        # Randomly select 2 materials (avoiding already used ones)
        available_materials = [j for j in range(len(materials)) if j not in used_material_indices]
        if len(available_materials) < 2:
            # If not enough unused materials, reset and use any materials
            available_materials = list(range(len(materials)))
            used_material_indices.clear()
        
        selected_materials = np.random.choice(available_materials, 2, replace=False)
        used_material_indices.update(selected_materials)
        
        mat1 = materials[selected_materials[0]]
        mat2 = materials[selected_materials[1]]
        
        blend_mixtures.append({
            'type': 'binary',
            'materials': [mat1, mat2],
            'equal_vols': [0.5, 0.5],
            'uneven1_vols': [0.2, 0.8],
            'uneven2_vols': [0.8, 0.2],
            'name': f"Binary_{i+1}_{mat1[0]}_{mat1[1]}-{mat2[0]}_{mat2[1]}"
        })
    
    # Ternary blends (all second)
    for i in range(num_ternary):
        # Randomly select 3 materials (avoiding already used ones)
        available_materials = [j for j in range(len(materials)) if j not in used_material_indices]
        if len(available_materials) < 3:
            # If not enough unused materials, reset and use any materials
            available_materials = list(range(len(materials)))
            used_material_indices.clear()
        
        selected_materials = np.random.choice(available_materials, 3, replace=False)
        used_material_indices.update(selected_materials)
        
        mat1 = materials[selected_materials[0]]
        mat2 = materials[selected_materials[1]]
        mat3 = materials[selected_materials[2]]
        
        blend_mixtures.append({
            'type': 'ternary',
            'materials': [mat1, mat2, mat3],
            'equal_vols': [0.33, 0.33, 0.34],  # Slightly adjust to sum to 1.0
            'uneven1_vols': [0.1, 0.3, 0.6],
            'uneven2_vols': [0.6, 0.3, 0.1],
            'name': f"Ternary_{i+1}_{mat1[0]}_{mat1[1]}-{mat2[0]}_{mat2[1]}-{mat3[0]}_{mat3[1]}"
        })
    
    # Quaternary blends (all third)
    for i in range(num_quaternary):
        # Randomly select 4 materials (avoiding already used ones)
        available_materials = [j for j in range(len(materials)) if j not in used_material_indices]
        if len(available_materials) < 4:
            # If not enough unused materials, reset and use any materials
            available_materials = list(range(len(materials)))
            used_material_indices.clear()
        
        selected_materials = np.random.choice(available_materials, 4, replace=False)
        used_material_indices.update(selected_materials)
        
        mat1 = materials[selected_materials[0]]
        mat2 = materials[selected_materials[1]]
        mat3 = materials[selected_materials[2]]
        mat4 = materials[selected_materials[3]]
        
        blend_mixtures.append({
            'type': 'quaternary',
            'materials': [mat1, mat2, mat3, mat4],
            'equal_vols': [0.25, 0.25, 0.25, 0.25],
            'uneven1_vols': [0.4, 0.4, 0.1, 0.1],
            'uneven2_vols': [0.1, 0.1, 0.4, 0.4],
            'name': f"Quaternary_{i+1}_{mat1[0]}_{mat1[1]}-{mat2[0]}_{mat2[1]}-{mat3[0]}_{mat3[1]}-{mat4[0]}_{mat4[1]}"
        })
    

    
    logger.info(f"✅ Created {len(blend_mixtures)} blend mixtures:")
    logger.info(f"   - Binary: {sum(1 for b in blend_mixtures if b['type'] == 'binary')}")
    logger.info(f"   - Ternary: {sum(1 for b in blend_mixtures if b['type'] == 'ternary')}")
    
    return blend_mixtures

def create_blend_string(materials: List[Tuple[str, str]], volumes: List[float]) -> str:
    """Create blend string for predict_unified_blend.py."""
    parts = []
    for (material, grade), vol in zip(materials, volumes):
        parts.extend([material, grade, str(vol)])
    return ", ".join(parts)

def create_blend_description(materials: List[Tuple[str, str]], volumes: List[float], thickness: int, temperature: int, rh: int) -> str:
    """Create the exact command line argument format for predict_unified_blend.py."""
    # Create the blend string in the format: "Material1, Grade1, vol_fraction1, Material2, Grade2, vol_fraction2, ..."
    blend_parts = []
    for (material, grade), vol in zip(materials, volumes):
        blend_parts.extend([material, grade, str(vol)])
    
    blend_str = ", ".join(blend_parts)
    
    # Add environmental parameters
    env_params = f" temperature={temperature} rh={rh} thickness={thickness}"
    
    return f'"{blend_str}"{env_params}'

def run_prediction(blend_input: str, temperature: int, rh: int, thickness: int, property_type: str = 'all') -> Dict[str, Any]:
    """
    Run predictions using the modules directly (like predict_unified_blend.py does).
    
    Args:
        blend_input: Blend string in format "Material1, Grade1, vol_fraction1, ..."
        temperature: Temperature in °C
        rh: Relative humidity in %
        thickness: Thickness in μm
        property_type: Property to predict ('all', 'wvtr', 'ts', 'eab', 'cobb', 'compost')
    
    Returns:
        Dictionary with prediction results
    """
    try:
        # Change to root directory to ensure all files are found
        original_cwd = os.getcwd()
        os.chdir(ROOT_DIR)
        
        # Import modules (same as predict_unified_blend.py)
        from modules.input_parser import load_and_validate_material_dictionary, parse_polymer_input
        from modules.prediction_engine import predict_single_property
        from modules.prediction_utils import PROPERTY_CONFIGS
        
        # Import home-compost modules
        try:
            from homecompost_modules.blend_generator import generate_blend
            HOMECOMPOST_AVAILABLE = True
        except ImportError as e:
            logging.warning(f"Home-compost modules not available: {e}")
            HOMECOMPOST_AVAILABLE = False
        
        # Load material dictionary
        material_dict = load_and_validate_material_dictionary()
        if material_dict is None:
            os.chdir(original_cwd)
            return {}
        
        # Parse polymer input
        polymers, parsed_env_params = parse_polymer_input(blend_input, property_type)
        if polymers is None:
            os.chdir(original_cwd)
            return {}
        
        # Set up environmental parameters
        available_env_params = {
            'Temperature (C)': temperature,
            'RH (%)': rh,
            'Thickness (um)': thickness
        }
        
        # Merge any parsed environmental parameters
        if parsed_env_params:
            available_env_params.update(parsed_env_params)
        
        predictions = {}
        
        if property_type == 'all':
            # Predict all properties
            for prop_type in ['wvtr', 'ts', 'eab', 'cobb']:
                result = predict_single_property(prop_type, polymers, available_env_params, material_dict, include_errors=False)
                if result:
                    predictions[prop_type] = result['prediction']
            
            # Compostability
            if HOMECOMPOST_AVAILABLE:
                try:
                    # Convert polymers to blend string format
                    blend_parts = []
                    for material, grade, vol_fraction in polymers:
                        blend_parts.extend([material, grade, str(vol_fraction)])
                    blend_str = ",".join(blend_parts)
                    
                    # Get thickness from environmental parameters (default 50 μm)
                    thickness_mm = available_env_params.get('Thickness (um)', 50) / 1000.0  # Convert to mm
                    
                    # Generate blend (suppress verbose output)
                    import contextlib
                    import io
                    
                    # Capture and suppress verbose output
                    with contextlib.redirect_stdout(io.StringIO()):
                        material_info, blend_curve = generate_blend(blend_str, actual_thickness=thickness_mm)
                    
                    if material_info and len(blend_curve) > 0:
                        max_disintegration = max(blend_curve)
                        predictions['disintegration'] = max_disintegration
                except Exception as e:
                    logger.error(f"❌ Compostability prediction failed: {e}")
        
        else:
            # Single property
            result = predict_single_property(property_type, polymers, available_env_params, material_dict, include_errors=False)
            if result:
                predictions[property_type] = result['prediction']
        
        # Change back to original directory
        os.chdir(original_cwd)
        return predictions
        
    except Exception as e:
        logger.error(f"❌ Error running prediction: {e}")
        # Make sure we change back to original directory even on error
        try:
            os.chdir(original_cwd)
        except:
            pass
        return {}

def test_neat_polymers(materials: List[Tuple[str, str]], temperature: int, rh: int, thickness: int) -> List[Dict[str, Any]]:
    """
    Test individual neat polymers from a blend.
    
    Args:
        materials: List of (material, grade) tuples
        temperature: Temperature in °C
        rh: Relative humidity in %
        thickness: Thickness in μm
    
    Returns:
        List of prediction results for each neat polymer
    """
    neat_results = []
    
    for material, grade in materials:
        # Create neat polymer blend string (100% of one polymer)
        neat_blend = f"{material}, {grade}, 1.0"
        
        logger.info(f"Testing neat polymer: {material} {grade}")
        predictions = run_prediction(neat_blend, temperature, rh, thickness)
        
        neat_results.append({
            'material': material,
            'grade': grade,
            'predictions': predictions
        })
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.5)
    
    return neat_results

def create_qa_dataframe() -> pd.DataFrame:
    """Create the QA DataFrame with the correct column structure."""
    columns = [
        'Blend', ' Thickness (um)', 'Temperature', 'RH',
        'Blend WVTR', 'Blend Cobb', 'Blend Tensile Strength', 'Blend Elongation at Break', 'Blend Disintegration'
    ]
    
    # Add columns for up to 4 neat polymers
    for i in range(1, 5):
        columns.extend([
            f'Neat{i} WVTR', f'Neat{i} Cobb', f'Neat{i} Tensile Strength', 
            f'Neat{i} Elongation at Break', f'Neat{i} Disintegration'
        ])
    
    return pd.DataFrame(columns=columns)

def populate_qa_csv(num_binary: int = 1, num_ternary: int = 1, num_quaternary: int = 0):
    """Main function to populate the QA CSV with systematic blend and neat polymer predictions."""
    logger.info("🚀 Starting systematic QA testing for polymer blends")
    
    # Load materials
    materials = load_material_dictionary()
    if not materials:
        logger.error("❌ Cannot proceed without materials")
        return
    
    # Create blend mixtures
    blend_mixtures = create_blend_mixtures(materials, num_binary, num_ternary, num_quaternary)
    
    # Create or load QA DataFrame
    if os.path.exists(QA_CSV_PATH):
        qa_df = pd.read_csv(QA_CSV_PATH)
        logger.info(f"📁 Loaded existing QA CSV with {len(qa_df)} rows")
    else:
        qa_df = create_qa_dataframe()
        logger.info("📁 Created new QA DataFrame")
    
    # Track progress
    total_blends = len(blend_mixtures)
    total_tests = total_blends * 7  # 7 tests per blend
    
    current_test = 0
    
    for blend_idx, blend in enumerate(blend_mixtures):
        logger.info(f"\n🔬 Testing blend mixture {blend_idx + 1}/{total_blends}: {blend['name']}")
        logger.info(f"   Type: {blend['type']}, Materials: {len(blend['materials'])}")
        
        # Test 1: Equal blend mixture with default conditions
        current_test += 1
        logger.info(f"\n     Test {current_test}/{total_tests}: Equal blend (default conditions)")
        equal_blend_str = create_blend_string(blend['materials'], blend['equal_vols'])
        blend_predictions = run_prediction(equal_blend_str, DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        # Test neat polymers with same environmental conditions
        logger.info("     🔄 Testing neat polymers with default conditions...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        # Create row data
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['equal_vols'], DEFAULT_THICKNESS, DEFAULT_TEMPERATURE, DEFAULT_RH),
            ' Thickness (um)': DEFAULT_THICKNESS,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:  # Limit to 4 neat polymers
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        # Add row to DataFrame
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 2: Uneven blend mixture 1 with default conditions
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Uneven blend 1 (default conditions)")
        uneven1_blend_str = create_blend_string(blend['materials'], blend['uneven1_vols'])
        blend_predictions = run_prediction(uneven1_blend_str, DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        # Test neat polymers with same environmental conditions
        logger.info("     🔄 Testing neat polymers with default conditions...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['uneven1_vols'], DEFAULT_THICKNESS, DEFAULT_TEMPERATURE, DEFAULT_RH),
            ' Thickness (um)': DEFAULT_THICKNESS,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 3: Uneven blend mixture 2 with default conditions
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Uneven blend 2 (default conditions)")
        uneven2_blend_str = create_blend_string(blend['materials'], blend['uneven2_vols'])
        blend_predictions = run_prediction(uneven2_blend_str, DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        # Test neat polymers with same environmental conditions
        logger.info("     🔄 Testing neat polymers with default conditions...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, DEFAULT_RH, DEFAULT_THICKNESS)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['uneven2_vols'], DEFAULT_THICKNESS, DEFAULT_TEMPERATURE, DEFAULT_RH),
            ' Thickness (um)': DEFAULT_THICKNESS,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 4: Equal blend with varied thickness (20 μm)
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Equal blend (thickness 20μm)")
        blend_predictions = run_prediction(equal_blend_str, DEFAULT_TEMPERATURE, DEFAULT_RH, 20)
        
        # Test neat polymers with same environmental conditions (20μm thickness)
        logger.info("     🔄 Testing neat polymers with 20μm thickness...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, DEFAULT_RH, 20)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['equal_vols'], 20, DEFAULT_TEMPERATURE, DEFAULT_RH),
            ' Thickness (um)': 20,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 5: Equal blend with varied thickness (100 μm)
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Equal blend (thickness 100μm)")
        blend_predictions = run_prediction(equal_blend_str, DEFAULT_TEMPERATURE, DEFAULT_RH, 100)
        
        # Test neat polymers with same environmental conditions (100μm thickness)
        logger.info("     🔄 Testing neat polymers with 100μm thickness...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, DEFAULT_RH, 100)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['equal_vols'], 100, DEFAULT_TEMPERATURE, DEFAULT_RH),
            ' Thickness (um)': 100,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 6: Equal blend with varied temperature (37°C)
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Equal blend (temperature 37°C)")
        blend_predictions = run_prediction(equal_blend_str, 37, DEFAULT_RH, DEFAULT_THICKNESS)
        
        # Test neat polymers with same environmental conditions (37°C temperature)
        logger.info("     🔄 Testing neat polymers with 37°C temperature...")
        neat_results = test_neat_polymers(blend['materials'], 37, DEFAULT_RH, DEFAULT_THICKNESS)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['equal_vols'], DEFAULT_THICKNESS, 37, DEFAULT_RH),
            ' Thickness (um)': DEFAULT_THICKNESS,
            'Temperature': 37,
            'RH': DEFAULT_RH,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Test 7: Equal blend with varied RH (90%)
        current_test += 1
        logger.info(f"     Test {current_test}/{total_tests}: Equal blend (RH 90%)")
        blend_predictions = run_prediction(equal_blend_str, DEFAULT_TEMPERATURE, 90, DEFAULT_THICKNESS)
        
        # Test neat polymers with same environmental conditions (90% RH)
        logger.info("     🔄 Testing neat polymers with 90% RH...")
        neat_results = test_neat_polymers(blend['materials'], DEFAULT_TEMPERATURE, 90, DEFAULT_THICKNESS)
        
        row_data = {
            'Blend': create_blend_description(blend['materials'], blend['equal_vols'], DEFAULT_THICKNESS, DEFAULT_TEMPERATURE, 90),
            ' Thickness (um)': DEFAULT_THICKNESS,
            'Temperature': DEFAULT_TEMPERATURE,
            'RH': 90,
            'Blend WVTR': blend_predictions.get('wvtr', np.nan),
            'Blend Cobb': blend_predictions.get('cobb', np.nan),
            'Blend Tensile Strength': blend_predictions.get('ts', np.nan),
            'Blend Elongation at Break': blend_predictions.get('eab', np.nan),
            'Blend Disintegration': blend_predictions.get('disintegration', np.nan)
        }
        
        # Add neat polymer results
        for i, neat_result in enumerate(neat_results, 1):
            if i <= 4:
                predictions = neat_result['predictions']
                row_data.update({
                    f'Neat{i} WVTR': predictions.get('wvtr', np.nan),
                    f'Neat{i} Cobb': predictions.get('cobb', np.nan),
                    f'Neat{i} Tensile Strength': predictions.get('ts', np.nan),
                    f'Neat{i} Elongation at Break': predictions.get('eab', np.nan),
                    f'Neat{i} Disintegration': predictions.get('disintegration', np.nan)
                })
        
        qa_df = pd.concat([qa_df, pd.DataFrame([row_data])], ignore_index=True)
        
        # Small delay between blends
        time.sleep(1)
        
        # Small delay between blend mixtures
        time.sleep(2)
    
    # Save final results
    qa_df.to_csv(QA_CSV_PATH, index=False)
    logger.info(f"\n✅ Systematic QA testing completed! Results saved to {QA_CSV_PATH}")
    logger.info(f"📊 Total tests performed: {len(qa_df)}")
    
    # Print summary
    print("\n" + "="*80)
    print("SYSTEMATIC QA TESTING SUMMARY")
    print("="*80)
    print(f"Total tests performed: {len(qa_df)}")
    print(f"Blend mixtures tested: {len(blend_mixtures)}")
    print(f"  - Binary blends: {sum(1 for b in blend_mixtures if b['type'] == 'binary')}")
    print(f"  - Ternary blends: {sum(1 for b in blend_mixtures if b['type'] == 'ternary')}")
    print(f"  - Quaternary blends: {sum(1 for b in blend_mixtures if b['type'] == 'quaternary')}")
    print(f"Environmental variations (equal blends only):")
    print(f"  - Thickness: {THICKNESS_VALUES} μm")
    print(f"  - Temperature: {TEMPERATURE_VALUES} °C")
    print(f"  - RH: {RH_VALUES} %")
    print(f"Volume fraction patterns:")
    print(f"  - Equal: [0.5, 0.5], [0.33, 0.33, 0.34], [0.25, 0.25, 0.25, 0.25]")
    print(f"  - Uneven 1: [0.2, 0.8], [0.1, 0.3, 0.6], [0.4, 0.4, 0.1, 0.1]")
    print(f"  - Uneven 2: [0.8, 0.2], [0.6, 0.3, 0.1], [0.1, 0.1, 0.4, 0.4]")
    print("="*80)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Systematic QA testing for polymer blend models')
    parser.add_argument('--binary', type=int, default=1, help='Number of binary blends to test (default: 1)')
    parser.add_argument('--ternary', type=int, default=1, help='Number of ternary blends to test (default: 1)')
    parser.add_argument('--quaternary', type=int, default=0, help='Number of quaternary blends to test (default: 0)')
    parser.add_argument('--total', type=int, help='Total number of blends (will be distributed as binary/ternary/quaternary)')
    
    args = parser.parse_args()
    
    # If total is specified, distribute the blends
    if args.total:
        total = args.total
        if total <= 0:
            logger.error("❌ Total number of blends must be positive")
            return
        
        # Simple distribution: 70% binary, 25% ternary, 5% quaternary
        args.binary = max(1, int(total * 0.7))
        args.ternary = max(0, int(total * 0.25))
        args.quaternary = max(0, total - args.binary - args.ternary)
        
        logger.info(f"📊 Distributed {total} total blends: {args.binary} binary, {args.ternary} ternary, {args.quaternary} quaternary")
    
    print("🧪 Systematic Polymer Blend Model QA Testing")
    print("="*60)
    print(f"Output: {QA_CSV_PATH}")
    print(f"Materials: {MATERIAL_DICT_PATH}")
    print(f"Blends: {args.binary} binary, {args.ternary} ternary, {args.quaternary} quaternary")
    print("="*60)
    
    # Check if material dictionary exists
    if not os.path.exists(MATERIAL_DICT_PATH):
        logger.error(f"❌ Material dictionary not found: {MATERIAL_DICT_PATH}")
        return
    
    try:
        populate_qa_csv(args.binary, args.ternary, args.quaternary)
    except KeyboardInterrupt:
        logger.info("\n⚠️ Testing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()
