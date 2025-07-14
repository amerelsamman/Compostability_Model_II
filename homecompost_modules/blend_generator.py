import pandas as pd
import numpy as np
from scipy.stats import dirichlet
import hashlib
from .core_model import (
    sus, DAYS, SAMPLE_AREA, THICKNESS_DEFAULT,
    generate_material_curve, generate_material_curve_with_synergistic_boost, parse_thickness, find_material_by_grade,
    parse_blend_input, is_home_compostable_certified, is_medium_disintegration_polymer, GLOBAL_SEED
)

def deterministic_hash(s):
    """Deterministically hash a string to a 32-bit integer seed."""
    return int(hashlib.md5(s.encode('utf-8')).hexdigest(), 16) % 2**32

def generate_blend(blend_str, actual_thickness=None):
    """Generate a blend - core function used by both CSV and plotting"""
    materials = parse_blend_input(blend_str)
    
    # Find materials and generate curves
    material_info = []
    for mat in materials:
        material_row = find_material_by_grade(sus, mat['material'], mat['grade'])
        if material_row is not None:
            thickness_val = parse_thickness(material_row.get('Thickness 1', None))
            material_info.append({
                'polymer': material_row['Polymer Category'],
                'grade': material_row['Grade'],
                'vol_frac': mat['vol_frac'],
                'tuv_home': str(material_row.get('TUV Home', '')),
                'thickness': thickness_val
            })
    if not material_info:
        raise ValueError("No valid materials found for blend")
    # Calculate home-compostable fraction in the blend
    home_fraction = sum(mat['vol_frac'] for mat in material_info if is_home_compostable_certified(mat['tuv_home']))
    # Generate curves with synergistic effects using material-specific seeds
    for i, material in enumerate(material_info):
        # Create a deterministic seed for each material based on its grade
        material_seed = deterministic_hash(material['grade'])
        material['curve'] = generate_material_curve_with_synergistic_boost(
            material['polymer'],
            material['grade'],
            material['tuv_home'],
            material['thickness'],
            home_fraction,
            material_seed=material_seed,
            actual_thickness=actual_thickness
        )
    # Calculate blend curve
    blend_curve = np.zeros(DAYS)
    for material in material_info:
        blend_curve += material['curve'] * material['vol_frac']
    # Set seed for blend curve monotonicity
    np.random.seed(GLOBAL_SEED + 3000)  # Offset for blend curve adjustments
    # Ensure blend curve is monotonically increasing
    for i in range(1, len(blend_curve)):
        if blend_curve[i] < blend_curve[i-1]:
            blend_curve[i] = blend_curve[i-1] + np.random.uniform(0, 0.01)  # Very small positive increment
    # No clipping - let sigmoid handle everything naturally
    blend_curve = np.clip(blend_curve, 0, None)
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return material_info, blend_curve

def generate_csv_for_single_blend(blend_str, output_path=None, actual_thickness=None):
    """Generate disintegration profile for a single blend and optionally save to CSV"""
    material_info, blend_curve = generate_blend(blend_str, actual_thickness=actual_thickness)
    
    # Determine blend properties
    is_home_blend = any(is_home_compostable_certified(mat['tuv_home']) for mat in material_info)
    home_status = "home" if is_home_blend else "non-home"
    
    # Create blend label
    blend_label = " + ".join([f"{mat['grade']}({mat['vol_frac']:.2f})" for mat in material_info])
    
    # Create dictionary of {time: prediction}
    time_prediction_dict = {}
    for day in range(1, DAYS + 1):
        time_prediction_dict[day] = blend_curve[day-1]
    
    # Optionally save CSV if output_path is provided
    if output_path is not None:
        # Generate CSV data
        all_data = []
        for day in range(1, DAYS + 1):
            row_data = {
                'Type': 'Disintegration',
                'Polymer Grade 1': material_info[0]['grade'] if len(material_info) > 0 else '',
                'Polymer Grade 2': material_info[1]['grade'] if len(material_info) > 1 else '',
                'Polymer Grade 3': material_info[2]['grade'] if len(material_info) > 2 else '',
                'Polymer Grade 4': material_info[3]['grade'] if len(material_info) > 3 else '',
                'Polymer Grade 5': material_info[4]['grade'] if len(material_info) > 4 else '',
                'SMILES1': '', 'SMILES2': '', 'SMILES3': '', 'SMILES4': '', 'SMILES5': '',
                'vol_fraction1': material_info[0]['vol_frac'] if len(material_info) > 0 else 0,
                'vol_fraction2': material_info[1]['vol_frac'] if len(material_info) > 1 else 0,
                'vol_fraction3': material_info[2]['vol_frac'] if len(material_info) > 2 else 0,
                'vol_fraction4': material_info[3]['vol_frac'] if len(material_info) > 3 else 0,
                'vol_fraction5': material_info[4]['vol_frac'] if len(material_info) > 4 else 0,
                'Temperature (C)': 28.0,
                'Sample area (mm^2)': SAMPLE_AREA,
                'Thickness (mm)': actual_thickness if actual_thickness is not None else '',
                'Time(day)': day,
                'property': blend_curve[day-1],
                'home_status': home_status,
                'blend_label': blend_label
            }
            all_data.append(row_data)
        
        # Save CSV
        df = pd.DataFrame(all_data)
        df.to_csv(output_path, index=False)
    
    return time_prediction_dict

def generate_random_blends(num_blends, max_materials):
    """Generate random blends using Dirichlet distribution for volume fractions"""
    print(f"Generating homopolymers first, then {num_blends} random blends with up to {max_materials} materials each...")
    
    # Get all available materials from sustainability.csv
    available_materials = []
    for _, row in sus.iterrows():
        if pd.notna(row['Grade']) and str(row['Grade']).strip() != '':
            available_materials.append({
                'polymer': row['Polymer Category'],
                'grade': row['Grade'],
                'tuv_home': str(row.get('TUV Home', '')),
                'thickness': parse_thickness(row.get('Thickness 1', None))
            })
    
    print(f"Found {len(available_materials)} available materials")
    
    all_blend_data = []
    
    # STEP 1: Generate all homopolymers (single materials) first
    print(f"\n=== STEP 1: Generating {len(available_materials)} homopolymers ===")
    for material_idx, material in enumerate(available_materials):
        # Create deterministic seed for homopolymer
        material_seed = deterministic_hash(material['grade'])
        
        # Single material with 100% volume fraction
        material_info = [{
            'polymer': material['polymer'],
            'grade': material['grade'],
            'vol_frac': 1.0,
            'curve': generate_material_curve(
                material['polymer'],
                material['grade'],
                material['tuv_home'],
                material['thickness'],
                material_seed=material_seed
            ),
            'tuv_home': material['tuv_home'],
            'thickness': material['thickness']
        }]
        
        # Determine blend properties
        is_home_blend = is_home_compostable_certified(material['tuv_home'])
        home_status = "home" if is_home_blend else "non-home"
        
        # Calculate thickness
        thickness_val = material['thickness'] if material['thickness'] != THICKNESS_DEFAULT else None
        
        # Create blend label
        blend_label = f"{material['grade']}(1.00)"
        
        # Add data for each day
        for day in range(1, DAYS + 1):
            row_data = {
                'Type': 'Disintegration',
                'Polymer Grade 1': material['grade'],
                'Polymer Grade 2': '',
                'Polymer Grade 3': '',
                'Polymer Grade 4': '',
                'Polymer Grade 5': '',
                'SMILES1': '',
                'SMILES2': '',
                'SMILES3': '',
                'SMILES4': '',
                'SMILES5': '',
                'vol_fraction1': 1.0,
                'vol_fraction2': 0,
                'vol_fraction3': 0,
                'vol_fraction4': 0,
                'vol_fraction5': 0,
                'Temperature (C)': 28.0,
                'Sample area (mm^2)': SAMPLE_AREA,
                'Thickness (mm)': thickness_val if thickness_val is not None else '',
                'Time(day)': day,
                'property': material_info[0]['curve'][day-1],
                'home_status': home_status,
                'blend_label': blend_label
            }
            all_blend_data.append(row_data)
        
        if (material_idx + 1) % 10 == 0:
            print(f"Generated {material_idx + 1}/{len(available_materials)} homopolymers")
    
    print(f"✅ Generated all {len(available_materials)} homopolymers")
    
    # STEP 2: Generate random blends
    print(f"\n=== STEP 2: Generating {num_blends} random blends ===")
    for blend_idx in range(num_blends):
        # Set seed for this specific blend
        blend_seed = GLOBAL_SEED + 4000 + blend_idx
        np.random.seed(blend_seed)
        
        # Randomly choose number of materials (2 to max_materials for blends)
        num_materials = np.random.randint(2, max_materials + 1)
        
        # Randomly select materials
        selected_materials = np.random.choice(available_materials, num_materials, replace=False)
        
        # Generate volume fractions using Dirichlet distribution
        alpha = np.ones(num_materials)  # Uniform Dirichlet distribution
        volume_fractions = dirichlet.rvs(alpha, size=1)[0]
        
        # Calculate home-compostable fraction for synergistic effects
        home_fraction = sum(vol_frac for i, vol_frac in enumerate(volume_fractions) 
                           if is_home_compostable_certified(selected_materials[i]['tuv_home']))
        
        # Generate blend curve
        blend_curve = np.zeros(DAYS)
        material_info = []
        
        for i, material in enumerate(selected_materials):
            # Create deterministic seed for each material in this blend
            material_seed = deterministic_hash(f"{material['grade']}_{blend_idx}")
            
            # Generate individual material curve with synergistic effects
            curve = generate_material_curve_with_synergistic_boost(
                material['polymer'],
                material['grade'],
                material['tuv_home'],
                material['thickness'],
                home_fraction,
                material_seed=material_seed
            )
            
            material_info.append({
                'polymer': material['polymer'],
                'grade': material['grade'],
                'vol_frac': volume_fractions[i],
                'curve': curve,
                'tuv_home': material['tuv_home'],
                'thickness': material['thickness']
            })
            
            blend_curve += curve * volume_fractions[i]
        
        # Set seed for blend curve monotonicity
        np.random.seed(blend_seed + 1000)
        
        # Ensure blend curve is monotonically increasing
        for i in range(1, len(blend_curve)):
            if blend_curve[i] < blend_curve[i-1]:
                blend_curve[i] = blend_curve[i-1] + np.random.uniform(0, 0.01)
        
        blend_curve = np.clip(blend_curve, 0, None)
        
        # Determine blend properties
        is_home_blend = any(is_home_compostable_certified(mat['tuv_home']) for mat in material_info)
        home_status = "home" if is_home_blend else "non-home"
        
        # Calculate average thickness
        thicknesses = [mat['thickness'] for mat in material_info if mat['thickness'] != THICKNESS_DEFAULT]
        avg_thickness = np.mean(thicknesses) if thicknesses else None
        
        # Create blend label
        blend_label = " + ".join([f"{mat['grade']}({mat['vol_frac']:.2f})" for mat in material_info])
        
        # Add data for each day
        for day in range(1, DAYS + 1):
            row_data = {
                'Type': 'Disintegration',
                'Polymer Grade 1': material_info[0]['grade'] if len(material_info) > 0 else '',
                'Polymer Grade 2': material_info[1]['grade'] if len(material_info) > 1 else '',
                'Polymer Grade 3': material_info[2]['grade'] if len(material_info) > 2 else '',
                'Polymer Grade 4': material_info[3]['grade'] if len(material_info) > 3 else '',
                'Polymer Grade 5': material_info[4]['grade'] if len(material_info) > 4 else '',
                'SMILES1': '',
                'SMILES2': '',
                'SMILES3': '',
                'SMILES4': '',
                'SMILES5': '',
                'vol_fraction1': material_info[0]['vol_frac'] if len(material_info) > 0 else 0,
                'vol_fraction2': material_info[1]['vol_frac'] if len(material_info) > 1 else 0,
                'vol_fraction3': material_info[2]['vol_frac'] if len(material_info) > 2 else 0,
                'vol_fraction4': material_info[3]['vol_frac'] if len(material_info) > 3 else 0,
                'vol_fraction5': material_info[4]['vol_frac'] if len(material_info) > 4 else 0,
                'Temperature (C)': 28.0,
                'Sample area (mm^2)': SAMPLE_AREA,
                'Thickness (mm)': avg_thickness if avg_thickness is not None else '',
                'Time(day)': day,
                'property': blend_curve[day-1],
                'home_status': home_status,
                'blend_label': blend_label
            }
            all_blend_data.append(row_data)
        
        if (blend_idx + 1) % 10 == 0:
            print(f"Generated {blend_idx + 1}/{num_blends} blends")
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return all_blend_data 