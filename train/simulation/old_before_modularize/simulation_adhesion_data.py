import pandas as pd
import numpy as np
import random
from itertools import combinations
import re
import os

def load_data():
    """Load the original adhesion data and material-SMILES dictionary"""
    # Load original adhesion data
    adhesion_data = pd.read_csv('training/data/adhesion/masterdata.csv')
    
    # Load material-SMILES dictionary
    smiles_dict = pd.read_csv('material-smiles-dictionary.csv')
    
    return adhesion_data, smiles_dict

def create_material_grade_mapping(adhesion_data, smiles_dict):
    """Create a mapping from material+grade to SMILES and adhesion data"""
    mapping = {}
    
    for _, row in adhesion_data.iterrows():
        material = row['Materials']
        grade = row['Polymer Grade 1']
        adhesion = row['property']  # Single adhesion property
        thickness = row['Thickness (um)']
        sealing_temp = row['Sealing Temperature (C)']  # Sealing temperature
        
        # Find corresponding SMILES
        mask = (smiles_dict['Material'] == material) & (smiles_dict['Grade'] == grade)
        if mask.any():
            smiles = smiles_dict[mask]['SMILES'].iloc[0]
            mapping[f"{material}_{grade}"] = {
                'material': material,
                'grade': grade,
                'smiles': smiles,
                'adhesion': adhesion,  # Single adhesion property
                'thickness': thickness,
                'sealing_temp': sealing_temp  # Sealing temperature
            }
    
    return mapping

def rule_of_mixtures(compositions, adhesion_values):
    """
    Calculate adhesion using rule of mixtures weighted by volume fraction
    Adhesion_blend = Σ(vol_fraction_i * adhesion_i)
    """
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate adhesion weighted by volume fraction
    blend_adhesion = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        blend_adhesion += comp * adhesion
    
    return blend_adhesion

def inverse_rule_of_mixtures(compositions, adhesion_values):
    """
    Calculate adhesion using inverse rule of mixtures
    1/Adhesion_blend = Σ(vol_fraction_i / adhesion_i)
    """
    if len(compositions) != len(adhesion_values):
        raise ValueError("Compositions and adhesion values must have same length")
    
    # Calculate inverse adhesion
    inverse_adhesion_sum = 0
    for comp, adhesion in zip(compositions, adhesion_values):
        if adhesion > 0:  # Avoid division by zero
            inverse_adhesion_sum += comp / adhesion
    
    # Return the final adhesion
    if inverse_adhesion_sum > 0:
        return 1 / inverse_adhesion_sum
    else:
        return 0

def combined_rule_of_mixtures(compositions, adhesion_values, thickness):
    """
    Calculate adhesion using combined rule: 50% rule of mixtures + 50% inverse rule of mixtures
    Only applies when thickness < 30 micrometers
    """
    if thickness < 30:
        # Use 50/50 combination for thin films
        rom_adhesion = rule_of_mixtures(compositions, adhesion_values)
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, adhesion_values)
        combined_adhesion = 0.5 * rom_adhesion + 0.5 * inv_rom_adhesion
        return combined_adhesion
    else:
        # Use standard rule of mixtures for thicker films
        return rule_of_mixtures(compositions, adhesion_values)



def scale_adhesion_with_thickness(base_adhesion, thickness, reference_thickness=20):
    """Scale adhesion with thickness scaling using fixed 20 μm reference"""
    empirical_exponent = 0.5  # Moderate scaling for balanced thickness sensitivity
    return base_adhesion * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))

def calculate_dynamic_thickness_reference(polymers, compositions):
    """Calculate dynamic thickness reference based on weighted average of parent polymers"""
    weighted_thickness = 0
    for polymer, composition in zip(polymers, compositions):
        weighted_thickness += composition * polymer['thickness']
    return weighted_thickness

def scale_adhesion_with_dynamic_thickness(base_adhesion, blend_thickness, polymers, compositions):
    """Scale adhesion using dynamic thickness reference based on parent polymer blend"""
    # Calculate dynamic reference thickness from parent polymers
    dynamic_reference = calculate_dynamic_thickness_reference(polymers, compositions)
    
    # Use the same empirical exponent but with dynamic reference
    empirical_exponent = 0.5  # Moderate scaling for balanced thickness sensitivity
    return base_adhesion * ((blend_thickness ** empirical_exponent) / (dynamic_reference ** empirical_exponent))

def generate_random_composition(num_polymers):
    """Generate a completely random composition for n polymers"""
    # Use Dirichlet distribution to ensure compositions sum to 1
    composition = np.random.dirichlet(np.ones(num_polymers))
    return composition.tolist()

def create_blend_row(polymers, compositions, mapping, blend_number):
    """Create a single blend row with the given polymers and compositions"""
    # Use blend number for Materials column
    blend_name = str(blend_number)
    
    # Fill polymer grades
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(compositions))
    
    # Fill volume fractions
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Generate random thickness (same range as TS data)
    thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm
    
    # Determine max sealing temperature for the blend (lowest among polymers)
    sealing_temps = [p['sealing_temp'] for p in polymers]
    blend_max_sealing_temp = min(sealing_temps)  # Lowest sealing temperature determines blend capability
    
    # Use the blend's max sealing temperature directly (no random temperature)
    blend_temperature = blend_max_sealing_temp
    
    # Use combined rule of mixtures for thin films (< 30 μm), standard rule for thicker films
    adhesion_values = [p['adhesion'] for p in polymers]
    blend_adhesion = combined_rule_of_mixtures(compositions, adhesion_values, thickness)
    
    # Debug: Show which rule was used
    if thickness < 30:
        rom_adhesion = rule_of_mixtures(compositions, adhesion_values)
        inv_rom_adhesion = inverse_rule_of_mixtures(compositions, adhesion_values)
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm < 30 μm - Using 50/50 combined rule")
        print(f"  Rule of Mixtures: {rom_adhesion:.3f}, Inverse Rule: {inv_rom_adhesion:.3f}, Combined: {blend_adhesion:.3f}")
    else:
        print(f"Blend {blend_number}: Thickness {thickness:.1f} μm ≥ 30 μm - Using standard rule of mixtures: {blend_adhesion:.3f}")
    
    # Scale adhesion based on thickness using fixed 20 μm reference
    blend_adhesion = scale_adhesion_with_thickness(blend_adhesion, thickness, reference_thickness=20)
    
    # No temperature scaling needed - we're at the optimal sealing temperature
    # The blend adhesion is already calculated at the blend's max sealing temperature
    
    # Add 5% Gaussian noise to make the data more realistic
    noise_level = 0.05  # 5% noise
    blend_adhesion_noisy = blend_adhesion * (1 + np.random.normal(0, noise_level))
    
    # Ensure the results stay positive
    blend_adhesion_noisy = max(blend_adhesion_noisy, 0.1)  # Minimum adhesion of 0.1
    
    # DEBUG: Print the property value to ensure it's not NaN
    if pd.isna(blend_adhesion_noisy) or blend_adhesion_noisy <= 0:
        print(f"WARNING: Invalid property value for blend {blend_number}: {blend_adhesion_noisy}")
        blend_adhesion_noisy = 0.5  # Fallback value
    
    return {
        'Materials': blend_name,
        'Polymer Grade 1': grades[0],
        'Polymer Grade 2': grades[1],
        'Polymer Grade 3': grades[2],
        'Polymer Grade 4': grades[3],
        'Polymer Grade 5': grades[4],
        'SMILES1': smiles[0],
        'SMILES2': smiles[1],
        'SMILES3': smiles[2],
        'SMILES4': smiles[3],
        'SMILES5': smiles[4],
        'vol_fraction1': vol_fractions[0],
        'vol_fraction2': vol_fractions[1],
        'vol_fraction3': vol_fractions[2],
        'vol_fraction4': vol_fractions[3],
        'vol_fraction5': vol_fractions[4],
        'Thickness (um)': thickness,
        'Sealing Temperature (C)': blend_temperature,  # Blend's max sealing temperature
        'property': blend_adhesion_noisy,
        'unit': 'N/15mm'  # Default unit for adhesion
    }

def get_random_polymer_combination(available_polymers, max_polymers=5):
    """Randomly select a combination of polymers with weighted probability"""
    # Weighted probability for number of polymers (favor 2-3, less for 4-5)
    weights = {2: 0.7, 3: 0.2, 4: 0.05, 5: 0.05}  # 50% 2-polymer, 30% 3-polymer, etc.
    
    # Randomly select number of polymers based on weights
    num_polymers = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    
    # Ensure we don't exceed available polymers
    num_polymers = min(num_polymers, len(available_polymers))
    
    # Randomly select polymers
    selected_polymers = random.sample(available_polymers, num_polymers)
    
    return selected_polymers

def augment_adhesion_data():
    """Main function to augment adhesion data with simplified sealing temperature rules"""
    print("Loading adhesion data...")
    adhesion_data, smiles_dict = load_data()
    
    print("Creating material mapping...")
    mapping = create_material_grade_mapping(adhesion_data, smiles_dict)
    
    # Get list of ALL available polymers (all grades)
    available_polymers = list(mapping.values())
    
    print(f"Found {len(available_polymers)} unique polymer grades")
    
    # Generate augmented data
    augmented_rows = []
    target_total = 5000
    
    print(f"Generating {target_total} random blend combinations...")
    
    # Track used combinations to avoid exact duplicates
    used_combinations = set()
    
    attempts = 0
    max_attempts = target_total * 10  # Prevent infinite loop
    
    while len(augmented_rows) < target_total and attempts < max_attempts:
        attempts += 1
        
        # Randomly select polymer combination
        polymers = get_random_polymer_combination(available_polymers)
        
        # Create a unique key for this combination
        polymer_key = tuple(sorted([f"{p['material']}_{p['grade']}" for p in polymers]))
        
        # Skip if we've used this exact combination too many times
        if polymer_key in used_combinations:
            continue
        
        # Generate random composition
        composition = generate_random_composition(len(polymers))
        
        # Create blend row
        row = create_blend_row(polymers, composition, mapping, len(augmented_rows) + 1)
        augmented_rows.append(row)
        
        # Mark this combination as used
        used_combinations.add(polymer_key)
        
        # Progress update
        if len(augmented_rows) % 200 == 0:
            print(f"Generated {len(augmented_rows)} samples...")
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    
    # Combine with original data
    combined_df = pd.concat([adhesion_data, augmented_df], ignore_index=True)
    
    print(f"Generated {len(augmented_rows)} augmented rows")
    print(f"Total dataset size: {len(combined_df)} rows")
    
    # Save augmented data
    output_path = 'training/data/adhesion/masterdata_augmented.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    
    return combined_df, augmented_df

def create_ml_dataset(combined_data):
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = 'training/data/adhesion/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data = pd.read_csv('training/data/adhesion/masterdata.csv')
            original_count = len(original_data)
            
            # Extract only the augmented portion (skip the original data that's already in combined_data)
            augmented_only = combined_data.iloc[original_count:].copy()
            print(f"Augmented data only: {len(augmented_only)} samples")
            
            # Combine: original data + augmented data + validation blends
            ml_dataset = pd.concat([original_data, augmented_only, validation_data], ignore_index=True)
            print(f"Final ML dataset: {len(ml_dataset)} total samples")
            print(f"  - Original data: {len(original_data)} samples")
            print(f"  - Augmented data: {len(augmented_only)} samples") 
            print(f"  - Validation blends: {len(validation_data)} samples")
        else:
            print("Validation blends file not found, using only augmented data")
            ml_dataset = combined_data.copy()
        
        # Save the ML dataset
        ml_output_path = 'training/data/adhesion/polymerblends_for_ml.csv'
        ml_dataset.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        
        return ml_dataset
        
    except Exception as e:
        print(f"Warning: Could not create ML dataset: {e}")
        print("Using only augmented data")
        ml_output_path = 'data/adhesion/polymerblends_for_ml.csv'
        combined_data.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        return combined_data

def generate_report(original_data, augmented_data, combined_data):
    """Generate a detailed report of the augmentation process"""
    report = []
    report.append("=" * 60)
    report.append("ADHESION DATA AUGMENTATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Original data statistics
    report.append("ORIGINAL DATA STATISTICS:")
    report.append(f"- Number of original samples: {len(original_data)}")
    report.append(f"- Unique materials: {original_data['Materials'].nunique()}")
    report.append(f"- Adhesion range: {original_data['property'].min():.2f} - {original_data['property'].max():.2f}")
    if 'Sealing Temperature (C)' in original_data.columns:
        report.append(f"- Sealing temperature range: {original_data['Sealing Temperature (C)'].min():.0f} - {original_data['Sealing Temperature (C)'].max():.0f}°C")
    report.append("")
    
    # Augmented data statistics
    report.append("AUGMENTED DATA STATISTICS:")
    report.append(f"- Number of augmented samples: {len(augmented_data)}")
    report.append(f"- Thickness range: {augmented_data['Thickness (um)'].min():.1f} - {augmented_data['Thickness (um)'].max():.1f} μm")
    if 'Sealing Temperature (C)' in augmented_data.columns:
        report.append(f"- Sealing temperature range: {augmented_data['Sealing Temperature (C)'].min():.0f} - {augmented_data['Sealing Temperature (C)'].max():.0f}°C (blend-specific max sealing temperatures)")
    
    # Analyze blend types
    blend_types = []
    for material in augmented_data['Materials']:
        if '[' in material and ']' in material:
            polymers = material[1:-1].split('/')
            blend_types.append(len(polymers))
    
    blend_type_counts = pd.Series(blend_types).value_counts().sort_index()
    report.append("- Blend type distribution:")
    for blend_type, count in blend_type_counts.items():
        percentage = (count / len(augmented_data)) * 100
        report.append(f"  {blend_type}-polymer blends: {count} ({percentage:.1f}%)")
    
    report.append(f"- Adhesion range: {augmented_data['property'].min():.2f} - {augmented_data['property'].max():.2f}")
    report.append("")
    
    # Combined data statistics
    report.append("COMBINED DATASET STATISTICS:")
    report.append(f"- Total samples: {len(combined_data)}")
    report.append(f"- Unique materials: {combined_data['Materials'].nunique()}")
    report.append(f"- Adhesion range: {combined_data['property'].min():.2f} - {combined_data['property'].max():.2f}")
    report.append("")
    
    # Sample augmented entries
    report.append("SAMPLE AUGMENTED ENTRIES:")
    for i, row in augmented_data.head(5).iterrows():
        report.append(f"- {row['Materials']}: Adhesion={row['property']:.2f}, Temperature={row['Sealing Temperature (C)']:.0f}°C")
    report.append("")
    
    # Method used
    report.append("METHOD USED:")
    report.append("- Temperature Logic:")
    report.append("  * Blend max sealing temperature = minimum sealing temperature among polymers")
    report.append("  * Sealing temperature = blend's max sealing temperature (optimal condition)")
    report.append("- Property Calculation:")
    report.append("  * Always use Pure Rule of Mixtures: Adhesion_blend = Σ(vol_fraction_i * adhesion_i)")
    report.append("  * No complex temperature logic - properties calculated at optimal sealing temperature")
    report.append("- Temperature Scaling:")
    report.append("  * No temperature scaling needed - we're at the optimal sealing temperature")
    report.append("  * Properties are calculated at the blend's maximum capability")
    report.append("- Thickness Scaling: Dynamic reference based on weighted average of parent polymer thicknesses")
    report.append("  * Reference thickness = Σ(composition_i × thickness_i) for parent polymers")
    report.append("  * Scaling: Adhesion * (blend_thickness^0.125 / dynamic_reference^0.125)")
    report.append("- Random thickness generation: 10-600μm")
    report.append("- SMILES structures mapped from material-grade dictionary")
    report.append("- Added 5% Gaussian noise to adhesion predictions for realism")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting adhesion data augmentation with simplified sealing temperature rules...")
    
    # Load original data
    original_data = pd.read_csv('training/data/adhesion/masterdata.csv')
    
    # Perform augmentation
    combined_data, augmented_data = augment_adhesion_data()
    
    # Generate and save report
    report = generate_report(original_data, augmented_data, combined_data)
    
    with open('adhesion_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print("\nAugmentation complete! Report saved to adhesion_augmentation_report.txt")
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    ml_dataset = create_ml_dataset(combined_data) 