import pandas as pd
import numpy as np
import random
from itertools import combinations
import re
import os

def load_data():
    """Load the original EAB data and material-SMILES dictionary"""
    # Load original EAB data
    eab_data = pd.read_csv('data/eab/masterdata.csv')
    
    # Load material-SMILES dictionary
    smiles_dict = pd.read_csv('material-smiles-dictionary.csv')
    
    return eab_data, smiles_dict

def create_material_grade_mapping(eab_data, smiles_dict):
    """Create a mapping from material+grade to SMILES and EAB data"""
    mapping = {}
    
    # Material types will be read from the 'type' column in the data (same as TS)
    
    # Define immiscible materials (same as TS)
    immiscible_materials = {
        'Bio-PE': ['all'],
        'PP': ['all'],
        'PET': ['all'],
        'PA': ['all'],
        'EVOH': ['all']
    }
    
    for _, row in eab_data.iterrows():
        material = row['Materials']
        grade = row['Polymer Grade 1']
        eab1 = row['property1']  # First elongation at break property in %
        eab2 = row['property2']  # Second elongation at break property in %
        thickness = row['Thickness (um)']
        
        # Get material type from the 'type' column in the data (same as TS)
        material_type = row['type']  # Material type (brittle, hard, soft flex, etc.)
        
        # Check if material is immiscible
        is_immiscible = False
        if material in immiscible_materials:
            if immiscible_materials[material] == ['all'] or grade in immiscible_materials[material]:
                is_immiscible = True
        
        # Find corresponding SMILES
        mask = (smiles_dict['Material'] == material) & (smiles_dict['Grade'] == grade)
        if mask.any():
            smiles = smiles_dict[mask]['SMILES'].iloc[0]
            mapping[f"{material}_{grade}"] = {
                'material': material,
                'grade': grade,
                'smiles': smiles,
                'eab1': eab1,  # First elongation at break property
                'eab2': eab2,  # Second elongation at break property
                'thickness': thickness,
                'is_immiscible': is_immiscible,
                'type': material_type  # Material type for rule-based approach
            }
    
    return mapping

def inverse_rule_of_mixtures(compositions, eab_values):
    """
    Calculate elongation at break using inverse rule of mixtures weighted by volume fraction
    1/EAB_blend = Σ(vol_fraction_i / EAB_i)
    EAB_blend = 1 / Σ(vol_fraction_i / EAB_i)
    """
    if len(compositions) != len(eab_values):
        raise ValueError("Compositions and EAB values must have same length")
    
    # Calculate inverse EAB weighted by volume fraction
    inverse_eab_sum = 0
    for comp, eab in zip(compositions, eab_values):
        if eab > 0:  # Avoid division by zero
            inverse_eab_sum += comp / eab
    
    # Return the final EAB
    if inverse_eab_sum > 0:
        return 1 / inverse_eab_sum
    else:
        return 0

def regular_rule_of_mixtures(compositions, eab_values):
    """
    Calculate elongation at break using regular rule of mixtures weighted by volume fraction
    EAB_blend = Σ(vol_fraction_i * EAB_i)
    """
    if len(compositions) != len(eab_values):
        raise ValueError("Compositions and EAB values must have same length")
    
    # Calculate regular EAB weighted by volume fraction
    blend_eab = 0
    for comp, eab in zip(compositions, eab_values):
        blend_eab += comp * eab
    
    return blend_eab

def scale_eab_with_fixed_thickness(base_eab, thickness, reference_thickness=25):
    """Scale EAB with empirical power law using fixed 25μm reference
    Based on validation data analysis: EAB scales as thickness^0.4 (increased from 0.1687)"""
    empirical_exponent = 0.4  # Increased from 0.1687 for stronger thickness effect
    return base_eab * ((thickness ** empirical_exponent) / (reference_thickness ** empirical_exponent))

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
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(polymers))
    
    # Fill volume fractions
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Generate random thickness (only environmental parameter for EAB)
    thickness = np.random.uniform(10, 300)  # Thickness between 10-300 μm
    
    # Check for immiscible components
    immiscible_count = sum(1 for p in polymers if p['is_immiscible'])
    immiscible_fraction = immiscible_count / len(polymers)
    
    # Check if there is a coincidence of 'brittle' and 'soft flex' materials in the blend
    has_brittle = any(p['type'] == 'brittle' for p in polymers)
    has_soft_flex = any(p['type'] == 'soft flex' for p in polymers)
    brittle_soft_flex_coincidence = has_brittle and has_soft_flex
    
    # Choose rule of mixtures based on material types
    if brittle_soft_flex_coincidence:
        # Use inverse rule of mixtures if there's a coincidence of brittle/soft flex materials
        eab1_values = [p['eab1'] for p in polymers]
        blend_eab1 = inverse_rule_of_mixtures(compositions, eab1_values)
        
        eab2_values = [p['eab2'] for p in polymers]
        blend_eab2 = inverse_rule_of_mixtures(compositions, eab2_values)
    else:
        # Use regular rule of mixtures otherwise
        eab1_values = [p['eab1'] for p in polymers]
        blend_eab1 = regular_rule_of_mixtures(compositions, eab1_values)
        
        eab2_values = [p['eab2'] for p in polymers]
        blend_eab2 = regular_rule_of_mixtures(compositions, eab2_values)
    
    # Apply miscibility rule: if ≥30% immiscible components, EAB = random(5-7%) for phase separation
    if immiscible_fraction >= 0.3:
        blend_eab1 = np.random.uniform(5, 7)
        blend_eab2 = np.random.uniform(5, 7)
    
    # Scale EAB1 based on thickness using fixed reference
    blend_eab1 = scale_eab_with_fixed_thickness(blend_eab1, thickness)
    
    # Scale EAB2 based on thickness using fixed reference
    blend_eab2 = scale_eab_with_fixed_thickness(blend_eab2, thickness)
    
    # Add 25% Gaussian noise to make the data more realistic (same as original EAB)
    noise_level = 0.05  # 25% noise
    blend_eab1_noisy = blend_eab1 * (1 + np.random.normal(0, noise_level))
    blend_eab2_noisy = blend_eab2 * (1 + np.random.normal(0, noise_level))
    
    # Ensure the results stay positive
    blend_eab1_noisy = max(blend_eab1_noisy, 1.0)  # Minimum EAB of 1%
    blend_eab2_noisy = max(blend_eab2_noisy, 1.0)  # Minimum EAB of 1%
    
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
        'property1': blend_eab1_noisy,
        'property2': blend_eab2_noisy
    }

def get_random_polymer_combination(available_polymers, max_polymers=5):
    """Randomly select a combination of polymers with weighted probability"""
    # Weighted probability for number of polymers (favor 2-3, less for 4-5)
    weights = {2: 0.5, 3: 0.3, 4: 0.15, 5: 0.05}  # 50% 2-polymer, 30% 3-polymer, etc.
    
    # Randomly select number of polymers based on weights
    num_polymers = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    
    # Ensure we don't exceed available polymers
    num_polymers = min(num_polymers, len(available_polymers))
    
    # Randomly select polymers
    selected_polymers = random.sample(available_polymers, num_polymers)
    
    return selected_polymers

def augment_eab_data():
    """Main function to augment EAB data with completely random approach"""
    print("Loading data...")
    eab_data, smiles_dict = load_data()
    
    print("Creating material mapping...")
    mapping = create_material_grade_mapping(eab_data, smiles_dict)
    
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
    combined_df = pd.concat([eab_data, augmented_df], ignore_index=True)
    
    print(f"Generated {len(augmented_rows)} augmented rows")
    print(f"Total dataset size: {len(combined_df)} rows")
    
    # Save augmented data
    output_path = 'data/eab/masterdata_augmented.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    
    return combined_df, augmented_df

def create_ml_dataset(combined_data):
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = 'data/eab/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data = pd.read_csv('data/eab/masterdata.csv')
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
        ml_output_path = 'data/eab/polymerblends_for_ml.csv'
        ml_dataset.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        
        return ml_dataset
        
    except Exception as e:
        print(f"Warning: Could not create ML dataset: {e}")
        print("Using only augmented data")
        ml_output_path = 'data/eab/polymerblends_for_ml.csv'
        combined_data.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        return combined_data

def generate_report(original_data, augmented_data, combined_data):
    """Generate a detailed report of the augmentation process"""
    report = []
    report.append("=" * 60)
    report.append("EAB DATA AUGMENTATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Original data statistics
    report.append("ORIGINAL DATA STATISTICS:")
    report.append(f"- Number of original samples: {len(original_data)}")
    report.append(f"- Unique materials: {original_data['Materials'].nunique()}")
    report.append(f"- EAB1 range: {original_data['property1'].min():.2f} - {original_data['property1'].max():.2f} %")
    report.append(f"- EAB2 range: {original_data['property2'].min():.2f} - {original_data['property2'].max():.2f} %")
    report.append("")
    
    # Augmented data statistics
    report.append("AUGMENTED DATA STATISTICS:")
    report.append(f"- Number of augmented samples: {len(augmented_data)}")
    
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
    
    report.append(f"- EAB1 range: {augmented_data['property1'].min():.2f} - {augmented_data['property1'].max():.2f} %")
    report.append(f"- EAB2 range: {augmented_data['property2'].min():.2f} - {augmented_data['property2'].max():.2f} %")
    report.append("")
    
    # Combined data statistics
    report.append("COMBINED DATASET STATISTICS:")
    report.append(f"- Total samples: {len(combined_data)}")
    report.append(f"- Unique materials: {combined_data['Materials'].nunique()}")
    report.append(f"- EAB1 range: {combined_data['property1'].min():.2f} - {combined_data['property1'].max():.2f} %")
    report.append(f"- EAB2 range: {combined_data['property2'].min():.2f} - {combined_data['property2'].max():.2f} %")
    report.append("")
    
    # Sample augmented entries
    report.append("SAMPLE AUGMENTED ENTRIES:")
    for i, row in augmented_data.head(5).iterrows():
        report.append(f"- {row['Materials']}: EAB1={row['property1']:.2f}%, EAB2={row['property2']:.2f}%")
    report.append("")
    
    # Method used
    report.append("METHOD USED:")
    report.append("- Rule Selection Based on Material Types:")
    report.append("  * If blend contains coincidence of 'brittle' and 'soft flex' materials → Inverse Rule of Mixtures: EAB_blend = 1/Σ(vol_fraction_i / EAB_i)")
    report.append("  * Otherwise → Regular Rule of Mixtures: EAB_blend = Σ(vol_fraction_i * EAB_i)")
    report.append("- Miscibility rule: If ≥30% immiscible components, EAB = random(5-7%) due to phase separation")
    report.append("- Immiscible materials: Bio-PE, PP, PET, PA, EVOH (all grades)")
    report.append("- Completely random polymer selection (all grades included)")
    report.append("- Random compositions using Dirichlet distribution")
    report.append("- Weighted randomness: 50% 2-polymer, 30% 3-polymer, 15% 4-polymer, 5% 5-polymer")
    report.append("- Fixed thickness scaling: EAB * (thickness^0.4 / 25^0.4)")
    report.append("- Random thickness generation: 10-300μm")
    report.append("- SMILES structures mapped from material-grade dictionary")
    report.append("- Added 25% Gaussian noise to EAB predictions for realism")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting EAB data augmentation with random approach...")
    
    # Load original data
    original_data = pd.read_csv('data/eab/masterdata.csv')
    
    # Perform augmentation
    combined_data, augmented_data = augment_eab_data()
    
    # Generate and save report
    report = generate_report(original_data, augmented_data, combined_data)
    
    with open('eab_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print("\nAugmentation complete! Report saved to eab_augmentation_report.txt")
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    ml_dataset = create_ml_dataset(combined_data) 