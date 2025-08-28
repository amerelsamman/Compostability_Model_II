import pandas as pd
import numpy as np
import random
from itertools import combinations
import re
import os

def load_data():
    """Load the original OTR data and material-SMILES dictionary"""
    # Load original OTR data
    otr_data = pd.read_csv('training/data/otr/masterdata.csv')
    
    # Load material-SMILES dictionary
    smiles_dict = pd.read_csv('material-smiles-dictionary.csv')
    
    return otr_data, smiles_dict

def create_material_grade_mapping(otr_data, smiles_dict):
    """Create a mapping from material+grade to SMILES and OTR data"""
    mapping = {}
    
    for _, row in otr_data.iterrows():
        material = row['Materials']
        grade = row['Polymer Grade 1']
        otr = row['property']
        temp = row['Temperature (C)']
        rh = row['RH (%)']
        thickness = row['Thickness (um)']
        
        # Find corresponding SMILES
        mask = (smiles_dict['Material'] == material) & (smiles_dict['Grade'] == grade)
        if mask.any():
            smiles = smiles_dict[mask]['SMILES'].iloc[0]
            mapping[f"{material}_{grade}"] = {
                'material': material,
                'grade': grade,
                'smiles': smiles,
                'otr': otr,
                'temp': temp,
                'rh': rh,
                'thickness': thickness
            }
    
    return mapping

def inverse_rule_of_mixtures(compositions, otr_values):
    """
    Calculate OTR using inverse rule of mixtures
    1/OTR_blend = Σ(vol_fraction_i / OTR_i)
    """
    if len(compositions) != len(otr_values):
        raise ValueError("Compositions and OTR values must have same length")
    
    # Calculate inverse OTR
    inverse_otr_sum = 0
    for comp, otr in zip(compositions, otr_values):
        if otr > 0:  # Avoid division by zero
            inverse_otr_sum += comp / otr
    
    # Return the final OTR
    if inverse_otr_sum > 0:
        return 1 / inverse_otr_sum
    else:
        return 0

def scale_otr_with_dynamic_thickness(base_otr, thickness, polymers, compositions):
    """Scale OTR using dynamic reference thickness based on blend composition
    The reference thickness is calculated as the weighted average of individual polymer thicknesses"""
    
    # Calculate weighted average thickness of the blend
    weighted_thickness_sum = 0
    total_composition = 0
    
    for polymer, composition in zip(polymers, compositions):
        weighted_thickness_sum += polymer['thickness'] * composition
        total_composition += composition
    
    # Calculate the dynamic reference thickness
    dynamic_reference_thickness = weighted_thickness_sum / total_composition if total_composition > 0 else 25
    
    # Scale OTR using the dynamic reference with power law of 0.5
    return base_otr * ((thickness ** 0.5) / (dynamic_reference_thickness ** 0.5))

def scale_otr_with_thickness(base_otr, thickness, reference_thickness=25):
    """Scale OTR with power law of 0.5 for thickness
    As thickness increases, OTR increases with power law of 0.5 (more gradual scaling)"""
    return base_otr * ((thickness ** 0.5) / (reference_thickness ** 0.5))

def scale_otr_with_temperature(otr, temperature, reference_temp=23):
    """Scale OTR logarithmically with temperature
    As temperature increases, OTR increases logarithmically with upper bound"""
    max_scale = 5  # maximum scaling factor
    return otr * min(max_scale, 1 + np.log1p((temperature - reference_temp) / 10))

def scale_otr_with_rh(otr, rh, reference_rh=50):
    """Scale OTR logarithmically with RH
    As RH increases, OTR increases logarithmically with upper bound"""
    max_scale = 3  # maximum scaling factor
    return otr * min(max_scale, 1 + np.log1p((rh - reference_rh) / 20))

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
    
    # Generate random environmental conditions
    temp = np.random.uniform(23, 50)  # Temperature between 23-50°C
    rh = np.random.uniform(50, 95)    # RH between 50-95%
    thickness = np.random.uniform(10, 600)  # Thickness between 10-600 μm
    
    # Calculate base OTR using inverse rule of mixtures
    otr_values = [p['otr'] for p in polymers]
    blend_otr = inverse_rule_of_mixtures(compositions, otr_values)
    
    # Scale OTR based on environmental conditions using dynamic thickness reference
    blend_otr = scale_otr_with_dynamic_thickness(blend_otr, thickness, polymers, compositions)
    blend_otr = scale_otr_with_temperature(blend_otr, temp)
    blend_otr = scale_otr_with_rh(blend_otr, rh)
    
    # Add 25% Gaussian noise to make the data more realistic
    noise_level = 0.1  # 25% noise
    blend_otr_noisy = blend_otr * (1 + np.random.normal(0, noise_level))
    
    # Ensure the result stays positive
    blend_otr_noisy = max(blend_otr_noisy, 0.01)  # Minimum OTR of 0.01
    
    # Calculate property_unnnormal (OTR without thickness normalization)
    property_unnnormal = blend_otr_noisy * thickness
    
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
        'Temperature (C)': temp,
        'RH (%)': rh,
        'Thickness (um)': thickness,
        'property': blend_otr_noisy,
        'property_unnnormal': property_unnnormal,
        'unit': 'cc*um/m2/day'
    }

def get_random_polymer_combination(available_polymers, max_polymers=5):
    """Randomly select a combination of polymers with weighted probability"""
    # Weighted probability for number of polymers (favor 2-3, less for 4-5)
    weights = {2: 0.4, 3: 0.3, 4: 0.05, 5: 0.05, 1:0.2}  # 50% 2-polymer, 30% 3-polymer, etc.
    
    # Randomly select number of polymers based on weights
    num_polymers = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    
    # Ensure we don't exceed available polymers
    num_polymers = min(num_polymers, len(available_polymers))
    
    # Randomly select polymers
    selected_polymers = random.sample(available_polymers, num_polymers)
    
    return selected_polymers

def augment_otr_data():
    """Main function to augment OTR data with completely random approach"""
    print("Loading data...")
    otr_data, smiles_dict = load_data()
    
    print("Creating material mapping...")
    mapping = create_material_grade_mapping(otr_data, smiles_dict)
    
    # Get list of ALL available polymers (all grades)
    available_polymers = list(mapping.values())
    
    print(f"Found {len(available_polymers)} unique polymer grades")
    
    # Generate augmented data
    augmented_rows = []
    target_total = 2000
    
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
    combined_df = pd.concat([otr_data, augmented_df], ignore_index=True)
    
    print(f"Generated {len(augmented_rows)} augmented rows")
    print(f"Total dataset size: {len(combined_df)} rows")
    
    # Save augmented data
    output_path = 'training/data/otr/masterdata_augmented.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    
    return combined_df, augmented_df

def generate_report(original_data, augmented_data, combined_data):
    """Generate a detailed report of the augmentation process"""
    report = []
    report.append("=" * 60)
    report.append("OTR DATA AUGMENTATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Original data statistics
    report.append("ORIGINAL DATA STATISTICS:")
    report.append(f"- Number of original samples: {len(original_data)}")
    report.append(f"- Unique materials: {original_data['Materials'].nunique()}")
    report.append(f"- OTR range: {original_data['property'].min():.2f} - {original_data['property'].max():.2f} cc·μm/m²·day")
    report.append("")
    
    # Augmented data statistics
    report.append("AUGMENTED DATA STATISTICS:")
    report.append(f"- Number of augmented samples: {len(augmented_data)}")
    report.append(f"- Temperature range: {augmented_data['Temperature (C)'].min():.1f} - {augmented_data['Temperature (C)'].max():.1f}°C")
    report.append(f"- RH range: {augmented_data['RH (%)'].min():.1f} - {augmented_data['RH (%)'].max():.1f}%")
    report.append(f"- Thickness range: {augmented_data['Thickness (um)'].min():.1f} - {augmented_data['Thickness (um)'].max():.1f} μm")
    report.append(f"- OTR range: {augmented_data['property'].min():.2f} - {augmented_data['property'].max():.2f} cc·μm/m²·day")
    report.append("")
    
    # Combined data statistics
    report.append("COMBINED DATASET STATISTICS:")
    report.append(f"- Total samples: {len(combined_data)}")
    report.append(f"- Unique materials: {combined_data['Materials'].nunique()}")
    report.append(f"- OTR range: {combined_data['property'].min():.2f} - {combined_data['property'].max():.2f} cc·μm/m²·day")
    report.append("")
    
    return "\n".join(report)

def create_ml_dataset(combined_data):
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = 'training/data/otr/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data = pd.read_csv('training/data/otr/masterdata.csv')
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
        ml_output_path = 'training/data/otr/polymerblends_for_ml.csv'
        ml_dataset.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        
        return ml_dataset
        
    except Exception as e:
        print(f"Warning: Could not create ML dataset: {e}")
        print("Using only augmented data")
        ml_output_path = 'training/data/otr/polymerblends_for_ml.csv'
        combined_data.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        return combined_data

def visualize_scaling_effects(augmented_data):
    """Create visualizations to verify scaling behavior"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: OTR vs Thickness
    sns.scatterplot(data=augmented_data, x='Thickness (um)', y='property', alpha=0.5, ax=ax1)
    ax1.set_title('OTR vs Thickness')
    ax1.set_xlabel('Thickness (μm)')
    ax1.set_ylabel('OTR (cc·μm/m²·day)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: OTR vs Temperature
    sns.scatterplot(data=augmented_data, x='Temperature (C)', y='property', alpha=0.5, ax=ax2)
    ax2.set_title('OTR vs Temperature')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('OTR (cc·μm/m²·day)')
    ax2.set_yscale('log')
    
    # Plot 3: OTR vs RH
    sns.scatterplot(data=augmented_data, x='RH (%)', y='property', alpha=0.5, ax=ax3)
    ax3.set_title('OTR vs Relative Humidity')
    ax3.set_xlabel('RH (%)')
    ax3.set_ylabel('OTR (cc·μm/m²·day)')
    ax3.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('training/data/otr/augmented_data_scaling_effects.png')
    plt.close()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting OTR data augmentation with random approach...")
    
    # Load original data
    original_data = pd.read_csv('training/data/otr/masterdata.csv')
    
    # Perform augmentation
    combined_data, augmented_data = augment_otr_data()
    
    # Generate and save report
    report = generate_report(original_data, augmented_data, combined_data)
    
    with open('otr_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print("\nAugmentation complete! Report saved to otr_augmentation_report.txt")
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    ml_dataset = create_ml_dataset(combined_data)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualize_scaling_effects(augmented_data)
    print("Visualizations saved to training/data/otr/augmented_data_scaling_effects.png") 