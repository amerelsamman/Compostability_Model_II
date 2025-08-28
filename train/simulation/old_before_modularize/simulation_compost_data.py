import pandas as pd
import numpy as np
import random
from itertools import combinations
import re
import os

def load_data():
    """Load the original compostability data and material-SMILES dictionary"""
    # Load original compostability data
    compost_data = pd.read_csv('data/eol/masterdata.csv')
    
    # Load material-SMILES dictionary
    smiles_dict = pd.read_csv('../material-smiles-dictionary.csv')
    
    return compost_data, smiles_dict

def create_material_grade_mapping(compost_data, smiles_dict):
    """Create a mapping from material+grade to SMILES and compostability data"""
    mapping = {}
    
    for _, row in compost_data.iterrows():
        material = row['Materials']
        grade = row['Polymer Grade 1']
        max_L = row['property1']  # Disintegration level (0-100, >90 = home-compostable)
        t0 = row['property2']     # Time to 50% disintegration (days)
        
        # Find corresponding SMILES
        mask = (smiles_dict['Material'] == material) & (smiles_dict['Grade'] == grade)
        if mask.any():
            smiles = smiles_dict[mask]['SMILES'].iloc[0]
            mapping[f"{material}_{grade}"] = {
                'material': material,
                'grade': grade,
                'smiles': smiles,
                'max_L': max_L,
                't0': t0
            }
    
    return mapping

def rule_of_mixtures(compositions, values):
    """
    Calculate property using rule of mixtures weighted by volume fraction
    Property_blend = Σ(vol_fraction_i * Property_i)
    """
    if len(compositions) != len(values):
        raise ValueError("Compositions and values must have same length")
    
    # Calculate weighted average
    blend_property = 0
    for comp, prop in zip(compositions, values):
        if pd.notna(prop):  # Handle NaN values
            blend_property += comp * prop
    
    return blend_property

def apply_compostability_rules(polymers, compositions, mapping):
    """
    Apply compostability-specific rules to determine max_L and t0
    """
    max_L_values = []
    t0_values = []
    pla_fraction = 0.0
    compostable_fraction = 0.0
    non_compostable_fraction = 0.0
    
    # Collect polymer properties and identify PLA
    for polymer, fraction in zip(polymers, compositions):
        polymer_key = f"{polymer['material']}_{polymer['grade']}"
        if polymer_key in mapping:
            polymer_data = mapping[polymer_key]
            max_L = polymer_data['max_L']
            t0 = polymer_data['t0']
            
            max_L_values.append(max_L)
            t0_values.append(t0)
            
            # Check if it's PLA
            if 'PLA' in polymer['material'].upper():
                pla_fraction += fraction
            
            # Check if it's compostable (max_L > 90)
            if max_L > 90:
                compostable_fraction += fraction
            else:
                non_compostable_fraction += fraction
    
    # Apply Rule 1: If all polymers have known values and total fraction = 1.0 (but NOT PLA blends)
    if (len(max_L_values) == len(polymers) and 
        all(pd.notna(max_L) for max_L in max_L_values) and
        all(pd.notna(t0) for t0 in t0_values) and
        abs(sum(compositions) - 1.0) < 0.01 and
        pla_fraction == 0):  # Exclude PLA blends from Rule 1
        
        # If all polymers are home-compostable (max_L > 90)
        if all(max_L > 90 for max_L in max_L_values):
            # Use random max_L between 90-95 for purely home-compostable blends
            max_L_pred = random.uniform(90.0, 95.0)
            # Calculate weighted average t0
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  All polymers home-compostable: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
        else:
            # Use weighted averages for mixed blends
            max_L_pred = rule_of_mixtures(compositions, max_L_values)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  Mixed blend: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
    
    # Apply Rule 2: PLA + Compostable polymer rule
    if (pla_fraction > 0 and 
        compostable_fraction >= 0.15):
        
        # For max_L: Exclude PLA from calculation, use only non-PLA polymers
        non_pla_max_L_values = []
        non_pla_compositions = []
        non_pla_t0_values = []
        
        for i, (polymer, fraction) in enumerate(zip(polymers, compositions)):
            if 'PLA' not in polymer['material'].upper():
                non_pla_max_L_values.append(max_L_values[i])
                non_pla_compositions.append(fraction)
                non_pla_t0_values.append(t0_values[i])
        
        # Normalize non-PLA compositions to sum to 1
        if sum(non_pla_compositions) > 0:
            normalized_compositions = [f / sum(non_pla_compositions) for f in non_pla_compositions]
            
            # Calculate max_L excluding PLA (rule of mixtures on non-PLA polymers only)
            max_L_pred = rule_of_mixtures(normalized_compositions, non_pla_max_L_values)
            
            # For t0: PLA still contributes normally (rule of mixtures on all polymers)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            
            print(f"  PLA rule applies: max_L = {max_L_pred:.2f} (PLA excluded), t0 = {t0_pred:.2f} (PLA included)")
            return max_L_pred, t0_pred
        else:
            # Fallback if somehow no non-PLA polymers
            max_L_pred = rule_of_mixtures(compositions, max_L_values)
            t0_pred = rule_of_mixtures(compositions, t0_values)
            print(f"  PLA rule fallback: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
            return max_L_pred, t0_pred
    
    # Default: Use rule of mixtures for both properties
    max_L_pred = rule_of_mixtures(compositions, max_L_values)
    t0_pred = rule_of_mixtures(compositions, t0_values)
    print(f"  Default rule of mixtures: max_L = {max_L_pred:.2f}, t0 = {t0_pred:.2f}")
    return max_L_pred, t0_pred





def generate_random_composition(num_polymers):
    """Generate a completely random composition for n polymers"""
    # Use Dirichlet distribution to ensure compositions sum to 1
    composition = np.random.dirichlet(np.ones(num_polymers))
    return composition.tolist()

def create_blend_row(polymers, compositions, mapping, blend_number):
    """Create a single blend row with compostability properties"""
    # Use blend number for Materials column (consistent with other simulation files)
    blend_name = str(blend_number)
    
    # Fill polymer grades
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(polymers))
    
    # Fill volume fractions
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Apply compostability rules to determine property1 (max_L) and property2 (t0)
    max_L_pred, t0_pred = apply_compostability_rules(polymers, compositions, mapping)
    
    # Add some noise for realism (±5% for max_L, ±10% for t0)
    max_L_noise = random.uniform(0.95, 1.05)
    t0_noise = random.uniform(0.90, 1.10)
    
    max_L_final = max_L_pred * max_L_noise
    t0_final = t0_pred * t0_noise
    
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
        'property1': max_L_final,
        'property2': t0_final
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

def augment_compostability_data():
    """Main function to augment compostability data with rule-based approach"""
    print("Loading data...")
    compost_data, smiles_dict = load_data()
    
    print("Creating material mapping...")
    mapping = create_material_grade_mapping(compost_data, smiles_dict)
    
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
    combined_df = pd.concat([compost_data, augmented_df], ignore_index=True)
    
    print(f"Generated {len(augmented_rows)} augmented rows")
    print(f"Total dataset size: {len(combined_df)} rows")
    
    # Save augmented data
    output_path = 'data/eol/masterdata_augmented.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    
    return combined_df, augmented_df

def create_ml_dataset(combined_data):
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = 'data/eol/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data = pd.read_csv('data/eol/masterdata.csv')
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
        ml_output_path = 'data/eol/polymerblends_for_ml.csv'
        ml_dataset.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        
        return ml_dataset
        
    except Exception as e:
        print(f"Warning: Could not create ML dataset: {e}")
        print("Using only augmented data")
        ml_output_path = 'data/eol/polymerblends_for_ml.csv'
        combined_data.to_csv(ml_output_path, index=False)
        print(f"✅ ML dataset saved to {ml_output_path}")
        return combined_data

def generate_report(original_data, augmented_data, combined_data):
    """Generate a detailed report of the augmentation process"""
    report = []
    report.append("=" * 60)
    report.append("COMPOSTABILITY DATA AUGMENTATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Original data statistics
    report.append("ORIGINAL DATA STATISTICS:")
    report.append(f"- Number of original samples: {len(original_data)}")
    report.append(f"- Unique materials: {original_data['Materials'].nunique()}")
    report.append(f"- Property1 (max_L) range: {original_data['property1'].min():.2f} - {original_data['property1'].max():.2f}")
    report.append(f"- Property2 (t0) range: {original_data['property2'].min():.2f} - {original_data['property2'].max():.2f} days")
    report.append("")
    
    # Augmented data statistics
    report.append("AUGMENTED DATA STATISTICS:")
    report.append(f"- Number of augmented samples: {len(augmented_data)}")
    
    # Analyze blend types (Materials column now contains index numbers, not blend descriptions)
    # Blend type information is available in the polymer grade columns
    blend_types = []
    for _, row in augmented_data.iterrows():
        # Count non-empty polymer grades
        num_polymers = sum(1 for i in range(1, 6) if pd.notna(row[f'Polymer Grade {i}']) and row[f'Polymer Grade {i}'] != 'Unknown')
        blend_types.append(num_polymers)
    
    blend_type_counts = pd.Series(blend_types).value_counts().sort_index()
    report.append("- Blend type distribution:")
    for blend_type, count in blend_type_counts.items():
        percentage = (count / len(augmented_data)) * 100
        report.append(f"  {blend_type}-polymer blends: {count} ({percentage:.1f}%)")
    
    report.append(f"- Property1 (max_L) range: {augmented_data['property1'].min():.2f} - {augmented_data['property1'].max():.2f}")
    report.append(f"- Property2 (t0) range: {augmented_data['property2'].min():.2f} - {augmented_data['property2'].max():.2f} days")
    report.append("")
    
    # Combined data statistics
    report.append("COMBINED DATASET STATISTICS:")
    report.append(f"- Total samples: {len(combined_data)}")
    report.append(f"- Unique materials: {combined_data['Materials'].nunique()}")
    report.append(f"- Property1 (max_L) range: {combined_data['property1'].min():.2f} - {combined_data['property1'].max():.2f}")
    report.append(f"- Property2 (t0) range: {combined_data['property2'].min():.2f} - {combined_data['property2'].max():.2f} days")
    report.append("")
    
    # Sample augmented entries
    report.append("SAMPLE AUGMENTED ENTRIES:")
    for i, row in augmented_data.head(5).iterrows():
        # Build blend description from polymer grades and fractions
        blend_parts = []
        for j in range(1, 6):
            grade = row[f'Polymer Grade {j}']
            fraction = row[f'vol_fraction{j}']
            if pd.notna(grade) and grade != 'Unknown' and fraction > 0:
                blend_parts.append(f"{grade}({fraction:.2f})")
        blend_desc = "/".join(blend_parts) if blend_parts else "Unknown"
        report.append(f"- Blend {row['Materials']} ({blend_desc}): max_L={row['property1']:.2f}, t0={row['property2']:.2f} days")
    report.append("")
    
    # Method used
    report.append("METHOD USED:")
    report.append("- Compostability Rules Applied:")
    report.append("  * Rule 1: If all polymers have known max_L/t0 values → Use weighted averages or random(90-95) for home-compostable")
    report.append("  * Rule 2: PLA + compostable polymer (≥15% compostable) → max_L excludes PLA (rule of mixtures on non-PLA only), t0 includes PLA (rule of mixtures on all)")
    report.append("  * Default: Rule of mixtures for both max_L and t0")
    report.append("- Property1 (max_L): Disintegration level (0-100, >90 = home-compostable)")
    report.append("- Property2 (t0): Time to 50% disintegration (days)")
    report.append("- Completely random polymer selection (all grades included)")
    report.append("- Random compositions using Dirichlet distribution")
    report.append("- Weighted randomness: 50% 2-polymer, 30% 3-polymer, 15% 4-polymer, 5% 5-polymer")
    report.append("- Added 5% Gaussian noise to max_L and 10% to t0 for realism")
    report.append("- SMILES structures mapped from material-grade dictionary")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    print("Starting compostability data augmentation with rule-based approach...")
    
    # Load original data
    original_data = pd.read_csv('data/eol/masterdata.csv')
    
    # Perform augmentation
    combined_data, augmented_data = augment_compostability_data()
    
    # Generate and save report
    report = generate_report(original_data, augmented_data, combined_data)
    
    with open('compostability_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print("\nAugmentation complete! Report saved to compostability_augmentation_report.txt")
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    ml_dataset = create_ml_dataset(combined_data) 