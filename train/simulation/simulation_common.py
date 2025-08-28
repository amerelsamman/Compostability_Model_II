#!/usr/bin/env python3
"""
Common simulation functions for polymer blend property augmentation.
This module contains ALL shared functionality used across all property types.
"""

import pandas as pd
import numpy as np
import random
import os
from typing import List, Dict, Tuple, Any, Callable


def load_material_smiles_dict():
    """Load the material-SMILES dictionary (common across all properties)"""
    return pd.read_csv('../material-smiles-dictionary.csv')


def generate_random_composition(num_polymers: int) -> List[float]:
    """Generate a completely random composition for n polymers using Dirichlet distribution"""
    # Use Dirichlet distribution to ensure compositions sum to 1
    composition = np.random.dirichlet(np.ones(num_polymers))
    return composition.tolist()


def get_random_polymer_combination(available_polymers: List[Dict], max_polymers: int = 5) -> List[Dict]:
    """Randomly select a combination of polymers with weighted probability - CONSISTENT ACROSS ALL PROPERTIES"""
    # Weighted probability for number of polymers (favor 2-3, less for 4-5) - EXACTLY AS ORIGINAL
    weights = {2: 0.5, 3: 0.3, 4: 0.15, 5: 0.05}  # 50% 2-polymer, 30% 3-polymer, 15% 4-polymer, 5% 5-polymer
    
    # Randomly select number of polymers based on weights
    num_polymers = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    
    # Ensure we don't exceed available polymers
    num_polymers = min(num_polymers, len(available_polymers))
    
    # Randomly select polymers
    selected_polymers = random.sample(available_polymers, num_polymers)
    
    return selected_polymers


def create_blend_row_base(polymers: List[Dict], compositions: List[float], 
                          blend_number: int, property_values: Dict[str, Any]) -> Dict[str, Any]:
    """Create a base blend row with common fields"""
    # Use blend number for Materials column
    blend_name = str(blend_number)
    
    # Fill polymer grades
    grades = [p['grade'] for p in polymers] + ['Unknown'] * (5 - len(polymers))
    
    # Fill SMILES
    smiles = [p['smiles'] for p in polymers] + [''] * (5 - len(polymers))
    
    # Fill volume fractions
    vol_fractions = compositions + [0] * (5 - len(compositions))
    
    # Base row structure
    row = {
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
    }
    
    # Add property-specific values
    row.update(property_values)
    
    return row


def run_augmentation_loop(property_name: str, available_polymers: List[Dict], 
                          target_total: int, create_blend_row_func: callable,
                          progress_interval: int = 1000) -> pd.DataFrame:
    """Main augmentation loop (common across all properties)"""
    print(f"Generating {target_total} random blend combinations...")
    
    # Generate augmented data
    augmented_rows = []
    
    # Track used combinations to avoid exact duplicates
    used_combinations = set()
    
    attempts = 0
    max_attempts = target_total * 10  # Prevent infinite loop
    
    while len(augmented_rows) < target_total and attempts < max_attempts:
        attempts += 1
        
        # Randomly select polymer combination - consistent weights across all properties
        polymers = get_random_polymer_combination(available_polymers)
        
        # Create a unique key for this combination
        polymer_key = tuple(sorted([f"{p['material']}_{p['grade']}" for p in polymers]))
        
        # Skip if we've used this exact combination too many times
        if polymer_key in used_combinations:
            continue
        
        # Generate random composition
        composition = generate_random_composition(len(polymers))
        
        # Create blend row using property-specific function
        row = create_blend_row_func(polymers, composition, len(augmented_rows) + 1)
        augmented_rows.append(row)
        
        # Mark this combination as used
        used_combinations.add(polymer_key)
        
        # Progress update
        if len(augmented_rows) % progress_interval == 0:
            print(f"Generated {len(augmented_rows)} samples...")
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    
    print(f"Generated {len(augmented_rows)} augmented rows")
    
    return augmented_df


def combine_with_original_data(original_data: pd.DataFrame, augmented_data: pd.DataFrame) -> pd.DataFrame:
    """Combine original data with augmented data"""
    combined_df = pd.concat([original_data, augmented_data], ignore_index=True)
    print(f"Total dataset size: {len(combined_df)} rows")
    return combined_df


def create_ml_dataset(combined_data: pd.DataFrame, property_name: str) -> pd.DataFrame:
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = f'data/{property_name}/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data_path = f'data/{property_name}/masterdata.csv'
            original_data = pd.read_csv(original_data_path)
            original_count = len(original_data)
            
            # Extract only the augmented portion (skip the original data that's already in combined_data)
            augmented_only = combined_data.iloc[original_count:].copy()
            print(f"Augmented data only: {len(augmented_only)} samples")
            
            # Combine: original + augmented + validation
            final_data = pd.concat([original_data, augmented_only, validation_data], ignore_index=True)
            print(f"Final ML dataset: {len(final_data)} total samples")
            print(f"  - Original data: {len(original_data)} samples")
            print(f"  - Augmented data: {len(augmented_only)} samples")
            print(f"  - Validation blends: {len(validation_data)} samples")
            
            # Save final ML dataset
            ml_output_path = f'data/{property_name}/polymerblends_for_ml.csv'
            final_data.to_csv(ml_output_path, index=False)
            print(f"✅ ML dataset saved to {ml_output_path}")
            
            return final_data
        else:
            print("No validation blends found, using combined data as ML dataset")
            ml_output_path = f'data/{property_name}/polymerblends_for_ml.csv'
            combined_data.to_csv(ml_output_path, index=False)
            print(f"✅ ML dataset saved to {ml_output_path}")
            return combined_data
            
    except Exception as e:
        print(f"Error creating ML dataset: {e}")
        return combined_data


def save_augmented_data(augmented_data: pd.DataFrame, property_name: str) -> str:
    """Save augmented data to property-specific directory"""
    output_path = f'data/{property_name}/masterdata_augmented.csv'
    augmented_data.to_csv(output_path, index=False)
    print(f"Saved augmented data to {output_path}")
    return output_path


def generate_simple_report(property_name: str, original_data: pd.DataFrame, augmented_data: pd.DataFrame) -> str:
    """Generate a simple report showing what rules were applied and when"""
    report = []
    report.append("=" * 60)
    report.append(f"{property_name.upper()} DATA AUGMENTATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Basic statistics
    report.append("BASIC STATISTICS:")
    report.append(f"- Number of original samples: {len(original_data)}")
    report.append(f"- Number of augmented samples: {len(augmented_data)}")
    report.append(f"- Total dataset size: {len(original_data) + len(augmented_data)} rows")
    report.append("")
    
    # Rules applied
    report.append("RULES APPLIED:")
    if property_name == 'ts':
        report.append("- Rule Selection Based on Material Types:")
        report.append("  * If blend contains coincidence of 'brittle' and 'soft flex' materials → Inverse Rule of Mixtures")
        report.append("  * If blend contains coincidence of 'hard' and 'soft flex' materials → Inverse Rule of Mixtures")
        report.append("  * Otherwise → Regular Rule of Mixtures")
        report.append("- Miscibility rule: If ≥30% immiscible components, both MD and TD = random(5-7 MPa) due to phase separation")
        report.append("- Immiscible materials: Bio-PE, PP, PET, PA, EVOH (all grades)")
        report.append("- Fixed thickness scaling: TS * (thickness^0.125 / 25^0.125) for both MD and TD")
        report.append("- Random thickness generation: 10-600μm")
        report.append("- Added 5% Gaussian noise to both MD and TD predictions")
    elif property_name == 'wvtr':
        report.append("- Inverse Rule of Mixtures for all blends")
        report.append("- Temperature scaling: logarithmic with upper bound of 5")
        report.append("- Humidity scaling: logarithmic with upper bound of 3")
        report.append("- Thickness scaling: power law of -0.8")
        report.append("- Random thickness generation: 10-300μm")
        report.append("- Added 20% Gaussian noise")
    elif property_name == 'cobb':
        report.append("- Inverse Rule of Mixtures for all blends")
        report.append("- Thickness scaling: power law of 0.15 (Cobb decreases with thickness)")
        report.append("- Random thickness generation: 10-300μm")
        report.append("- Added 25% Gaussian noise")
    elif property_name == 'eab':
        report.append("- Rule Selection Based on Material Types:")
        report.append("  * If blend contains coincidence of 'brittle' and 'soft flex' materials → Inverse Rule of Mixtures")
        report.append("  * Otherwise → Regular Rule of Mixtures")
        report.append("- Thickness scaling: power law of 0.1 (EAB increases with thickness)")
        report.append("- Random thickness generation: 10-300μm")
        report.append("- Added 25% Gaussian noise")
    elif property_name == 'eol':
        report.append("- Complex blending rules for max_L (disintegration) and t0 (time to 50% disintegration):")
        report.append("  * Rule 1: All home-compostable polymers (max_L > 90) → max_L = random(90-95), t0 = weighted average")
        report.append("  * Rule 2: PLA + compostable polymer (≥15%) → max_L excludes PLA, t0 includes PLA")
        report.append("  * Default: Rule of mixtures for both properties")
        report.append("- Noise: ±5% for max_L, ±10% for t0")
        report.append("- Two properties: property1 (max_L), property2 (t0)")
    elif property_name == 'adhesion':
        report.append("- Regular Rule of Mixtures for all blends")
        report.append("- Temperature scaling: logarithmic with upper bound of 5")
        report.append("- Random temperature generation: 15-50°C")
        report.append("- Added 20% Gaussian noise")
    elif property_name == 'otr':
        report.append("- Inverse Rule of Mixtures for all blends")
        report.append("- Thickness scaling: power law of -0.9 (OTR decreases with thickness)")
        report.append("- Random thickness generation: 10-300μm")
        report.append("- Added 20% Gaussian noise")
    
    report.append("")
    report.append("GENERAL FEATURES:")
    report.append("- Completely random polymer selection (all grades included)")
    report.append("- Random compositions using Dirichlet distribution")
    report.append("- Weighted randomness: 70% 2-polymer, 20% 3-polymer, 5% 4-polymer, 5% 5-polymer")
    report.append("- SMILES structures mapped from material-grade dictionary")
    report.append("- Immiscible materials: Bio-PE, PP, PET, PA, EVOH (all grades)")
    
    return "\n".join(report)


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seeds set to {seed} for reproducibility")


# Advanced scaling functions from original system
def scale_with_dynamic_thickness(base_value: float, thickness: float, polymers: List[Dict], 
                                compositions: List[float], power_law: float, reference_thickness: float = 25) -> float:
    """Scale property using dynamic reference thickness based on blend composition
    The reference thickness is calculated as the weighted average of individual polymer thicknesses"""
    
    # Calculate weighted average thickness of the blend
    weighted_thickness_sum = 0
    total_composition = 0
    
    for polymer, composition in zip(polymers, compositions):
        if 'thickness' in polymer:
            weighted_thickness_sum += polymer['thickness'] * composition
            total_composition += composition
    
    # Calculate the dynamic reference thickness
    dynamic_reference_thickness = weighted_thickness_sum / total_composition if total_composition > 0 else reference_thickness
    
    # Scale using the dynamic reference with specified power law
    return base_value * ((thickness ** power_law) / (dynamic_reference_thickness ** power_law))


def scale_with_fixed_thickness(base_value: float, thickness: float, power_law: float, reference_thickness: float = 25) -> float:
    """Scale property with fixed reference thickness using power law"""
    return base_value * ((thickness ** power_law) / (reference_thickness ** power_law))


def scale_with_temperature(value: float, temperature: float, reference_temp: float = 23, max_scale: float = 5) -> float:
    """Scale value logarithmically with temperature with upper bound"""
    return value * min(max_scale, 1 + np.log1p((temperature - reference_temp) / 10))


def scale_with_humidity(value: float, rh: float, reference_rh: float = 50, max_scale: float = 3) -> float:
    """Scale value logarithmically with humidity with upper bound"""
    return value * min(max_scale, 1 + np.log1p((rh - reference_rh) / 20))


def run_simulation_for_property(property_name: str, target_total: int, 
                               property_config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run simulation for a single property using the property configuration"""
    print(f"\n{'='*60}")
    print(f"Starting {property_config['name']} simulation...")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data...")
    original_data = property_config['load_data_func']()
    smiles_dict = load_material_smiles_dict()
    
    print("Creating polymer list from original data...")
    
    # Use the property-specific material mapping function (restored from original design)
    material_mapping = property_config['create_material_mapping']()
    available_polymers = list(material_mapping.values())
    
    print(f"Found {len(available_polymers)} unique polymer grades")
    
    # Run augmentation loop
    augmented_data = run_augmentation_loop(
        property_name=property_name,
        available_polymers=available_polymers,
        target_total=target_total,
        create_blend_row_func=property_config['create_blend_row_func'],
        progress_interval=1000
    )
    
    # Combine with original data
    combined_data = combine_with_original_data(original_data, augmented_data)
    
    # Save augmented data
    save_augmented_data(augmented_data, property_name)
    
    # Create ML dataset
    print("\nCreating ML dataset...")
    ml_dataset = create_ml_dataset(combined_data, property_name)
    
        # Generate and save simple report
    report = generate_simple_report(property_name, original_data, augmented_data)
    
    with open(f'simulation/reports/{property_name}_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print(f"\nAugmentation complete! Report saved to simulation/reports/{property_name}_augmentation_report.txt")
    
    return combined_data, augmented_data, ml_dataset


def run_all_simulations(target_total: int = 5000, seed: int = 42, 
                       properties_to_run: List[str] = None) -> Dict[str, Tuple]:
    """Run simulations for all properties or specified properties"""
    # PROPERTY_RULES is now defined in simulate.py
    from simulate import PROPERTY_RULES
    
    # Set random seeds
    set_random_seeds(seed)
    
    # Determine which properties to run
    if properties_to_run is None:
        properties_to_run = list(PROPERTY_RULES.keys())
    
    print(f"🚀 Running simulations for {len(properties_to_run)} properties...")
    print(f"📊 Target: {target_total:,} augmented samples per property")
    print(f"🎲 Random seed: {seed}")
    
    results = {}
    
    for property_name in properties_to_run:
        if property_name in PROPERTY_RULES:
            try:
                print(f"\n{'='*80}")
                print(f"PROPERTY {len(results)+1}/{len(properties_to_run)}: {property_name.upper()}")
                print(f"{'='*80}")
                
                result = run_simulation_for_property(
                    property_name=property_name,
                    target_total=target_total,
                    property_config=PROPERTY_RULES[property_name]
                )
                
                results[property_name] = result
                print(f"✅ {property_name.upper()} simulation completed successfully!")
                
            except Exception as e:
                print(f"❌ Error during {property_name.upper()} simulation: {e}")
                import traceback
                traceback.print_exc()
                results[property_name] = None
        else:
            print(f"⚠️  Property '{property_name}' not found in rules")
    
    # Summary
    successful = sum(1 for r in results.values() if r is not None)
    total = len(results)
    
    print(f"\n{'='*80}")
    print(f"SIMULATION SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total-successful}/{total}")
    
    if successful == total:
        print("\n🎉 All simulations completed successfully!")
    else:
        print(f"\n💥 {total-successful} simulation(s) failed. Check the logs above.")
    
    return results
