#!/usr/bin/env python3
"""
Common simulation functions for polymer blend property augmentation.
This module contains ALL shared functionality used across all property types.
"""

import pandas as pd
import numpy as np
import random
import os
import sys
from typing import List, Dict, Tuple, Any, Callable, Optional
from collections import defaultdict

# Import UMM3 correction module
from umm3_correction import UMM3Correction, load_ingredients_config, load_polymer_corrections_config, load_family_compatibility_config


def get_terminal_colors():
    """Get appropriate color codes based on terminal capabilities"""
    # Check if we're in a terminal that supports colors
    if not sys.stdout.isatty():
        # Not a terminal, use no colors
        return {
            'GREEN': '', 'BLUE': '', 'YELLOW': '', 'CYAN': '', 'WHITE': '', 'MAGENTA': '',
            'BOLD': '', 'RESET': ''
        }
    
    # Check if terminal supports colors (basic check)
    if os.environ.get('TERM', '').lower() in ['xterm', 'xterm-256color', 'screen', 'tmux']:
        # Full color support
        return {
            'GREEN': '\033[92m',
            'BLUE': '\033[94m', 
            'YELLOW': '\033[93m',
            'CYAN': '\033[96m',
            'WHITE': '\033[97m',
            'MAGENTA': '\033[95m',
            'BOLD': '\033[1m',
            'RESET': '\033[0m'
        }
    else:
        # Limited or no color support, use basic formatting
        return {
            'GREEN': '\033[32m',
            'BLUE': '\033[34m',
            'YELLOW': '\033[33m', 
            'CYAN': '\033[36m',
            'WHITE': '\033[37m',
            'MAGENTA': '\033[35m',
            'BOLD': '\033[1m',
            'RESET': '\033[0m'
        }


class RuleUsageTracker:
    """Track usage of different blending rules during simulation"""
    
    def __init__(self):
        self.rule_counts = defaultdict(int)
        self.total_blends = 0
    
    def record_rule_usage(self, rule_name: str):
        """Record that a specific rule was used"""
        self.rule_counts[rule_name] += 1
        self.total_blends += 1
    
    def get_summary(self) -> str:
        """Get a formatted summary of rule usage with colors"""
        if not self.rule_counts:
            return "No rules recorded"
        
        # Get terminal-appropriate colors
        colors = get_terminal_colors()
        
        summary = f"\n{colors['GREEN']}📊 Rule Usage Summary ({self.total_blends} total blends):{colors['RESET']}\n"
        summary += f"{colors['BLUE']}{'=' * 60}{colors['RESET']}\n"
        
        # Sort by usage count (descending)
        sorted_rules = sorted(self.rule_counts.items(), key=lambda x: x[1], reverse=True)
        
        for i, (rule_name, count) in enumerate(sorted_rules):
            percentage = (count / self.total_blends) * 100
            # Use different colors for different rules
            color = [colors['CYAN'], colors['YELLOW'], colors['GREEN'], colors['WHITE']][i % 4]
            summary += f"  {color}{colors['BOLD']}{rule_name:<35}{colors['RESET']} {colors['YELLOW']}{count:>6} times{colors['RESET']} {colors['GREEN']}({percentage:>5.1f}%){colors['RESET']}\n"
        
        return summary
    
    def get_rule_data(self) -> Dict[str, Any]:
        """Get rule usage data for Streamlit display"""
        if not self.rule_counts:
            return {"rules": [], "total_blends": 0}
        
        # Sort by usage count (descending)
        sorted_rules = sorted(self.rule_counts.items(), key=lambda x: x[1], reverse=True)
        
        rules_data = []
        for rule_name, count in sorted_rules:
            percentage = (count / self.total_blends) * 100
            rules_data.append({
                "rule_name": rule_name,
                "count": count,
                "percentage": percentage
            })
        
        return {
            "rules": rules_data,
            "total_blends": self.total_blends
        }


def load_material_smiles_dict():
    """Load the material-SMILES dictionary (common across all properties)"""
    import os
    
    # Load main polymer dictionary
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "material-smiles-dictionary.csv")
    polymer_df = pd.read_csv(csv_path)
    
    # Load additives/fillers dictionary
    additives_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "additives-fillers-dictionary.csv")
    additives_df = pd.read_csv(additives_path)
    
    # Combine both dataframes and sort for deterministic ordering
    combined_df = pd.concat([polymer_df, additives_df], ignore_index=True)
    combined_df = combined_df.sort_values(['Material', 'Grade']).reset_index(drop=True)
    
    return combined_df


def load_additives_fillers_config():
    """Load additives and fillers configuration for UMM3 corrections"""
    return load_ingredients_config()


def load_polymer_corrections_config():
    """Load polymer corrections configuration for UMM3 corrections"""
    try:
        from umm3_correction import load_polymer_corrections_config as _load_polymer_corrections_config
        import os
        # Try both possible config paths
        if os.path.exists("config"):
            return _load_polymer_corrections_config("config")
        elif os.path.exists("train/simulation/config"):
            return _load_polymer_corrections_config("train/simulation/config")
        else:
            raise FileNotFoundError("Could not find config directory")
    except Exception as e:
        print(f"Error: Could not load polymer corrections config: {e}")
        raise e

def load_family_compatibility_config(property_name: str = None):
    """Load material family compatibility configuration for UMM3 corrections"""
    try:
        from umm3_correction import load_family_compatibility_config as _load_family_compatibility_config
        import os
        
        # If property_name is specified, try to load property-specific compatibility file
        if property_name:
            compatibility_path = f"train/simulation/config/compatibility/{property_name}_compatibility.yaml"
            if os.path.exists(compatibility_path):
                return _load_family_compatibility_config("train/simulation/config/compatibility", property_name)
        
        # Fallback to general compatibility file
        if os.path.exists("config"):
            return _load_family_compatibility_config("config")
        elif os.path.exists("train/simulation/config"):
            return _load_family_compatibility_config("train/simulation/config")
        else:
            raise FileNotFoundError("Could not find config directory")
    except Exception as e:
        print(f"Error: Could not load family compatibility config: {e}")
        raise e


def load_environmental_controls_config():
    """Load environmental and thickness controls configuration"""
    try:
        import yaml
        import os
        # Try both possible config paths
        if os.path.exists("config/environmental_controls.yaml"):
            config_path = "config/environmental_controls.yaml"
        elif os.path.exists("train/simulation/config/environmental_controls.yaml"):
            config_path = "train/simulation/config/environmental_controls.yaml"
        else:
            raise FileNotFoundError("Could not find environmental_controls.yaml")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config["environmental_controls"]
    except Exception as e:
        print(f"Error: Could not load environmental controls config: {e}")
        raise e


def get_environmental_parameters(property_name: str, env_config: Dict[str, Any]) -> Dict[str, Any]:
    """Get environmental parameters for a specific property from config"""
    if property_name not in env_config:
        raise ValueError(f"No environmental controls found for property: {property_name}")
    
    prop_config = env_config[property_name]
    params = {}
    
    # Thickness parameters
    if 'thickness' in prop_config:
        thickness_config = prop_config['thickness']
        params['thickness'] = {
            'min': thickness_config['min'],
            'max': thickness_config['max'],
            'power_law': thickness_config['power_law'],
            'reference': thickness_config['reference'],
            'scaling_type': thickness_config['scaling_type']
        }
        
        # Add dynamic scaling parameters if available
        if 'dynamic_scaling' in thickness_config:
            params['thickness']['dynamic_scaling'] = thickness_config['dynamic_scaling']
    
    # Temperature parameters
    if 'temperature' in prop_config:
        temp_config = prop_config['temperature']
        params['temperature'] = {
            'min': temp_config['min'],
            'max': temp_config['max'],
            'reference': temp_config['reference'],
            'max_scale': temp_config['max_scale'],
            'scaling_type': temp_config['scaling_type']
        }
    
    # Humidity parameters
    if 'humidity' in prop_config:
        humidity_config = prop_config['humidity']
        params['humidity'] = {
            'min': humidity_config['min'],
            'max': humidity_config['max'],
            'reference': humidity_config['reference'],
            'max_scale': humidity_config['max_scale'],
            'scaling_type': humidity_config['scaling_type']
        }
    
    # Noise parameters
    if 'noise' in prop_config:
        noise_config = prop_config['noise']
        params['noise'] = {
            'enabled': noise_config['enabled'],
            'type': noise_config['type'],
            'level': noise_config['level']
        }
    
    # Special parameters (e.g., TS cap for adhesion)
    for key in ['ts_cap', 'miscibility_rule']:
        if key in prop_config:
            params[key] = prop_config[key]
    
    return params


def apply_umm3_corrections(property_values: Dict[str, Any], property_name: str, 
                          polymers: List[Dict], compositions: List[float], 
                          umm3_correction: UMM3Correction, ingredients_config: Dict[str, Any],
                          polymer_corrections_config: Dict[str, Any], 
                          family_compatibility_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Apply UMM3 corrections to property values for ALL polymers and additives/fillers in the blend.
    
    Args:
        property_values: Dictionary of property values to correct
        property_name: Name of the property (e.g., 'tensile', 'elongation')
        polymers: List of polymer dictionaries (includes all materials)
        compositions: List of volume fractions
        umm3_correction: UMM3 correction instance
        ingredients_config: Configuration of additives/fillers
        polymer_corrections_config: Configuration of polymer corrections
    
    Returns:
        Dictionary with corrected property values and corrections_applied metadata
    """
    corrected_values = property_values.copy()
    all_corrections_applied = {}
    
    # Map property names to UMM3 property names
    property_mapping = {
        'ts': 'tensile',
        'eab': 'elongation', 
        'otr': 'otr',
        'wvtr': 'wvtr',
        'seal': 'seal',
        'cobb': 'cobb'
    }
    
    umm3_property = property_mapping.get(property_name)
    if not umm3_property:
        return corrected_values
    
    # Apply corrections to ALL polymers and additives/fillers
    for polymer, composition in zip(polymers, compositions):
        material = polymer.get('material', '')
        grade = polymer.get('grade', '')
        
        # Create a unique key for this material+grade combination
        material_key = f"{material}_{grade}"
        
        # Get correction config (either from polymer corrections or ingredients)
        correction_config = None
        material_type = polymer.get('type', 'polymer')
        
        if material_type in ['additive', 'filler'] and ingredients_config and grade in ingredients_config:
            # This is an additive or filler
            correction_config = ingredients_config[grade]
        elif material_key in polymer_corrections_config:
            # This is a regular polymer with corrections
            correction_config = polymer_corrections_config[material_key]
        
        if correction_config:
            # Apply corrections to each property value
            corrections_applied = {}
            for prop_key, prop_value in property_values.items():
                if isinstance(prop_value, (int, float)) and prop_value > 0:
                    try:
                        # Create blend info for tracking
                        blend_info = {
                            'material': material,
                            'grade': grade,
                            'loading': composition,
                            'property': umm3_property,
                            'original_value': prop_value
                        }
                        
                        corrected_value, log_factor, was_clipped = umm3_correction.adjust_property(
                            prop_value, composition, correction_config, umm3_property,
                            material_name=material_key, blend_info=blend_info
                        )
                        corrected_values[prop_key] = corrected_value
                        corrections_applied[prop_key] = {
                            'original': prop_value,
                            'corrected': corrected_value,
                            'log_factor': log_factor,
                            'was_clipped': was_clipped,
                            'loading': composition,
                            'material': material,
                            'grade': grade
                        }
                    except Exception as e:
                        print(f"Warning: Could not apply correction to {prop_key} for {material_key}: {e}")
                        # Keep original value if correction fails
            
            if corrections_applied:
                all_corrections_applied[material_key] = corrections_applied
    
    # Apply pairwise interfacial compatibility corrections if config is available
    if family_compatibility_config:
        corrected_values = umm3_correction.apply_pairwise_compatibility_corrections(
            corrected_values, polymers, compositions, family_compatibility_config, property_name
        )
    
    # Add all corrections applied
    if all_corrections_applied:
        corrected_values['corrections_applied'] = all_corrections_applied
    
    return corrected_values


def generate_random_composition(num_polymers: int) -> List[float]:
    """Generate a completely random composition for n polymers using Dirichlet distribution"""
    # Use Dirichlet distribution to ensure compositions sum to 1
    composition = np.random.dirichlet(np.ones(num_polymers))
    return composition.tolist()


def get_random_polymer_combination(available_polymers: List[Dict], max_polymers: int = 5) -> List[Dict]:
    """Randomly select a combination of polymers with weighted probability - CONSISTENT ACROSS ALL PROPERTIES"""
    # Weighted probability for number of polymers (favor 2-3, less for 4-5) - EXACTLY AS ORIGINAL
    weights = {2: 0.8, 3: 0.1, 4: 0.05, 5: 0.05}  # 50% 2-polymer, 30% 3-polymer, 15% 4-polymer, 5% 5-polymer
    
    # Randomly select number of polymers based on weights
    num_polymers = random.choices(list(weights.keys()), weights=list(weights.values()))[0]
    
    # Ensure we don't exceed available polymers
    num_polymers = min(num_polymers, len(available_polymers))
    
    # Randomly select polymers
    if isinstance(available_polymers, dict):
        selected_polymers = random.sample(list(available_polymers.values()), num_polymers)
    else:
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
                          progress_interval: int = 1000, selected_rules: Dict[str, bool] = None,
                          additive_probability: float = 0.3, enable_additives: bool = True,
                          disable_ts_model: bool = False) -> Tuple[pd.DataFrame, RuleUsageTracker]:
    """Main augmentation loop (common across all properties) with UMM3 corrections for additives/fillers"""
    # Get terminal-appropriate colors
    colors = get_terminal_colors()
    
    print(f"{colors['YELLOW']}Generating {target_total} random blend combinations...{colors['RESET']}")
    
    # Load UMM3 correction system (always load polymer corrections, conditionally load additives)
    umm3_correction = None
    ingredients_config = None
    polymer_corrections_config = None
    family_compatibility_config = None
    
    # Always try to load polymer corrections
    try:
        polymer_corrections_config = load_polymer_corrections_config()
        import os
        if os.path.exists("config"):
            umm3_correction = UMM3Correction.from_config_files("config")
        elif os.path.exists("train/simulation/config"):
            umm3_correction = UMM3Correction.from_config_files("train/simulation/config")
        else:
            raise FileNotFoundError("Could not find config directory")
        print(f"{colors['CYAN']}Loaded UMM3 correction system with {len(polymer_corrections_config)} polymer corrections{colors['RESET']}")
    except Exception as e:
        print(f"{colors['YELLOW']}Warning: Could not load polymer corrections: {e}{colors['RESET']}")
        print(f"{colors['YELLOW']}Continuing without polymer corrections...{colors['RESET']}")
        polymer_corrections_config = None
        umm3_correction = None
    
    # Try to load family compatibility config
    try:
        family_compatibility_config = load_family_compatibility_config(property_name)
        print(f"{colors['CYAN']}Loaded family compatibility config with {len(family_compatibility_config)} material pairs{colors['RESET']}")
    except Exception as e:
        print(f"{colors['YELLOW']}Warning: Could not load family compatibility config: {e}{colors['RESET']}")
        print(f"{colors['YELLOW']}Continuing without pairwise compatibility corrections...{colors['RESET']}")
        family_compatibility_config = None
    
    
    # Try to load environmental controls config
    try:
        environmental_controls_config = load_environmental_controls_config()
        print(f"{colors['CYAN']}Loaded environmental controls config{colors['RESET']}")
    except Exception as e:
        print(f"{colors['YELLOW']}Warning: Could not load environmental controls config: {e}{colors['RESET']}")
        print(f"{colors['YELLOW']}Continuing with default environmental parameters...{colors['RESET']}")
        environmental_controls_config = None
    
    # Load additives/fillers only if enabled
    if enable_additives:
        try:
            ingredients_config = load_additives_fillers_config()
            print(f"{colors['CYAN']}Loaded {len(ingredients_config)} additives/fillers{colors['RESET']}")
        except Exception as e:
            print(f"{colors['YELLOW']}Warning: Could not load additives/fillers: {e}{colors['RESET']}")
            print(f"{colors['YELLOW']}Continuing without additives...{colors['RESET']}")
            enable_additives = False
    
    # Initialize rule usage tracker
    rule_tracker = RuleUsageTracker()
    
    # Generate augmented data
    augmented_rows = []
    
    # Track used combinations to avoid exact duplicates
    used_combinations = set()
    
    attempts = 0
    max_attempts = target_total * 10  # Prevent infinite loop
    
    while len(augmented_rows) < target_total and attempts < max_attempts:
        attempts += 1
        
        # Randomly select polymer combination - this now includes additives/fillers as regular materials
        polymers = get_random_polymer_combination(available_polymers)
        
        # Create a unique key for this combination
        polymer_key = tuple(sorted([f"{p['material']}_{p['grade']}" for p in polymers]))
        
        # Skip if we've used this exact combination too many times
        if polymer_key in used_combinations:
            continue
        
        # Generate random composition for all components (polymers + additives/fillers)
        polymer_composition = generate_random_composition(len(polymers))
        
        # Create blend row using property-specific function (with rule tracking and selected rules)
        if property_name == 'seal':
            row = create_blend_row_func(polymers, polymer_composition, len(augmented_rows) + 1, rule_tracker, selected_rules, environmental_controls_config, disable_ts_model)
        else:
            row = create_blend_row_func(polymers, polymer_composition, len(augmented_rows) + 1, rule_tracker, selected_rules, environmental_controls_config)
        
        # Apply UMM3 corrections if enabled (to ALL polymers and optionally additives/fillers)
        if umm3_correction and polymer_corrections_config:
            # Extract property values for correction
            property_values = {}
            for key, value in row.items():
                if key.startswith('property') and isinstance(value, (int, float)):
                    property_values[key] = value
            
            # Apply UMM3 corrections to all materials
            if property_values:
                corrected_property_values = apply_umm3_corrections(
                    property_values, property_name, polymers, polymer_composition, 
                    umm3_correction, ingredients_config, polymer_corrections_config,
                    family_compatibility_config
                )
                
                # Update the row with corrected values
                for key, value in corrected_property_values.items():
                    if key != 'corrections_applied':  # Handle corrections_applied separately
                        row[key] = value
                
                # Add corrections_applied if any were applied
                if 'corrections_applied' in corrected_property_values:
                    row['corrections_applied'] = corrected_property_values['corrections_applied']
        
        augmented_rows.append(row)
        
        # Mark this combination as used
        used_combinations.add(polymer_key)
        
        # Progress update
        if len(augmented_rows) % progress_interval == 0:
            print(f"{colors['CYAN']}Generated {len(augmented_rows)} samples...{colors['RESET']}")
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_rows)
    
    print(f"{colors['GREEN']}Generated {len(augmented_rows)} augmented rows{colors['RESET']}")
    
    # Report clipping statistics if UMM3 corrections were applied
    if umm3_correction:
        clipping_summary = umm3_correction.get_clipping_summary()
        if clipping_summary:
            print(f"\n{colors['MAGENTA']}📊 UMM3 Clipping Statistics:{colors['RESET']}")
            print("=" * 60)
            
            total_corrections = 0
            total_clipped = 0
            
            for key, stats in clipping_summary.items():
                material, prop = key.split('_', 1)
                total_corrections += stats['total_corrections']
                total_clipped += stats['clipped_corrections']
                
                clip_rate_pct = stats['clip_rate'] * 100
                print(f"{colors['CYAN']}{material} ({prop}):{colors['RESET']}")
                print(f"  Corrections: {stats['total_corrections']} total, {stats['clipped_corrections']} clipped ({clip_rate_pct:.1f}%)")
                
                # Show blend examples for clipped corrections
                if stats['blend_examples']:
                    print(f"  Example blends with clipping:")
                    for i, example in enumerate(stats['blend_examples'][:3]):  # Show first 3 examples
                        print(f"    {i+1}. {example['material']} {example['grade']} "
                              f"(loading: {example['loading']:.3f}, "
                              f"original: {example['original_value']:.2f})")
                print()
            
            overall_clip_rate = (total_clipped / total_corrections * 100) if total_corrections > 0 else 0
            print(f"{colors['YELLOW']}Overall: {total_corrections} corrections, {total_clipped} clipped ({overall_clip_rate:.1f}%){colors['RESET']}")
            print("=" * 60)
    
    return augmented_df, rule_tracker


def combine_with_original_data(original_data: pd.DataFrame, augmented_data: pd.DataFrame) -> pd.DataFrame:
    """Combine original data with augmented data"""
    combined_df = pd.concat([original_data, augmented_data], ignore_index=True)
    print(f"Total dataset size: {len(combined_df)} rows")
    return combined_df


def create_ml_dataset(combined_data: pd.DataFrame, property_name: str) -> pd.DataFrame:
    """Create polymerblends_for_ml.csv by combining augmented data with validation blends"""
    try:
        # Try to load validation blends
        validation_path = f'train/data/{property_name}/validationblends.csv'
        if os.path.exists(validation_path):
            validation_data = pd.read_csv(validation_path)
            print(f"Found validation blends: {len(validation_data)} samples")
            
            # Load original data to identify what's original vs validation
            original_data_path = f'train/data/{property_name}/masterdata.csv'
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
            ml_output_path = f'train/data/{property_name}/polymerblends_for_ml.csv'
            final_data.to_csv(ml_output_path, index=False)
            print(f"✅ ML dataset saved to {ml_output_path}")
            
            return final_data
        else:
            print("No validation blends found, using combined data as ML dataset")
            ml_output_path = f'train/data/{property_name}/polymerblends_for_ml.csv'
            combined_data.to_csv(ml_output_path, index=False)
            print(f"✅ ML dataset saved to {ml_output_path}")
            return combined_data
            
    except Exception as e:
        print(f"Error creating ML dataset: {e}")
        return combined_data


def save_augmented_data(augmented_data: pd.DataFrame, property_name: str) -> str:
    """Save augmented data to property-specific directory"""
    output_path = f'train/data/{property_name}/masterdata_augmented.csv'
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
        report.append("  * If >50% brittle content → HYBRID Rule: (1/brittle)^0.1 + (1/flexible)")
        report.append("  * If blend contains coincidence of 'brittle' and 'soft flex' materials → Inverse Rule of Mixtures")
        report.append("  * If blend contains coincidence of 'hard' and 'soft flex' materials → Inverse Rule of Mixtures")
        report.append("  * Otherwise → Regular Rule of Mixtures")
        report.append("- Miscibility rule: DISABLED (was: If ≥30% immiscible components, both MD and TD = random(5-7 MPa) due to phase separation)")
        report.append("- Immiscible materials: Bio-PE, PP, PET, PA, EVOH (all grades)")
        report.append("- Piecewise thickness scaling:")
        report.append("  * Below 25μm: TS * (thickness^-0.3 / 25^-0.3) - thinner = stronger")
        report.append("  * Above 25μm: TS * (thickness^0.2 / 25^0.2) - thicker = stronger")
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
        report.append("- Thickness scaling: power law of 0.4 (EAB increases with thickness)")
        report.append("- Random thickness generation: 10-300μm")
        report.append("- Added 5% Gaussian noise")
    elif property_name == 'eol':
        report.append("- Complex blending rules for max_L (disintegration) and t0 (time to 50% disintegration):")
        report.append("  * Rule 1: All home-compostable polymers (max_L > 90) → max_L = random(90-95), t0 = weighted average")
        report.append("  * Rule 2: PLA + compostable polymer (≥15%) → max_L excludes PLA, t0 includes PLA")
        report.append("  * Default: Rule of mixtures for both properties")
        report.append("- Noise: ±5% for max_L, ±10% for t0")
        report.append("- Two properties: property1 (max_L), property2 (t0)")
    elif property_name == 'seal':
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


def scale_with_temperature(value: float, temperature: float, reference_temp: float = 23, max_scale: float = 5, divisor: float = 10) -> float:
    """Scale value logarithmically with temperature with upper bound"""
    return value * min(max_scale, 1 + np.log1p((temperature - reference_temp) / divisor))


def scale_with_humidity(value: float, rh: float, reference_rh: float = 50, max_scale: float = 3, divisor: float = 20) -> float:
    """Scale value logarithmically with humidity with upper bound"""
    return value * min(max_scale, 1 + np.log1p((rh - reference_rh) / divisor))


def scale_with_temperature_power_law(value: float, temperature: float, reference_temp: float = 23, power_law: float = 0.5) -> float:
    """Scale value with temperature using power law scaling"""
    return value * ((temperature ** power_law) / (reference_temp ** power_law))


def scale_with_humidity_power_law(value: float, rh: float, reference_rh: float = 50, power_law: float = 0.3) -> float:
    """Scale value with humidity using power law scaling"""
    return value * ((rh ** power_law) / (reference_rh ** power_law))


def scale_with_piecewise_thickness(base_value: float, thickness: float, thin_regime: Dict[str, Any],
                                 thick_regime: Dict[str, Any]) -> float:
    """Scale property using piecewise thickness scaling with different power laws for thin/thick regimes."""
    
    thin_max = thin_regime['max_thickness']
    thick_min = thick_regime['min_thickness']
    
    if thickness <= thin_max:
        # Thin regime: thinner = higher value (negative power law)
        power_law = thin_regime['power_law']
        reference = thin_regime['reference']
        return base_value * ((thickness ** power_law) / (reference ** power_law))
    else:
        # Thick regime: thicker = higher value (positive power law)
        power_law = thick_regime['power_law']
        reference = thick_regime['reference']
        return base_value * ((thickness ** power_law) / (reference ** power_law))


def run_simulation_for_property(property_name: str, target_total: int, 
                               property_config: Dict[str, Any], selected_rules: Dict[str, bool] = None,
                               additive_probability: float = 0.3, enable_additives: bool = True,
                               disable_ts_model: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run simulation for a single property using the property configuration"""
    # Get terminal-appropriate colors
    colors = get_terminal_colors()
    
    print(f"\n{colors['BLUE']}{'='*60}{colors['RESET']}")
    print(f"{colors['GREEN']}{colors['BOLD']}Starting {property_config['name']} simulation...{colors['RESET']}")
    print(f"{colors['BLUE']}{'='*60}{colors['RESET']}")
    
    # Load data
    print(f"{colors['CYAN']}Loading data...{colors['RESET']}")
    original_data = property_config['load_data_func']()
    smiles_dict = load_material_smiles_dict()
    
    print(f"{colors['CYAN']}Creating polymer list from original data...{colors['RESET']}")
    
    # Use the property-specific material mapping function (restored from original design)
    material_mapping = property_config['create_material_mapping'](enable_additives)
    available_polymers = list(material_mapping.values())
    
    print(f"{colors['GREEN']}Found {len(available_polymers)} unique polymer grades{colors['RESET']}")
    
    # Run augmentation loop
    augmented_data, rule_tracker = run_augmentation_loop(
        property_name=property_name,
        available_polymers=available_polymers,
        target_total=target_total,
        create_blend_row_func=property_config['create_blend_row_func'],
        progress_interval=1000,
        selected_rules=selected_rules,
        additive_probability=additive_probability,
        enable_additives=enable_additives,
        disable_ts_model=disable_ts_model
    )
    
    # Combine with original data
    combined_data = combine_with_original_data(original_data, augmented_data)
    
    # Save augmented data
    save_augmented_data(augmented_data, property_name)
    
    # Create ML dataset
    print(f"\n{colors['CYAN']}Creating ML dataset...{colors['RESET']}")
    ml_dataset = create_ml_dataset(combined_data, property_name)
    
    # Display rule usage summary
    print(rule_tracker.get_summary())
    
    # Generate and save simple report
    report = generate_simple_report(property_name, original_data, augmented_data)
    
    # Add rule usage summary to report
    report += rule_tracker.get_summary()
    
    with open(f'train/simulation/reports/{property_name}_augmentation_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\n{colors['GREEN']}{colors['BOLD']}Augmentation complete!{colors['RESET']}")
    print(f"{colors['BLUE']}Report saved to train/simulation/reports/{property_name}_augmentation_report.txt{colors['RESET']}")
    
    # Run validation on optimization results using the SAME pipeline
    try:
        # Load UMM3 correction system for validation (same as main simulation)
        umm3_correction = None
        polymer_corrections_config = None
        family_compatibility_config = None
        
        try:
            polymer_corrections_config = load_polymer_corrections_config()
            import os
            if os.path.exists("config"):
                umm3_correction = UMM3Correction.from_config_files("config")
            elif os.path.exists("train/simulation/config"):
                umm3_correction = UMM3Correction.from_config_files("train/simulation/config")
            else:
                raise FileNotFoundError("Could not find config directory")
            
            family_compatibility_config = load_family_compatibility_config(property_name)
        except Exception as e:
            print(f"{colors['YELLOW']}Warning: Could not load UMM3 correction system for validation: {e}{colors['RESET']}")
            umm3_correction = None
            polymer_corrections_config = None
            family_compatibility_config = None
        
        validation_results = validate_simulation_pipeline(
            property_name=property_name,
            property_config=property_config,
            material_mapping=material_mapping,
            umm3_correction=umm3_correction,
            polymer_corrections_config=polymer_corrections_config,
            family_compatibility_config=family_compatibility_config
        )
        
        if "error" not in validation_results:
            print(f"\n{colors['GREEN']}✅ Simulation pipeline validation completed!{colors['RESET']}")
            print(f"{colors['CYAN']}   MAE: {validation_results['mae']:.4f}{colors['RESET']}")
            print(f"{colors['CYAN']}   Accuracy: {validation_results['accuracy']:.2f}%{colors['RESET']}")
        else:
            print(f"\n{colors['YELLOW']}⚠️  Validation skipped: {validation_results['error']}{colors['RESET']}")
    except Exception as e:
        print(f"\n{colors['YELLOW']}⚠️  Validation failed: {e}{colors['RESET']}")
    
    # Prepare simulation summary data for Streamlit
    simulation_summary = {
        "property_name": property_name,
        "property_display_name": property_config['name'],
        "target_total": target_total,
        "original_data_count": len(original_data),
        "augmented_data_count": len(augmented_data),
        "combined_data_count": len(combined_data),
        "available_polymers": len(available_polymers),
        "rule_usage": rule_tracker.get_rule_data(),
        "status": "completed"
    }
    
    return combined_data, augmented_data, ml_dataset, simulation_summary


def validate_simulation_pipeline(property_name: str, property_config: Dict[str, Any], 
                                material_mapping: Dict[str, Any], umm3_correction: UMM3Correction,
                                polymer_corrections_config: Dict[str, Any], 
                                family_compatibility_config: Dict[str, Any] = None) -> Dict[str, float]:
    """
    Validate simulation pipeline using validationblends.csv to ensure it matches optimization results.
    This uses the EXACT same pipeline as the main simulation.
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    
    colors = get_terminal_colors()
    
    print(f"\n{colors['CYAN']}🔍 Running simulation pipeline validation...{colors['RESET']}")
    
    # Load validation data
    validation_csv = f"train/data/{property_name}/validationblends.csv"
    if not os.path.exists(validation_csv):
        print(f"{colors['YELLOW']}⚠️  Validation CSV not found: {validation_csv}{colors['RESET']}")
        return {"error": "Validation CSV not found"}
    
    validation_df = pd.read_csv(validation_csv)
    print(f"{colors['CYAN']}📊 Loaded {len(validation_df)} validation blends{colors['RESET']}")
    
    # Check if material mapping is valid
    if not material_mapping:
        print(f"{colors['YELLOW']}⚠️  Material mapping is empty or None{colors['RESET']}")
        return {"error": "Material mapping is empty"}
    
    # Simulate each validation blend using the EXACT same pipeline as main simulation
    predicted_values = []
    actual_values = []
    blend_details = []
    
    for idx, blend_row in validation_df.iterrows():
        try:
            # Extract polymer data using the SAME logic as main simulation
            polymer_data_list = []
            compositions = []
            
            for i in range(1, 6):  # Polymer Grade 1-5
                grade_col = f'Polymer Grade {i}'
                vol_frac_col = f'vol_fraction{i}'
                
                if pd.notna(blend_row[grade_col]) and pd.notna(blend_row[vol_frac_col]):
                    grade = blend_row[grade_col]
                    vol_frac = blend_row[vol_frac_col]
                    
                    # Look up by grade name in material mapping (same as main simulation)
                    found_polymer = None
                    if material_mapping:
                        for key, polymer_data in material_mapping.items():
                            if polymer_data and polymer_data.get('grade') == grade:
                                found_polymer = polymer_data
                                break
                    
                    if found_polymer:
                        polymer_data_list.append(found_polymer)
                        compositions.append(float(vol_frac))
                    else:
                        continue
            
            if len(polymer_data_list) < 2:
                continue
            
            # Normalize compositions
            total_comp = sum(compositions)
            compositions = [c / total_comp for c in compositions]
            
            # Load environmental config EXACTLY like main simulation - NO FALLBACKS
            try:
                environmental_controls_config = load_environmental_controls_config()
                if property_name not in environmental_controls_config:
                    raise ValueError(f"No environmental controls found for property: {property_name}")
                
                # Use the SAME environmental config as main simulation
                env_config = {property_name: environmental_controls_config[property_name].copy()}
                
                # Override thickness with validation data (but keep all other optimized parameters)
                validation_thickness = blend_row['Thickness (um)']
                if 'thickness' in env_config[property_name]:
                    env_config[property_name]['thickness']['min'] = validation_thickness
                    env_config[property_name]['thickness']['max'] = validation_thickness
                else:
                    raise ValueError(f"No thickness config found for property: {property_name}")
                
                # Handle temperature and humidity for WVTR/OTR properties
                if property_name in ['wvtr', 'otr']:
                    validation_temp = blend_row['Temperature (C)']
                    validation_rh = blend_row['RH (%)']
                    
                    if 'temperature' in env_config[property_name]:
                        env_config[property_name]['temperature']['min'] = validation_temp
                        env_config[property_name]['temperature']['max'] = validation_temp
                    else:
                        raise ValueError(f"No temperature config found for property: {property_name}")
                    
                    if 'humidity' in env_config[property_name]:
                        env_config[property_name]['humidity']['min'] = validation_rh
                        env_config[property_name]['humidity']['max'] = validation_rh
                    else:
                        raise ValueError(f"No humidity config found for property: {property_name}")
                        
            except Exception as e:
                raise RuntimeError(f"Failed to load environmental config for validation: {e}")
            
            # Create blend row using the EXACT SAME function as main simulation
            create_blend_row_func = property_config['create_blend_row_func']
            blend_row_data = create_blend_row_func(
                polymers=polymer_data_list,
                compositions=compositions,
                blend_number=idx + 1,
                rule_tracker=None,
                selected_rules=None,
                environmental_config=env_config
            )
            
            # Apply UMM3 corrections using the EXACT SAME logic as main simulation
            property_values = {}
            for key, value in blend_row_data.items():
                if key.startswith('property') and isinstance(value, (int, float)):
                    property_values[key] = value
            
            if property_values:
                corrected_property_values = apply_umm3_corrections(
                    property_values, property_name, polymer_data_list, compositions, 
                    umm3_correction, None, polymer_corrections_config,
                    family_compatibility_config
                )
                
                # Update the blend_row_data with corrected values
                for key, value in corrected_property_values.items():
                    if key != 'corrections_applied':
                        blend_row_data[key] = value
            
            # Extract predicted value (same logic as optimization)
            if property_name in ['ts', 'eab']:
                predicted_value = blend_row_data.get('property1', 0.0)
            else:
                predicted_value = blend_row_data.get('property', 0.0)
            
            # Extract actual value
            if property_name in ['ts', 'eab']:
                actual_value = blend_row.get('property1', 0.0)
            else:
                actual_value = blend_row.get('property', 0.0)
            
            predicted_values.append(predicted_value)
            actual_values.append(actual_value)
            
            blend_details.append({
                'blend_id': idx + 1,
                'predicted': predicted_value,
                'actual': actual_value,
                'error': abs(predicted_value - actual_value),
                'pct_error': abs(predicted_value - actual_value) / actual_value * 100 if actual_value != 0 else 0
            })
            
        except Exception as e:
            print(f"{colors['YELLOW']}⚠️  Error simulating blend {idx + 1}: {e}{colors['RESET']}")
            continue
    
    if not predicted_values:
        print(f"❌ No valid predictions generated!")
        return {"error": "No valid predictions"}
    
    # Calculate metrics
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    
    mae = np.mean(np.abs(predicted_values - actual_values))
    mse = np.mean((predicted_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    
    # Calculate accuracy (percentage within 20% of actual)
    accuracy_mask = np.abs(predicted_values - actual_values) / actual_values <= 0.2
    accuracy = np.mean(accuracy_mask) * 100
    
    print(f"\n{colors['GREEN']}📊 SIMULATION PIPELINE VALIDATION RESULTS:{colors['RESET']}")
    print(f"{colors['CYAN']}   MAE: {mae:.4f}{colors['RESET']}")
    print(f"{colors['CYAN']}   RMSE: {rmse:.4f}{colors['RESET']}")
    print(f"{colors['CYAN']}   Accuracy (within 20%): {accuracy:.2f}%{colors['RESET']}")
    print(f"{colors['CYAN']}   Blends processed: {len(predicted_values)}{colors['RESET']}")
    
    # Create validation performance plot (same style as optimization)
    create_validation_performance_plot(actual_values, predicted_values, property_name)
    
    # Save detailed results
    results_df = pd.DataFrame(blend_details)
    results_file = f"{property_name}_simulation_validation_detailed_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"{colors['GREEN']}📊 Detailed results saved to: {results_file}{colors['RESET']}")
    
    return {
        "mae": mae,
        "rmse": rmse,
        "accuracy": accuracy,
        "n_blends": len(predicted_values)
    }


def create_validation_performance_plot(actual_vals: np.ndarray, predicted_vals: np.ndarray, property_name: str):
    """Create validation performance plot matching optimization style"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predicted vs Actual
    ax1.scatter(actual_vals, predicted_vals, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(actual_vals.min(), predicted_vals.min())
    max_val = max(actual_vals.max(), predicted_vals.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Values')
    ax1.set_ylabel('Predicted Values')
    
    # Calculate MAE for display
    mae = np.mean(np.abs(predicted_vals - actual_vals))
    ax1.set_title(f'{property_name.upper()} Simulation Pipeline Validation: Predicted vs Actual\nMAE: {mae:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = predicted_vals - actual_vals
    ax2.scatter(actual_vals, residuals, alpha=0.6, s=50)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Actual Values')
    ax2.set_ylabel('Residuals (Predicted - Actual)')
    ax2.set_title(f'{property_name.upper()} Simulation Pipeline Validation: Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{property_name}_simulation_validation_performance.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Validation performance plot saved to: {plot_file}")


def run_all_simulations(target_total: int = 5000, seed: int = 42, 
                       properties_to_run: List[str] = None, additive_probability: float = 0.3,
                       enable_additives: bool = True) -> Dict[str, Tuple]:
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
                    property_config=PROPERTY_RULES[property_name],
                    additive_probability=additive_probability,
                    enable_additives=enable_additives
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
