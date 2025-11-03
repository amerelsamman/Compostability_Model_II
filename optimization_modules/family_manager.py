"""
Family Management Module
Functions for handling polymer families and family groups
"""

import pandas as pd
import numpy as np
from typing import Dict, List


def extract_polymer_families_from_blend(blend_row: pd.Series, family_mapping: Dict[str, str]) -> List[str]:
    """Extract polymer families from a blend row."""
    families = []
    missing_grades = []
    
    for i in range(1, 6):  # Polymer Grade 1-5
        grade_col = f'Polymer Grade {i}'
        vol_frac_col = f'vol_fraction{i}'
        
        # Only process if grade exists AND has a volume fraction > 0
        if pd.notna(blend_row[grade_col]) and pd.notna(blend_row[vol_frac_col]) and blend_row[vol_frac_col] > 0:
            grade = blend_row[grade_col]
            
            # Skip 'Unknown' grades
            if grade == 'Unknown':
                continue
                
            if grade in family_mapping:
                family = family_mapping[grade]
                if family not in families:
                    families.append(family)
            else:
                missing_grades.append(grade)
    
    # THROW ERROR if any grades are missing from family mapping
    if missing_grades:
        blend_name = blend_row.get('Materials', 'Unknown')
        raise ValueError(f"❌ BLEND {blend_name}: Missing grades in family mapping: {missing_grades}")
    
    return families


def get_polymer_family_groups() -> Dict[str, List[str]]:
    """Define polymer family groups for optimization."""
    return {
        "rigids": ["PLA", "PGA", "PHAs"],
        "brittles": ["PHB", "PHA", "PHBV"], 
        "soft_flex": ["PHBH", "PHAs","PHAa"],
        "good_flex": ["PBAT", "PCL", "PBS","PBSA"],
        "bio_pe": ["Bio-PE"],
        "traditional": ["LDPE", "PP", "PET", "PVDC", "PA", "EVOH"]
    }


def get_family_group_for_polymer(polymer_family: str, family_groups: Dict[str, List[str]]) -> str:
    """Get the family group for a given polymer family."""
    for group_name, families in family_groups.items():
        if polymer_family in families:
            return group_name
    return polymer_family


def get_ki_overrides_from_vector(ki_vector: np.ndarray, pair_mapping: Dict[str, int], 
                                families_in_blend: List[str]) -> Dict[str, float]:
    """Get KI overrides for a specific blend from the KI vector."""
    ki_overrides = {}
    missing_pairs = []
    
    # Create all possible pairs from families in this blend
    for i in range(len(families_in_blend)):
        for j in range(i + 1, len(families_in_blend)):
            family1, family2 = sorted([families_in_blend[i], families_in_blend[j]])
            pair_name = f"{family1}-{family2}"
            
            if pair_name in pair_mapping:
                ki_value = ki_vector[pair_mapping[pair_name]]
                ki_overrides[pair_name] = ki_value
            else:
                missing_pairs.append(pair_name)
    
    if missing_pairs:
        raise ValueError(f"❌ Missing pairs in KI vector: {missing_pairs}. "
                        f"Available pairs: {list(pair_mapping.keys())}")
    
    return ki_overrides
