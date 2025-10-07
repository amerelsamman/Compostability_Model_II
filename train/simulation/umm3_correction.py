#!/usr/bin/env python3
"""
UMM3 (Universal Material Modification Model) Correction Layer
Universal correction wrapper for additives/fillers on top of existing RoM/EMT framework.
"""

import math
import os
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict

# Try to import yaml, fall back to hardcoded configs if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


class ClippingTracker:
    """Track clipping statistics for UMM3 corrections."""
    
    def __init__(self):
        self.clipping_stats = defaultdict(lambda: {
            'total_corrections': 0,
            'clipped_corrections': 0,
            'materials_clipped': defaultdict(int),
            'properties_clipped': defaultdict(int),
            'blend_examples': []
        })
    
    def record_correction(self, material: str, property_name: str, was_clipped: bool, 
                         blend_info: Dict[str, Any] = None):
        """Record a correction event."""
        key = f"{material}_{property_name}"
        self.clipping_stats[key]['total_corrections'] += 1
        
        if was_clipped:
            self.clipping_stats[key]['clipped_corrections'] += 1
            self.clipping_stats[key]['materials_clipped'][material] += 1
            self.clipping_stats[key]['properties_clipped'][property_name] += 1
            
            # Store blend example (limit to 5 per material-property combo)
            if len(self.clipping_stats[key]['blend_examples']) < 5 and blend_info:
                self.clipping_stats[key]['blend_examples'].append(blend_info)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get clipping summary statistics."""
        summary = {}
        for key, stats in self.clipping_stats.items():
            if stats['total_corrections'] > 0:
                clip_rate = stats['clipped_corrections'] / stats['total_corrections']
                summary[key] = {
                    'total_corrections': stats['total_corrections'],
                    'clipped_corrections': stats['clipped_corrections'],
                    'clip_rate': clip_rate,
                    'materials_clipped': dict(stats['materials_clipped']),
                    'properties_clipped': dict(stats['properties_clipped']),
                    'blend_examples': stats['blend_examples']
                }
        return summary


class UMM3Correction:
    """
    Universal correction layer for additives and fillers.
    Applies exponential correction factor to base RoM/EMT property values.
    """
    
    def __init__(self, weights: Dict[str, Dict[str, float]], 
                 shapes: Dict[str, float], 
                 clips: Dict[str, Dict[str, float]],
                 clipping_tracker: Optional[ClippingTracker] = None):
        """
        Initialize UMM3 correction with configuration parameters.
        
        Args:
            weights: dict[property] -> {wS, wT, wI} - per-property effect weights
            shapes: dict -> {aS, aT, bT} - universal shape function parameters
            clips: dict[property] -> {lo, hi} - per-property clipping ranges
            clipping_tracker: optional tracker for clipping statistics
        """
        self.weights = weights
        self.aS = shapes.get("aS", 25.0)
        self.aT = shapes.get("aT", 15.0)
        self.bT = shapes.get("bT", 0.02)
        self.clips = clips
        self.clipping_tracker = clipping_tracker or ClippingTracker()
    
    @classmethod
    def from_config_files(cls, config_dir: str = None):
        """Load configuration from YAML files."""
        if not YAML_AVAILABLE:
            raise ImportError("YAML module not available. Please install pyyaml: pip install pyyaml")
        
        # Set default config_dir relative to this file's location
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "config")
        
        weights_path = os.path.join(config_dir, "weights.yaml")
        shapes_path = os.path.join(config_dir, "shapes.yaml")
        clips_path = os.path.join(config_dir, "clips.yaml")
        
        with open(weights_path, 'r') as f:
            weights_config = yaml.safe_load(f)
        with open(shapes_path, 'r') as f:
            shapes_config = yaml.safe_load(f)
        with open(clips_path, 'r') as f:
            clips_config = yaml.safe_load(f)
        
        return cls(weights_config["weights"], shapes_config["shapes"], clips_config["clips"])
    
    @classmethod
    
    # Universal shape functions
    def f_S(self, phi: float) -> float:
        """Structure shape function: f_S(phi) = 1 - exp(-aS * phi)"""
        return 1.0 - math.exp(-self.aS * phi)
    
    def f_T(self, phi: float, G: float) -> float:
        """Transport shape function: f_T(phi, G) = (1 - exp(-aT * phi)) * (1 + bT * G * phi)"""
        return (1.0 - math.exp(-self.aT * phi)) * (1.0 + self.bT * G * phi)
    
    def f_I(self, phi: float) -> float:
        """Interface shape function: f_I(phi) = phi * (1 - phi)"""
        return phi * (1.0 - phi)
    
    def adjust_property(self, X_rom: float, phi: float, ingredient: Dict[str, Any], 
                       prop: str, material_name: str = "", blend_info: Dict[str, Any] = None) -> Tuple[float, float, bool]:
        """
        Apply UMM3 correction to a property value.
        
        Args:
            X_rom: base value from existing RoM/EMT framework
            phi: loading fraction (0..1)
            ingredient: dict with property-specific K parameters (K_ts, K_wvtr, etc.) and G from polymer_corrections.yaml
            prop: property name ('tensile', 'elongation', 'otr', 'wvtr', 'seal', 'cobb', 'compost')
            material_name: name of the material for tracking
            blend_info: additional blend information for tracking
        
        Returns:
            Tuple of (adjusted_value, log_factor, was_clipped)
        """
        # Extract ingredient parameters (Ki removed - now handled by pairwise system)
        G = float(ingredient.get("G", 0.0))
        
        # Get property-specific K parameter
        property_k_map = {
            'tensile': 'K_ts',
            'elongation': 'K_eab', 
            'otr': 'K_otr',
            'wvtr': 'K_wvtr',
            'seal': 'K_seal',
            'cobb': 'K_cobb',
            'compost': 'K_compost'
        }
        
        if prop not in property_k_map:
            raise ValueError(f"Unknown property: {prop}. Available: {list(property_k_map.keys())}")
        
        K_prop = float(ingredient.get(property_k_map[prop], 0.0))
        
        # Get property-specific weights
        if prop not in self.weights:
            raise ValueError(f"Unknown property: {prop}. Available: {list(self.weights.keys())}")
        
        w = self.weights[prop]
        
        # Calculate log factor (only transport effect, interface handled by pairwise system)
        logfac = w["wT"] * K_prop * self.f_T(phi, G)
        
        # Apply clipping
        if prop not in self.clips:
            raise ValueError(f"No clipping range defined for property: {prop}")
        
        lo, hi = self.clips[prop]["lo"], self.clips[prop]["hi"]
        original_logfac = logfac
        logfac = max(lo, min(hi, logfac))
        was_clipped = (logfac != original_logfac)
        
        # Apply correction
        X_adj = X_rom * math.exp(logfac)
        
        # Record clipping statistics
        if material_name:
            self.clipping_tracker.record_correction(
                material_name, prop, was_clipped, blend_info
            )
        
        return X_adj, logfac, was_clipped
    
    def get_clipping_summary(self) -> Dict[str, Any]:
        """Get clipping statistics summary."""
        return self.clipping_tracker.get_summary()
    
    def apply_pairwise_compatibility_corrections(self, property_values: Dict[str, Any], 
                                               polymers: List[Dict], compositions: List[float],
                                               family_config: Dict[str, Dict[str, Any]], 
                                               property_name: str = None) -> Dict[str, Any]:
        """Apply pairwise interfacial compatibility corrections"""
        
        # Calculate pairwise interface contributions (including self-interactions)
        interface_contribution = 0.0
        
        for i in range(len(polymers)):
            for j in range(len(polymers)):  # Include j=i for self-interactions
                material_i = polymers[i]['material']
                material_j = polymers[j]['material']
                phi_i = compositions[i]
                phi_j = compositions[j]
                
                # Get KI for this material family pair (including self-interactions)
                # material_i and material_j are already family names from the polymer dictionary
                family_i = material_i
                family_j = material_j
                KI_ij = get_family_compatibility(family_i, family_j, family_config)
                
                # Calculate interface function for this pair
                if i == j:
                    # Self-interaction: use phi_i^2
                    f_I_pair = phi_i * phi_i
                else:
                    # Cross-interaction: use phi_i * phi_j
                    f_I_pair = phi_i * phi_j
                
                interface_contribution += KI_ij * f_I_pair
        
        # Map property names to UMM3 property names based on the property being simulated
        if property_name == 'wvtr':
            property_mapping = {
                'property1': 'wvtr',
                'property2': 'wvtr',
                'property': 'wvtr'
            }
        elif property_name == 'ts':
            property_mapping = {
                'property1': 'tensile',
                'property2': 'tensile'
            }
        elif property_name == 'eab':
            property_mapping = {
                'property1': 'elongation',
                'property2': 'elongation'
            }
        elif property_name == 'otr':
            property_mapping = {
                'property1': 'otr',
                'property2': 'otr',
                'property': 'otr'
            }
        elif property_name == 'adhesion':
            property_mapping = {
                'property1': 'seal',
                'property2': 'seal'
            }
        elif property_name == 'seal':
            property_mapping = {
                'property1': 'seal',
                'property2': 'seal',
                'property': 'seal'
            }
        elif property_name == 'cobb':
            property_mapping = {
                'property1': 'cobb',
                'property2': 'cobb',
                'property': 'cobb'
            }
        else:
            # Default mapping
            property_mapping = {
                'property1': 'tensile',
                'property2': 'tensile',
                'ts1': 'tensile',
                'ts2': 'tensile',
                'eab1': 'elongation',
                'eab2': 'elongation',
                'max_L': 'elongation',
                't0': 'elongation',
                'otr': 'otr',
                'wvtr': 'wvtr',
                'adhesion': 'seal',
                'cobb': 'cobb'
            }
        
        # Apply to property values
        corrected_values = {}
        for prop_key, prop_value in property_values.items():
            if isinstance(prop_value, (int, float)) and prop_value > 0:
                # Get UMM3 property name
                umm3_prop = property_mapping.get(prop_key, prop_key)
                
                # Get interface weight for this property
                if umm3_prop in self.weights and 'wI' in self.weights[umm3_prop]:
                    wI = self.weights[umm3_prop]['wI']
                    log_factor = wI * interface_contribution
                    
                    # Apply clipping (same as individual corrections)
                    if umm3_prop in self.clips:
                        lo, hi = self.clips[umm3_prop]["lo"], self.clips[umm3_prop]["hi"]
                        log_factor = max(lo, min(hi, log_factor))
                    
                    corrected_value = prop_value * math.exp(log_factor)
                    corrected_values[prop_key] = corrected_value
                else:
                    corrected_values[prop_key] = prop_value
            else:
                corrected_values[prop_key] = prop_value
        
        return corrected_values
    
    def _extract_family_name(self, material_name: str) -> str:
        """
        Extract polymer family name from material name using material-smiles-dictionary.csv.
        NO DEFAULTS - must be explicitly configured.
        """
        import csv
        import os
        
        # Load the material dictionary CSV - use absolute path from project root
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "material-smiles-dictionary.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Material dictionary not found at {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # First check if material_name is already a family name
                if row['Material'] == material_name:
                    return material_name
                # Then check if it's a grade name
                elif row['Grade'] == material_name:
                    return row['Material']
        
        # NO DEFAULTS - Error out if material not found
        raise KeyError(f"Material '{material_name}' not found in material-smiles-dictionary.csv")
    
    def get_correction_info(self, phi: float, ingredient: Dict[str, Any], 
                           prop: str) -> Dict[str, Any]:
        """
        Get detailed correction information without applying it.
        
        Returns:
            Dict with correction details for analysis/debugging
        """
        Ks = float(ingredient.get("Ks", 0.0))
        Kt = float(ingredient.get("Kt", 0.0))
        # Ki removed - interface effects now handled by pairwise system
        G = float(ingredient.get("G", 0.0))
        
        w = self.weights[prop]
        
        f_S_val = self.f_S(phi)
        f_T_val = self.f_T(phi, G)
        # f_I_val removed - interface effects now handled by pairwise system
        
        # Individual contributions (interface effects now handled by pairwise system)
        struct_contrib = w["wS"] * Ks * f_S_val
        transport_contrib = w["wT"] * Kt * f_T_val
        # interface_contrib removed - now handled by pairwise compatibility system
        
        logfac = struct_contrib + transport_contrib
        
        # Apply clipping
        lo, hi = self.clips[prop]["lo"], self.clips[prop]["hi"]
        original_logfac = logfac
        logfac = max(lo, min(hi, logfac))
        was_clipped = (logfac != original_logfac)
        
        return {
            "phi": phi,
            "Ks": Ks, "Kt": Kt, "G": G,
            "f_S": f_S_val, "f_T": f_T_val, "f_I": f_I_val,
            "w_S": w["wS"], "w_T": w["wT"], "w_I": w["wI"],
            "struct_contrib": struct_contrib,
            "transport_contrib": transport_contrib,
            "interface_contrib": interface_contrib,
            "logfac": logfac,
            "was_clipped": was_clipped,
            "original_logfac": original_logfac
        }


def load_ingredients_config(config_dir: str = "train/simulation/config") -> Dict[str, Dict[str, Any]]:
    """Load ingredients configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("YAML module not available. Please install pyyaml: pip install pyyaml")
    
    ingredients_path = os.path.join(config_dir, "ingredients.yaml")
    if not os.path.exists(ingredients_path):
        raise FileNotFoundError(f"Ingredients config not found at {ingredients_path}")
    
    with open(ingredients_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["ingredients"]




def load_polymer_corrections_config(config_dir: str = "train/simulation/config") -> Dict[str, Dict[str, Any]]:
    """Load polymer corrections configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("YAML module not available. Please install pyyaml: pip install pyyaml")
    
    corrections_path = os.path.join(config_dir, "polymer_corrections.yaml")
    if not os.path.exists(corrections_path):
        raise FileNotFoundError(f"Polymer corrections config not found at {corrections_path}")
    
    with open(corrections_path, 'r') as f:
        config = yaml.safe_load(f)
    return config["polymer_corrections"]

def load_family_compatibility_config(config_dir: str = "train/simulation/config", property_name: str = None) -> Dict[str, Dict[str, Any]]:
    """Load material family compatibility configuration"""
    if not YAML_AVAILABLE:
        raise ImportError("YAML module not available. Please install pyyaml: pip install pyyaml")
    
    if not property_name:
        raise ValueError("property_name is required. No fallbacks allowed - specify the exact property.")
    
    # Load property-specific compatibility file - NO FALLBACKS
    compatibility_path = os.path.join(config_dir, f"{property_name}_compatibility.yaml")
    if not os.path.exists(compatibility_path):
        raise FileNotFoundError(f"Property-specific compatibility file not found: {compatibility_path}. Create this file for property '{property_name}'.")
    
    with open(compatibility_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if "material_family_compatibility" not in config:
        raise KeyError(f"Missing 'material_family_compatibility' key in {compatibility_path}")
    
    return config["material_family_compatibility"]

def load_ingredient_polymer_compatibility_config(config_dir: str = "train/simulation/config") -> Dict[str, Dict[str, Any]]:
    """Load ingredient-polymer compatibility configuration from YAML file."""
    if not YAML_AVAILABLE:
        raise ImportError("YAML module not available. Please install pyyaml: pip install pyyaml")
    
    ingredient_polymer_path = os.path.join(config_dir, "ingredient_polymer.yaml")
    if not os.path.exists(ingredient_polymer_path):
        raise FileNotFoundError(f"Ingredient-polymer compatibility file not found: {ingredient_polymer_path}")
    
    with open(ingredient_polymer_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if "additive_compatibility" not in config:
        raise KeyError(f"Missing 'additive_compatibility' key in {ingredient_polymer_path}")
    
    return config["additive_compatibility"]

def get_family_compatibility(material1: str, material2: str, family_config: Dict[str, Dict[str, Any]]) -> float:
    """Get KI for material family pair (including self-interactions)"""
    
    # Create pair key (sorted for consistency)
    pair_key = f"{material1}-{material2}"
    reverse_key = f"{material2}-{material1}"
    
    # Try both directions
    if pair_key in family_config:
        return family_config[pair_key]['KI']
    elif reverse_key in family_config:
        return family_config[reverse_key]['KI']
    
    # DEFAULT BEHAVIOR - Return 0.0 with warning if pair not found
    import warnings
    warnings.warn(f"Material family pair '{material1}-{material2}' not found in compatibility config. Using default KI=0.0 (no additional pairwise effect).", UserWarning)
    return 0.0

def get_additive_polymer_compatibility(additive_name: str, polymer_family: str, ingredient_polymer_config: Dict[str, Dict[str, Any]]) -> float:
    """Get KI for additive-polymer pair from ingredient_polymer.yaml"""
    
    if additive_name not in ingredient_polymer_config:
        raise KeyError(f"Additive '{additive_name}' not found in ingredient-polymer compatibility config")
    
    if polymer_family not in ingredient_polymer_config[additive_name]:
        raise KeyError(f"Polymer family '{polymer_family}' not found for additive '{additive_name}' in ingredient-polymer compatibility config")
    
    return float(ingredient_polymer_config[additive_name][polymer_family])

# Example usage and testing
if __name__ == "__main__":
    # Test the correction system
    weights = {
        "tensile": {"wS": 0.30, "wT": 0.05, "wI": 0.10},
        "elongation": {"wS": -0.60, "wT": 0.00, "wI": 0.30},
        "otr": {"wS": -0.20, "wT": -0.60, "wI": -0.20},
        "wvtr": {"wS": -0.10, "wT": -0.70, "wI": -0.15},
        "seal": {"wS": -0.40, "wT": 0.10, "wI": 0.35},
        "cobb": {"wS": 0.00, "wT": -0.80, "wI": -0.25},
    }
    shapes = {"aS": 25.0, "aT": 15.0, "bT": 0.02}
    clips = {
        "tensile": {"lo": -0.30, "hi": 0.30},
        "elongation": {"lo": -0.20, "hi": 2.30},
        "otr": {"lo": -1.50, "hi": 0.80},
        "wvtr": {"lo": -1.50, "hi": 0.80},
        "seal": {"lo": -0.70, "hi": 0.70},
        "cobb": {"lo": -0.50, "hi": 1.20},
    }
    
    umm3 = UMM3Correction(weights, shapes, clips)
    
    # Test with ADR4300 additive
    ingredient = {"Ks": 0.70, "Kt": 0.20, "G": 0}
    phi = 0.007
    X_rom_tensile = 58.2
    
    X_adj, logfac, clipped = umm3.adjust_property(X_rom_tensile, phi, ingredient, "tensile")
    print(f"Tensile: {X_rom_tensile:.2f} -> {X_adj:.2f} (logfac={logfac:.4f}, clipped={clipped})")
    
    # Test with nanoclay filler
    ingredient_filler = {"Ks": 0.40, "Kt": 0.80, "G": 200}
    phi_filler = 0.15
    
    X_adj_filler, logfac_filler, clipped_filler = umm3.adjust_property(
        X_rom_tensile, phi_filler, ingredient_filler, "tensile"
    )
    print(f"Tensile (nanoclay): {X_rom_tensile:.2f} -> {X_adj_filler:.2f} (logfac={logfac_filler:.4f}, clipped={clipped_filler})")
    
    # Get detailed correction info
    info = umm3.get_correction_info(phi_filler, ingredient_filler, "tensile")
    print(f"\nDetailed correction info:")
    print(f"  Structure contribution: {info['struct_contrib']:.4f}")
    print(f"  Transport contribution: {info['transport_contrib']:.4f}")
    print(f"  Interface contribution: {info['interface_contrib']:.4f}")
    print(f"  Total log factor: {info['logfac']:.4f}")
