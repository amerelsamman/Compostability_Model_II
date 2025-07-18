import pandas as pd
import numpy as np
import re
from scipy.stats import dirichlet

# --- GLOBAL SEED FOR REPRODUCIBILITY ---
GLOBAL_SEED = 42

def set_global_seed(seed=None):
    """Set global seed for all random operations"""
    if seed is None:
        seed = GLOBAL_SEED
    np.random.seed(seed)
    print(f"🌱 Set global seed to {seed} for reproducible results")

# Set initial seed
set_global_seed(GLOBAL_SEED)

# --- CONFIGURABLE PARAMETERS ---
DAYS = 200  # Number of days for the time series
SAMPLE_AREA = 100.0  # Default sample area (mm^2)
THICKNESS_DEFAULT = 1.0  # Default thickness (mm)
ACTUAL_THICKNESS_DEFAULT = 0.050  # Default actual thickness (50 μm = 0.050 mm)

# --- LOAD DATA ---
sus = pd.read_csv('sustainability.csv')
sus.columns = [c.strip() for c in sus.columns]

# --- SIGMOID FUNCTION ---
def sigmoid(t, L, k, t0):
    return L / (1 + np.exp(-k * (t - t0)))

def calculate_certification_thickness_scaling(certification_thickness):
    """
    Calculate thickness scaling factor based on certification thickness (original model behavior).
    
    Args:
        certification_thickness: Thickness from certification data (mm)
    
    Returns:
        scaling_factor: Factor to apply to kinetics based on certification thickness
    """
    # Ensure certification thickness is valid
    if np.isnan(certification_thickness) or certification_thickness <= 0:
        certification_thickness = THICKNESS_DEFAULT
    
    # Original model scaling: (certification_thickness / 1.0) ** 0.1
    scaling_factor = (certification_thickness / 1.0) ** 0.1
    
    return scaling_factor

def calculate_actual_thickness_scaling(actual_thickness=None):
    """
    Calculate thickness scaling factor based on actual thickness input.
    
    Args:
        actual_thickness: Actual thickness of material (mm), defaults to 50μm
    
    Returns:
        scaling_factor: Factor to apply to kinetics (50μm = 1.0, thinner = <1.0, thicker = >1.0)
    """
    if actual_thickness is None:
        actual_thickness = ACTUAL_THICKNESS_DEFAULT
    
    # Ensure actual thickness is valid
    if np.isnan(actual_thickness) or actual_thickness <= 0:
        actual_thickness = ACTUAL_THICKNESS_DEFAULT
    
    # Scaling factor: (actual_thickness / 50μm) ** 0.1 (gentler power law)
    # When actual = 50μm: scaling = 1.0
    # When actual < 50μm: scaling < 1.0 (faster kinetics)
    # When actual > 50μm: scaling > 1.0 (slower kinetics)
    scaling_factor = (actual_thickness / ACTUAL_THICKNESS_DEFAULT) ** 0.1
    
    return scaling_factor

def calculate_max_disintegration_modulation(actual_thickness=None):
    """
    Calculate max disintegration modulation based on actual thickness.
    
    Args:
        actual_thickness: Actual thickness of material (mm), defaults to 50μm
    
    Returns:
        modulation_factor: Factor to multiply max disintegration (thinner = higher max)
    """
    if actual_thickness is None:
        actual_thickness = ACTUAL_THICKNESS_DEFAULT
    
    # Ensure actual thickness is valid
    if np.isnan(actual_thickness) or actual_thickness <= 0:
        actual_thickness = ACTUAL_THICKNESS_DEFAULT
    
    # Modulation factor: (50μm / actual_thickness) ** 0.1 (gentler power law)
    # When actual = 50μm: modulation = 1.0
    # When actual < 50μm: modulation > 1.0 (higher max disintegration)
    # When actual > 50μm: modulation < 1.0 (lower max disintegration)
    modulation_factor = (ACTUAL_THICKNESS_DEFAULT / actual_thickness) ** 0.1
    
    # Clamp to reasonable bounds (0.5 to 2.0)
    modulation_factor = np.clip(modulation_factor, 0.5, 2.0)
    
    return modulation_factor

def parse_blend_input(blend_str):
    """Parse blend input string like 'PLA,4032D,0.5,PBAT,Ecoworld,0.5'"""
    parts = [part.strip() for part in blend_str.split(',')]
    if len(parts) % 3 != 0:
        raise ValueError("Blend input must have number of parts divisible by 3 (material,grade,vol_frac triples)")
    
    materials = []
    for i in range(0, len(parts), 3):
        material = parts[i]
        grade = parts[i+1]
        try:
            vol_frac = float(parts[i+2])
        except ValueError:
            raise ValueError(f"Invalid volume fraction: {parts[i+2]}")
        
        materials.append({'material': material, 'grade': grade, 'vol_frac': vol_frac})
    
    # Validate volume fractions sum to 1.0
    total_vol_frac = sum(mat['vol_frac'] for mat in materials)
    if abs(total_vol_frac - 1.0) > 0.01:
        raise ValueError(f"Volume fractions must sum to 1.0 (currently: {total_vol_frac:.2f})")
    
    return materials

def find_material_by_grade(sus_df, material, grade):
    """Find material in sustainability.csv by material name and grade"""
    for _, row in sus_df.iterrows():
        material_value = row.get('Polymer Category', '')
        grade_value = row.get('commercialName', '')  # Changed from 'Grade' to 'commercialName'
        
        if pd.isna(material_value) or pd.isna(grade_value):
            continue
            
        # Check if both material and grade match (case insensitive)
        if (material.lower() in str(material_value).lower() and 
            grade.lower() in str(grade_value).lower()):
            return row
    
    # Fallback: try to find by grade only if material not found
    for _, row in sus_df.iterrows():
        grade_value = row.get('commercialName', '')  # Changed from 'Grade' to 'commercialName'
        if pd.isna(grade_value):
            continue
        if grade.lower() in str(grade_value).lower():
            return row
    
    return None

# Hybrid classification functions - TUV Home certification takes priority, with fallback to polymer type
def is_home_compostable_certified(tuv_home):
    """Check if material is certified for home composting based on TUV Home column"""
    if pd.isna(tuv_home) or tuv_home == '':
        return False
    
    tuv_str = str(tuv_home).lower()
    # Check for certification indicators
    certified_indicators = ['certified', 'ok compost home', 'home', '20250403_okcompost_export home.pdf']
    not_certified_indicators = ['not certified', 'not certified', 'n/a']
    
    # If any "not certified" indicators are present, return False
    if any(indicator in tuv_str for indicator in not_certified_indicators):
        return False
    
    # If any certification indicators are present, return True
    if any(indicator in tuv_str for indicator in certified_indicators):
        return True
    
    return False

def is_low_disintegration_polymer(polymer):
    """Petroleum-based polymers and PLA that can't exceed 2% disintegration"""
    low_disintegration = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'BIO-PE', 'PLA']
    return any(x in polymer.upper() for x in low_disintegration)

def is_medium_disintegration_polymer(polymer):
    """All non-home-compostable biopolymers that aren't petroleum-based or PLA"""
    # This includes all biopolymers that aren't in the low disintegration category
    # Examples: PBAT, PHB, PBS, PCL, PHA, Starch-based, etc.
    low_disintegration = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'Bio-PE', 'PLA']
    return not any(x in polymer.upper() for x in low_disintegration)

def should_get_synergistic_boost(polymer):
    """Determine if a polymer should get synergistic boost when blended with home-compostable materials"""
    # Petroleum-based polymers that should NOT get boost
    petroleum_polymers = ['LDPE', 'PP', 'EVOH', 'PA', 'PET', 'PVDC', 'BIO-PE']
    
    # Check if it's a petroleum-based polymer
    is_petroleum = any(x in polymer.upper() for x in petroleum_polymers)
    
    # All biopolymers (including PLA) should get synergistic boost
    # Only petroleum-based polymers are excluded
    return not is_petroleum

def get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed=None):
    """Determine maximum disintegration percentage: TUV Home certification takes priority, then polymer type"""
    # Set seed for this specific material if provided
    if material_seed is not None:
        np.random.seed(material_seed)
    
    # First check TUV Home certification
    is_home_certified = is_home_compostable_certified(tuv_home)
    
    if is_home_certified:
        # Home-compostable certified materials: 90-95% max disintegration
        result = np.random.uniform(90, 95)
    else:
        # Not home-compostable certified - use polymer type classification
        if is_low_disintegration_polymer(polymer):
            result = np.random.uniform(0.5, 2)
        elif is_medium_disintegration_polymer(polymer):
            result = np.random.uniform(30, 80)
        else:
            result = np.random.uniform(10, 30)
    
    # Reset to global seed to ensure reproducibility
    np.random.seed(GLOBAL_SEED)
    return result

def generate_material_curve(polymer, grade, tuv_home, thickness_val, days=DAYS, material_seed=None, actual_thickness=None):
    """Generate disintegration curve for a single material based ONLY on TUV Home certification"""
    t = np.arange(0, days)  # Start at 0 instead of 1
    
    # Use ONLY the TUV Home certification data - NO HARDWIRED LOGIC
    is_home_compostable = is_home_compostable_certified(tuv_home)
    
    # Calculate scaling factors
    cert_scaling = calculate_certification_thickness_scaling(thickness_val)
    actual_scaling = calculate_actual_thickness_scaling(actual_thickness)
    max_modulation = calculate_max_disintegration_modulation(actual_thickness)
    
    if is_home_compostable:
        # Home-compostable certified: sigmoid curve with high disintegration
        base_k = 0.08
        base_t0 = 70
        
        # Apply both certification and actual thickness scaling to kinetics
        k = base_k * cert_scaling / actual_scaling
        t0 = base_t0 / cert_scaling * actual_scaling
        
        # Use hybrid maximum disintegration (certification takes priority) and apply thickness modulation
        base_max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        max_disintegration = base_max_disintegration * max_modulation
        max_disintegration = min(max_disintegration, 95)  # Cap at 95%
        y = sigmoid(t, max_disintegration, k, t0)
        
        actual_thickness_display = actual_thickness if actual_thickness is not None else ACTUAL_THICKNESS_DEFAULT
    else:
        # Not home-compostable certified: use hybrid classification
        cert_scaling = calculate_certification_thickness_scaling(thickness_val)
        actual_scaling = calculate_actual_thickness_scaling(actual_thickness)
        max_modulation = calculate_max_disintegration_modulation(actual_thickness)
        base_max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        max_disintegration = base_max_disintegration * max_modulation
        max_disintegration = min(max_disintegration, 95)  # Cap at 95%
        if max_disintegration < 5:  # Very low disintegration materials (petroleum-based)
            # Linear progression from 0 to max_disintegration with some randomness
            y = np.linspace(0, max_disintegration, len(t))
            # Add small random variations to make it more realistic
            if material_seed is not None:
                np.random.seed(material_seed + 500)  # Different seed for linear noise
            noise = np.random.normal(0, 0.05, size=y.shape)  # Small noise
            y = y + noise
            # Ensure it stays monotonically increasing and within bounds
            y = np.clip(y, 0, max_disintegration)
            for i in range(1, len(y)):
                if y[i] < y[i-1]:
                    y[i] = y[i-1] + np.random.uniform(0, 0.01)
        else:
            base_k = 0.02
            base_t0 = 120
            k = base_k * cert_scaling / actual_scaling
            t0 = base_t0 / cert_scaling * actual_scaling
            y = sigmoid(t, max_disintegration, k, t0)
            # Shift curve to start at 0 by subtracting the initial value
            y0 = sigmoid(0, max_disintegration, k, t0)
            y = y - y0
        actual_thickness_display = actual_thickness if actual_thickness is not None else ACTUAL_THICKNESS_DEFAULT
    
    # Set seed for noise generation
    if material_seed is not None:
        np.random.seed(material_seed + 1000)  # Offset for noise
    
    # Add very small noise for minimal realism (much reduced noise)
    y = y + np.random.normal(0, 0.1, size=y.shape)
    
    # Ensure curve is monotonically increasing (no decreases)
    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            y[i] = y[i-1] + np.random.uniform(0, 0.02)  # Very small positive increment
    
    # No clipping - let sigmoid handle everything naturally
    y = np.clip(y, 0, None)
    
    # Final check for NaN values
    if np.any(np.isnan(y)):
        print(f"    ERROR: NaN values in curve for {polymer} {grade}. Using flat curve at 2%.")
        y = np.full_like(t, 2.0)
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return y

def generate_material_curve_with_synergistic_boost(polymer, grade, tuv_home, thickness_val, home_fraction_in_blend, days=DAYS, material_seed=None, actual_thickness=None):
    """Generate disintegration curve for a single material with synergistic boost from home-compostable materials"""
    t = np.arange(0, days)  # Start at 0 instead of 1
    
    # Use ONLY the TUV Home certification data - NO HARDWIRED LOGIC
    is_home_compostable = is_home_compostable_certified(tuv_home)
    
    # Calculate scaling factors
    cert_scaling = calculate_certification_thickness_scaling(thickness_val)
    actual_scaling = calculate_actual_thickness_scaling(actual_thickness)
    max_modulation = calculate_max_disintegration_modulation(actual_thickness)
    
    # Check if this material should get a synergistic boost
    should_get_boost = (
        not is_home_compostable and  # Not home-compostable certified
        home_fraction_in_blend > 0 and  # There are home-compostable materials in blend
        should_get_synergistic_boost(polymer)  # All biopolymers (including PLA) get boost, petroleum-based don't
    )
    
    if is_home_compostable:
        base_k = 0.08
        base_t0 = 70
        k = base_k * cert_scaling / actual_scaling
        t0 = base_t0 / cert_scaling * actual_scaling
        base_max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        max_disintegration = base_max_disintegration * max_modulation
        max_disintegration = min(max_disintegration, 95)  # Cap at 95%
        y = sigmoid(t, max_disintegration, k, t0)
        actual_thickness_display = actual_thickness if actual_thickness is not None else ACTUAL_THICKNESS_DEFAULT
    else:
        base_max_disintegration = get_max_disintegration_hybrid(polymer, tuv_home, thickness_val, material_seed)
        max_disintegration = base_max_disintegration * max_modulation
        if should_get_boost:
            base_k = 0.08
            base_t0 = 70
            k = base_k * cert_scaling / actual_scaling
            t0 = base_t0 / cert_scaling * actual_scaling
            # Proportional max disintegration boost based on home-compostable fraction
            if 'PLA' in polymer.upper():
                max_boost_percent = 100
            else:
                max_boost_percent = 15
            actual_boost = max_boost_percent * home_fraction_in_blend
            max_disintegration = min(max_disintegration + actual_boost, 95)
            y = sigmoid(t, max_disintegration, k, t0)
        else:
            if max_disintegration < 5:
                y = np.linspace(0, max_disintegration, len(t))
                if material_seed is not None:
                    np.random.seed(material_seed + 500)
                noise = np.random.normal(0, 0.05, size=y.shape)
                y = y + noise
                y = np.clip(y, 0, max_disintegration)
                for i in range(1, len(y)):
                    if y[i] < y[i-1]:
                        y[i] = y[i-1] + np.random.uniform(0, 0.01)
            else:
                base_k = 0.02
                base_t0 = 120
                k = base_k * cert_scaling / actual_scaling
                t0 = base_t0 / cert_scaling * actual_scaling
                y = sigmoid(t, max_disintegration, k, t0)
                y0 = sigmoid(0, max_disintegration, k, t0)
                y = y - y0
        actual_thickness_display = actual_thickness if actual_thickness is not None else ACTUAL_THICKNESS_DEFAULT
    
    # Set seed for noise generation
    if material_seed is not None:
        np.random.seed(material_seed + 2000)  # Offset for synergistic noise
    
    # Add very small noise for minimal realism (much reduced noise)
    y = y + np.random.normal(0, 0.1, size=y.shape)
    
    # Ensure curve is monotonically increasing (no decreases)
    for i in range(1, len(y)):
        if y[i] < y[i-1]:
            y[i] = y[i-1] + np.random.uniform(0, 0.02)  # Very small positive increment
    
    # No clipping - let sigmoid handle everything naturally
    y = np.clip(y, 0, None)
    
    # Final check for NaN values
    if np.any(np.isnan(y)):
        print(f"    ERROR: NaN values in curve for {polymer} {grade}. Using flat curve at 2%.")
        y = np.full_like(t, 2.0)
    
    # Reset to global seed
    np.random.seed(GLOBAL_SEED)
    return y

def parse_thickness(thickness_cert):
    """Parse thickness from sustainability.csv format"""
    thickness_val = THICKNESS_DEFAULT
    if isinstance(thickness_cert, str):
        if 'um' in thickness_cert:
            try:
                thickness_val = float(re.sub(r'[^\d.]', '', thickness_cert)) / 1000.0
            except:
                thickness_val = THICKNESS_DEFAULT
        elif 'mm' in thickness_cert:
            try:
                thickness_val = float(re.sub(r'[^\d.]', '', thickness_cert))
            except:
                thickness_val = THICKNESS_DEFAULT
        else:
            try:
                thickness_val = float(thickness_cert)
            except:
                thickness_val = THICKNESS_DEFAULT
    else:
        try:
            thickness_val = float(thickness_cert)
        except:
            thickness_val = THICKNESS_DEFAULT
    return thickness_val 