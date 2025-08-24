#!/usr/bin/env python3
"""
Utilities module for Differentiable Label Optimization.
Contains utility functions for sigmoid calculations and other helper functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Thickness scaling constants
ACTUAL_THICKNESS_DEFAULT = 0.050  # Default actual thickness (50 μm = 0.050 mm)

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


def get_categorical_encoding_mapping():
    """
    Define the consistent mapping for categorical features.
    This ensures the same order of one-hot encoded columns in both training and prediction.
    
    Returns:
        dict: Mapping of categorical column names to their unique values in order
    """
    return {
        'Polymer Grade 1': [
            'Ingeo 4032D', 'Ingeo 4043D', 'Luminy® L175', 'BIOCYCLE® 1000', 'ENMAT Y3000',
            'ENMAT™ Y1000P', 'Green Planet™ PHBH', 'PHACT A1000P', 'PHACT S1000P',
            'Kuredux® PGA', 'Polylactide', 'ecoflex® F Blend C1200', 'Ecoworld',
            'Ecovance rf-PBAT', 'BioPBS™ FZ91', 'PBS TH803S', 'BioPBS™ FD92',
            "I'm green™ STN7006", 'LDPE LD 150', 'Capa™ 6500', 'Capa™ 6800',
            'Total PPH 3270', 'Mylar 48-F-OC Clear PET PET', 'CX-C6',
            'Aegis PCR-H135ZP Nylon 6', 'EVAL F171B', 'Ecovio 40 FG2331', 'EcoVio F2223',
            'EcoVio F2224', 'EcoVio F2330', 'EcoVio F2331', 'EcoVio F2332',
            'EcoVio F2341x', 'EcoVio F23B1', 'Terratek FX1515',
            'Terratek FX1515 - Blown Film', 'Terratek FX2217', 'Bio-Flex F1804',
            'Bio-Flex FX1137', 'Bio-Flex F1140', 'Bio-Flex FX1130', 'Bio-Flex FX1135',
            'Bio-Flex F1801', 'Bio-Flex F2110', 'Bio-Flex F2100', 'Bio-Flex FX1821',
            'BF2001', 'BF700Z-X', 'BF7002', 'Bioplast GF 106', 'Bioplast 500 A',
            'M Vera B5019', 'M Vera B5033', 'EcoWill', 'PHACT CA8570P', 'PHACT CA1240PF',
            'PHACT CA1270P', 'EcoVio PS1606', 'EcoVio 70 PS14H6', 'EcoVio T2308',
            'Nativia NTSS', 'BioBlend', 'BioPBS™ FZ78TM', 'BioPBS™ FZ79AC',
            'BioPBS™ FZ83AC', 'BioPBS™ FZ85AC'
        ],
        'enzyme kinetics': ['slow', 'fastest', 'fast', 'med', 'none']
    }


def one_hot_encode_categorical_features(df, categorical_mapping=None):
    """
    One-hot encode categorical features consistently.
    
    Args:
        df: DataFrame containing the data
        categorical_mapping: Optional mapping of categorical columns to their unique values
        
    Returns:
        DataFrame: DataFrame with categorical columns replaced by one-hot encoded columns
    """
    if categorical_mapping is None:
        categorical_mapping = get_categorical_encoding_mapping()
    
    df_encoded = df.copy()
    
    for col_name, unique_values in categorical_mapping.items():
        if col_name in df_encoded.columns:
            # Create one-hot encoded columns for each unique value
            for value in unique_values:
                col_name_encoded = f"{col_name}_{value.replace(' ', '_').replace('®', '').replace('™', '').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')}"
                df_encoded[col_name_encoded] = (df_encoded[col_name] == value).astype(int)
            
            # Drop the original categorical column
            df_encoded = df_encoded.drop(col_name, axis=1)
        else:
            # If the categorical column is missing, create all one-hot columns with zeros
            print(f"Warning: Categorical column '{col_name}' not found, creating zero-filled one-hot columns")
            for value in unique_values:
                col_name_encoded = f"{col_name}_{value.replace(' ', '_').replace('®', '').replace('™', '').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')}"
                df_encoded[col_name_encoded] = 0
    
    return df_encoded


def get_categorical_feature_names():
    """
    Get the names of all categorical features after one-hot encoding.
    
    Returns:
        list: List of categorical feature column names in the order they will appear
    """
    categorical_mapping = get_categorical_encoding_mapping()
    feature_names = []
    
    for col_name, unique_values in categorical_mapping.items():
        for value in unique_values:
            col_name_encoded = f"{col_name}_{value.replace(' ', '_').replace('®', '').replace('™', '').replace('(', '').replace(')', '').replace('/', '_').replace('-', '_')}"
            feature_names.append(col_name_encoded)
    
    return feature_names


def calculate_k0_from_sigmoid_params(max_L: float, t0: float, y0: float = 0.0, t_max: float = 200.0, 
                                   majority_polymer_high_disintegration: bool = None, 
                                   actual_thickness: float = None) -> float:
    """
    Calculate k0 (rate constant) from sigmoid parameters with optional thickness scaling.
    
    Args:
        max_L: Maximum disintegration level (predicted_property1)
        t0: Time at 50% disintegration (predicted_property2)
        y0: Initial disintegration level (default 0)
        t_max: Time at which max_L should be reached (default 200 days)
        majority_polymer_high_disintegration: If True, majority polymer has max_L > 5 (high disintegration)
                                            If False, majority polymer has max_L < 5 (low disintegration)
                                            If None, use original logic (take maximum)
        actual_thickness: Actual thickness of material (mm), defaults to 50μm
        
    Returns:
        k0: Rate constant for the sigmoid curve (scaled by thickness if provided)
    """
    # SIGMOID FUNCTION: y = max_L / (1 + e^(-k0 * (t - t0)))
    
    # We need to solve for k0 that satisfies both boundary conditions:
    # 1. At t=0: y ≈ 0 (very small value)
    # 2. At t=t_max: y ≈ max_L (very close to max_L)
    
    # For practical purposes, let's say:
    # At t=0: y = 0.001 * max_L (0.1% of max_L)
    # At t=t_max: y = 0.999 * max_L (99.9% of max_L)
    
    # From condition 1: 0.001 * max_L = max_L / (1 + e^(k0 * t0))
    # 0.001 = 1 / (1 + e^(k0 * t0))
    # 1 + e^(k0 * t0) = 1/0.001 = 1000
    # e^(k0 * t0) = 999
    # k0 * t0 = ln(999)
    # k0 = ln(999) / t0
    
    # From condition 2: 0.999 * max_L = max_L / (1 + e^(-k0 * (t_max - t0)))
    # 0.999 = 1 / (1 + e^(-k0 * (t_max - t0)))
    # 1 + e^(-k0 * (t_max - t0)) = 1/0.999
    # e^(-k0 * (t_max - t0)) = 1/0.999 - 1
    # -k0 * (t_max - t0) = ln(1/0.999 - 1)
    # k0 = -ln(1/0.999 - 1) / (t_max - t0)
    
    if t0 <= 0 or t_max <= t0:
        return 0.1  # Default value if parameters are invalid
    
    try:
        # Calculate k0 from both boundary conditions
        k0_from_start = np.log(999) / t0
        k0_from_end = -np.log(1/0.999 - 1) / (t_max - t0)
        
        # Choose k0 based on majority polymer behavior
        if majority_polymer_high_disintegration is None:
            # Original logic: take the maximum to satisfy both conditions
            k0 = max(k0_from_start, k0_from_end)
        elif majority_polymer_high_disintegration:
            # Majority polymer has max_L > 5 (high disintegration)
            # Use the maximum k0 for faster, more aggressive disintegration
            k0 = max(k0_from_start, k0_from_end)
        else:
            # Majority polymer has max_L < 5 (low disintegration)
            # Use the minimum k0 for slower, more conservative disintegration
            k0 = min(k0_from_start, k0_from_end)
        
        # Apply thickness scaling if thickness is provided
        if actual_thickness is not None:
            thickness_scaling = calculate_actual_thickness_scaling(actual_thickness)
            k0 = k0 / thickness_scaling  # Thicker materials = slower kinetics
        
        # Ensure k0 is positive and reasonable
        return max(0.01, min(5.0, k0))
    except (ValueError, ZeroDivisionError):
        return 0.1  # Default value if calculation fails


def generate_sigmoid_curves(max_L_values: np.ndarray, t0_values: np.ndarray, 
                           k0_values: np.ndarray, days: int = 200, 
                           save_csv: bool = True, save_plot: bool = True,
                           curve_type: str = 'disintegration', save_dir: str = '.',
                           actual_thickness: float = None) -> pd.DataFrame:
    """
    Generate sigmoid curves for all samples and save results with optional thickness scaling.
    
    Args:
        max_L_values: Array of max_L values for each sample
        t0_values: Array of t0 values for each sample
        k0_values: Array of k0 values for each sample
        days: Number of days to simulate (default 200)
        save_csv: Whether to save CSV with daily values
        save_plot: Whether to save PNG plot
        curve_type: Type of curve ('disintegration' or 'biodegradation')
        save_dir: Directory to save results
        actual_thickness: Actual thickness of material (mm), defaults to 50μm
        
    Returns:
        DataFrame with daily values for all samples
    """
    # Generate time points (1 day intervals)
    time_points = np.arange(0, days + 1, 1)
    
    # Calculate sigmoid curves for all samples
    sigmoid_data = []
    
    for sample_idx, (max_L, t0, k0) in enumerate(zip(max_L_values, t0_values, k0_values)):
        # Apply thickness scaling to max_L if thickness is provided
        if actual_thickness is not None:
            max_modulation = calculate_max_disintegration_modulation(actual_thickness)
            max_L = max_L * max_modulation
        
        # Cap max_L at 95 for physical realism (no values > 100 allowed)
        max_L_capped = min(max_L, 95.0)
        
        # PROPER SIGMOID FUNCTION: y = max_L / (1 + e^(-k0 * (t - t0)))
        # This gives true S-shape with inflection point at t0
        # k0 is calculated to naturally satisfy boundary conditions
        
        values = max_L_capped / (1 + np.exp(-k0 * (time_points - t0)))
        
        for day, y in zip(time_points, values):
            sigmoid_data.append({
                'sample_index': sample_idx,
                'day': day,
                curve_type: y,
                'max_L': max_L_capped,  # Use capped value in output
                't0': t0,
                'k0': k0
            })
    
    sigmoid_df = pd.DataFrame(sigmoid_data)
    
    # Save CSV
    if save_csv:
        csv_filename = os.path.join(save_dir, f'sigmoid_{curve_type}_curves.csv')
        sigmoid_df.to_csv(csv_filename, index=False)

    
    # Create and save plot
    if save_plot:
        import matplotlib as mpl
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000000')
        ax.set_facecolor('#000000')
        
        # Use purple color for the curve
        color = '#8942E5'  # Same purple as compostability.py
        
        # Plot curves for each sample
        for sample_idx in range(len(max_L_values)):
            sample_data = sigmoid_df[sigmoid_df['sample_index'] == sample_idx]
            ax.plot(sample_data['day'], sample_data[curve_type], 
                   linewidth=2, color=color)
        
        # Set axis and title colors to white
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        # Set title
        ax.set_title(f'Rate of {curve_type.capitalize()}', color='white', fontsize=18, weight='bold')
        
        # Remove grid for clean look
        ax.grid(False)
        
        # Set consistent axis ranges
        ax.set_xlim(0, days)  # X-axis from 0 to specified days
        ax.set_ylim(0, 105)   # Y-axis from 0 to 105% (same as compostability.py)
        
        # Set labels
        ax.set_xlabel('Time (day)', color='white')
        ax.set_ylabel(f'{curve_type.capitalize()} (%)', color='white')
        
        # Remove top and right spines for open graph look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = os.path.join(save_dir, f'sigmoid_{curve_type}_curves.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

    
    return sigmoid_df 


def generate_cubic_biodegradation_curve(disintegration_df, t0, max_disintegration, days=400, 
                                       save_csv=True, save_plot=True, save_dir='.',
                                       actual_thickness=None):
    """
    Generate cubic biodegradation curve based on disintegration curve points.
    
    Args:
        disintegration_df: DataFrame with disintegration curve data
        t0: Time to 50% disintegration
        max_disintegration: Maximum disintegration level
        days: Number of days to simulate (default 400)
        save_csv: Whether to save CSV with daily values
        save_plot: Whether to save PNG plot
        save_dir: Directory to save results
        actual_thickness: Actual thickness of material (mm)
        
    Returns:
        DataFrame with daily biodegradation values
    """
    # Get disintegration value at t0 (find closest day value since t0 might be float)
    day_values = disintegration_df['day'].values
    closest_day_idx = np.argmin(np.abs(day_values - t0))
    closest_day = day_values[closest_day_idx]
    dis_at_t0 = disintegration_df.iloc[closest_day_idx]['disintegration']

    # Introduce slight randomness to the final max: subtract a random value in [0, 5]
    # so the cubic endpoint is up to 5 units below the disintegration max
    import random as _rnd
    random_delta = _rnd.uniform(0.0, 5.0)
    randomized_max = max(0.0, float(max_disintegration) - random_delta)
    
    # Define the 3 points for cubic polynomial
    # Point 1: (0, 0) - starts at 0%
    # Point 2: (t0+30, dis_at_t0) - same y-value but shifted to t0+30
    # Point 3: (400, max_disintegration) - reaches maximum at day 400
    x_points = [0, t0+30, 400]
    y_points = [0, dis_at_t0, randomized_max]
    
    # Solve system of linear equations for cubic coefficients
    # y = ax^3 + bx^2 + cx + d
    # We have 3 points, so we can solve for 3 coefficients (d=0 since y=0 at x=0)
    
    # Create coefficient matrix A and vector b
    A = np.array([
        [(t0+30)**3, (t0+30)**2, t0+30],
        [400**3, 400**2, 400]
    ])
    
    b = np.array([dis_at_t0, randomized_max])
    
    try:
        # Solve for coefficients using least squares
        coefficients = np.linalg.lstsq(A, b, rcond=None)[0]
        a, b_coeff, c = coefficients
        
        # Generate time points (1 day intervals)
        time_points = np.arange(0, days + 1, 1)
        
        # Calculate cubic curve: y = ax^3 + bx^2 + cx
        biodegradation_values = a * time_points**3 + b_coeff * time_points**2 + c * time_points
        
        # Smoothly saturate near the maximum to avoid a sharp corner where a hard cap would occur.
        # Uses a softplus-based cap that is C1-continuous at the shoulder.
        # beta controls how sharp the saturation is (smaller = smoother plateau).
        beta = 0.1
        delta_to_cap = randomized_max - biodegradation_values
        biodegradation_values = randomized_max - (1.0 / beta) * np.log1p(np.exp(beta * delta_to_cap))

        # Ensure values are not negative
        biodegradation_values = np.maximum(0, biodegradation_values)

        # Enforce monotonic non-decreasing behavior to remove any tiny end dips
        biodegradation_values = np.maximum.accumulate(biodegradation_values)

        # Guarantee exact max at the final day without introducing a sharp kink:
        # scale up gently if we ended slightly below the cap, then clamp and re-monotonize
        final_val = biodegradation_values[-1]
        if final_val > 0 and final_val < randomized_max:
            scale = randomized_max / final_val
            biodegradation_values = biodegradation_values * scale
            biodegradation_values = np.minimum(biodegradation_values, randomized_max)
            biodegradation_values = np.maximum.accumulate(biodegradation_values)
        # Ensure the last point is exactly the randomized max
        biodegradation_values[-1] = randomized_max
        
        # Create DataFrame
        cubic_data = []
        for day, y in zip(time_points, biodegradation_values):
            cubic_data.append({
                'sample_index': 0,
                'day': day,
                'biodegradation': y,
                'max_L': randomized_max,
                't0': t0,
                'curve_type': 'cubic'
            })
        
        cubic_df = pd.DataFrame(cubic_data)
        
        # Save CSV
        if save_csv:
            csv_filename = os.path.join(save_dir, 'cubic_biodegradation_curves.csv')
            cubic_df.to_csv(csv_filename, index=False)
        
        # Create and save plot
        if save_plot:
            import matplotlib.pyplot as plt
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='#000000')
            ax.set_facecolor('#000000')
            
            # Plot disintegration curve
            dis_sample_data = disintegration_df[disintegration_df['sample_index'] == 0]
            ax.plot(dis_sample_data['day'], dis_sample_data['disintegration'], 
                   linewidth=3, color='#8942E5', label='Disintegration (Sigmoid)', alpha=0.8)
            
            # Plot cubic biodegradation curve
            ax.plot(cubic_df['day'], cubic_df['biodegradation'], 
                   linewidth=3, color='#FF6B6B', label='Biodegradation (Cubic)', alpha=0.8)
            
            # Mark key points
            ax.scatter(x_points, y_points, color='#FFD93D', s=100, zorder=5, 
                      label='Cubic Control Points')
            
            # Set axis and title colors to white
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Set title
            ax.set_title('Disintegration vs Biodegradation Curves', color='white', fontsize=18, weight='bold')
            
            # Remove grid for clean look
            ax.grid(False)
            
            # Set consistent axis ranges
            ax.set_xlim(0, days)
            ax.set_ylim(0, 105)
            
            # Set labels
            ax.set_xlabel('Time (day)', color='white')
            ax.set_ylabel('Percentage (%)', color='white')
            
            # Add legend
            ax.legend(facecolor='#000000', edgecolor='white', fontsize=12)
            
            # Remove top and right spines for open graph look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
                    # Save comparison plot
        comparison_filename = os.path.join(save_dir, 'quintic_vs_sigmoid_comparison.png')
        plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', facecolor='#000000')
        plt.close()

        
        # Also save individual quintic biodegradation plot
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000000')
        ax.set_facecolor('#000000')
        
        # Plot only the quintic biodegradation curve
        ax.plot(quintic_df['day'], quintic_df['biodegradation'], 
               linewidth=3, color='#FF6B6B', label='Biodegradation (Quintic)', alpha=0.8)
        
        # Set axis and title colors to white
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        # Set title
        ax.set_title('Quintic Biodegradation Curve', color='white', fontsize=18, weight='bold')
        
        # Remove grid for clean look
        ax.grid(False)
        
        # Set consistent axis ranges
        ax.set_xlim(0, days)
        ax.set_ylim(0, 105)
        
        # Set labels
        ax.set_xlabel('Time (day)', color='white')
        ax.set_ylabel('Biodegradation (%)', color='white')
        
        # Remove top and right spines for open graph look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        
        # Save individual plot
        individual_filename = os.path.join(save_dir, 'quintic_biodegradation_curves.png')
        plt.savefig(individual_filename, dpi=300, bbox_inches='tight', facecolor='#000000')
        plt.close()

        
        return quintic_df
        
    except np.linalg.LinAlgError as e:

        # Fallback to sigmoid if quintic fails
        return generate_sigmoid_curves(
            np.array([max_disintegration]), 
            np.array([t0 * 2.0]), 
            np.array([0.1]), 
            days=400, 
            curve_type='biodegradation',
            save_dir=save_dir,
            actual_thickness=actual_thickness
        ) 


def generate_quintic_biodegradation_curve(disintegration_df, t0, max_disintegration, days=400, 
                                         save_csv=True, save_plot=True, save_dir='.',
                                         actual_thickness=None):
    """
    Generate quintic biodegradation curve based on disintegration curve points.
    
    The curve automatically determines when biodegradation reaches maximum based on
    when disintegration is 99% complete, creating a biologically meaningful coupling.
    
    IMPORTANT: Thickness scaling is inherited from the disintegration curve.
    The disintegration_df should contain thickness-adjusted values, and the
    biodegradation curve will automatically follow the same thickness effects.
    
    Args:
        disintegration_df: DataFrame with disintegration curve data (should be thickness-adjusted)
        t0: Time to 50% disintegration
        max_disintegration: Maximum disintegration level
        days: Number of days to simulate (default 400)
        save_csv: Whether to save CSV with daily values
        save_plot: Whether to save PNG plot
        save_dir: Directory to save results
        actual_thickness: Actual thickness of material (mm) - used for logging only
        
    Returns:
        DataFrame with daily biodegradation values (inherits thickness scaling from disintegration)
    """
    # Get disintegration value at t0 (find closest day value since t0 might be float)
    day_values = disintegration_df['day'].values
    closest_day_idx = np.argmin(np.abs(day_values - t0))
    closest_day = day_values[closest_day_idx]
    dis_at_t0 = disintegration_df.iloc[closest_day_idx]['disintegration']

    # IMPORTANT: The biodegradation curve inherits thickness effects from the disintegration curve
    # The disintegration_df already contains thickness-adjusted values, so we use those directly
    # This ensures the biodegradation curve follows the same thickness scaling as disintegration
    
    # The biodegradation maximum should be based on the ACTUAL disintegration maximum from the curve
    # not the original max_disintegration parameter, because thickness scaling affects the curve
    actual_dis_max = disintegration_df['disintegration'].max()
    
    # Introduce slight randomness to the final max: subtract a random value in [0, 2.5]
    # so the quintic endpoint is up to 2.5 units below the ACTUAL disintegration max
    import random as _rnd
    random_delta = _rnd.uniform(0.0, 2.5)
    randomized_max = max(0.0, actual_dis_max - random_delta)
    
    # Find when disintegration reaches 99% of its maximum (within 1%)
    # Use the actual disintegration maximum from the thickness-adjusted curve
    dis_99_percent = actual_dis_max * 0.99
    dis_99_idx = np.where(disintegration_df['disintegration'] >= dis_99_percent)[0]
    
    if len(dis_99_idx) > 0:
        t0_dis_99 = disintegration_df.iloc[dis_99_idx[0]]['day']
    else:
        # Fallback: use t0+50 if we can't find 99% point
        t0_dis_99 = t0 + 50
    
    # Define the 5 points for quintic polynomial
    # Point 1: (0, 0) - starts at 0%
    # Point 2: (t0, dis_at_t0-7) - at t0, but 7 units BELOW disintegration value
    # Point 3: (t0+10, dis_at_t0) - at t0+10, reaches same value as disintegration at t0
    # Point 4: (t0_dis_99 + 20, randomized_max) - reaches maximum 20 days after disintegration is 99% complete
    # Point 5: (400, randomized_max) - maintains maximum until day 400
    x_points = [0, t0, t0+10, t0_dis_99 + 20, 400]
    y_points = [0, dis_at_t0-7, dis_at_t0, randomized_max, randomized_max]
    

    
    # Solve system of linear equations for quintic coefficients
    # y = ax^5 + bx^4 + cx^3 + dx^2 + ex
    # We have 5 points, so we can solve for 5 coefficients exactly
    
    # Create coefficient matrix A and vector b
    A = np.array([
        [t0**5, t0**4, t0**3, t0**2, t0],
        [(t0+10)**5, (t0+10)**4, (t0+10)**3, (t0+10)**2, t0+10],
        [(t0_dis_99 + 20)**5, (t0_dis_99 + 20)**4, (t0_dis_99 + 20)**3, (t0_dis_99 + 20)**2, t0_dis_99 + 20],
        [400**5, 400**4, 400**3, 400**2, 400]
    ])
    
    b = np.array([dis_at_t0-7, dis_at_t0, randomized_max, randomized_max])
    
    try:
        # Solve for coefficients using least squares (4 equations, 5 unknowns)
        # We need to add a constraint to make the system solvable
        # Let's add a smoothness constraint: minimize curvature at the start
        
        # Add smoothness constraint: minimize second derivative at x=0
        # This gives us a 5x5 system
        A_augmented = np.vstack([
            A,
            [20*0**3, 12*0**2, 6*0, 2, 0]  # Second derivative at x=0 should be small
        ])
        b_augmented = np.append(b, 0)  # Target second derivative = 0 at start
        
        # Solve for coefficients using least squares
        coefficients = np.linalg.lstsq(A_augmented, b_augmented, rcond=None)[0]
        a, b_coeff, c, d, e = coefficients
        

        
        # Generate time points (1 day intervals)
        time_points = np.arange(0, days + 1, 1)
        
        # Calculate quintic curve: y = ax^5 + bx^4 + cx^3 + dx^2 + ex
        biodegradation_values = a * time_points**5 + b_coeff * time_points**4 + c * time_points**3 + d * time_points**2 + e * time_points
        
        # Smoothly saturate near the maximum to avoid a sharp corner where a hard cap would occur.
        # Uses a softplus-based cap that is C1-continuous at the shoulder.
        # beta controls how sharp the saturation is (smaller = smoother plateau).
        beta = 0.1
        delta_to_cap = randomized_max - biodegradation_values
        biodegradation_values = randomized_max - (1.0 / beta) * np.log1p(np.exp(beta * delta_to_cap))

        # Ensure values are not negative
        biodegradation_values = np.maximum(0, biodegradation_values)

        # Enforce monotonic non-decreasing behavior to remove any tiny end dips
        biodegradation_values = np.maximum.accumulate(biodegradation_values)

        # Guarantee exact max at the final day without introducing a sharp kink:
        # scale up gently if we ended slightly below the cap, then clamp and re-monotonize
        final_val = biodegradation_values[-1]
        if final_val > 0 and final_val < randomized_max:
            scale = randomized_max / final_val
            biodegradation_values = biodegradation_values * scale
            biodegradation_values = np.minimum(biodegradation_values, randomized_max)
            biodegradation_values = np.maximum.accumulate(biodegradation_values)
        # Ensure the last point is exactly the randomized max
        biodegradation_values[-1] = randomized_max
        
        # Create DataFrame
        quintic_data = []
        for day, y in zip(time_points, biodegradation_values):
            quintic_data.append({
                'sample_index': 0,
                'day': day,
                'biodegradation': y,
                'max_L': randomized_max,
                't0': t0,
                'curve_type': 'quintic'
            })
        
        quintic_df = pd.DataFrame(quintic_data)
        
        # Save CSV
        if save_csv:
            csv_filename = os.path.join(save_dir, 'quintic_biodegradation_curves.csv')
            quintic_df.to_csv(csv_filename, index=False)
    
        
        # Create and save plot
        if save_plot:
            import matplotlib.pyplot as plt
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 8), facecolor='#000000')
            ax.set_facecolor('#000000')
            
            # Plot disintegration curve (already thickness-adjusted)
            dis_sample_data = disintegration_df[disintegration_df['sample_index'] == 0]
            ax.plot(dis_sample_data['day'], dis_sample_data['disintegration'], 
                   linewidth=3, color='#8942E5', label='Disintegration (Sigmoid, Thickness-Adjusted)', alpha=0.8)
            
            # Plot quintic biodegradation curve (inherits thickness scaling from disintegration)
            ax.plot(quintic_df['day'], quintic_df['biodegradation'], 
                   linewidth=3, color='#FF6B6B', label='Biodegradation (Quintic, Thickness-Inherited)', alpha=0.8)
            
            # Mark key points
            ax.scatter(x_points, y_points, color='#FFD93D', s=100, zorder=5, 
                      label='Quintic Control Points')
            
            # Set axis and title colors to white
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Set title
            ax.set_title('Disintegration vs Biodegradation Curves (Quintic)', color='white', fontsize=18, weight='bold')
            
            # Remove grid for clean look
            ax.grid(False)
            
            # Set consistent axis ranges
            ax.set_xlim(0, days)
            ax.set_ylim(0, 105)
            
            # Add legend
            ax.legend(facecolor='#000000', edgecolor='white', fontsize=12)
            
            # Remove top and right spines for open graph look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Save comparison plot
            comparison_filename = os.path.join(save_dir, 'quintic_vs_sigmoid_comparison.png')
            plt.savefig(comparison_filename, dpi=300, bbox_inches='tight', facecolor='#000000')
            plt.close()
    
            
            # Also save individual quintic biodegradation plot
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000000')
            ax.set_facecolor('#000000')
            
            # Plot only the quintic biodegradation curve
            ax.plot(quintic_df['day'], quintic_df['biodegradation'], 
                   linewidth=3, color='#FF6B6B', label='Biodegradation (Quintic)', alpha=0.8)
            
            # Set axis and title colors to white
            ax.tick_params(colors='white', which='both')
            for spine in ax.spines.values():
                spine.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Set title
            ax.set_title('Quintic Biodegradation Curve', color='white', fontsize=18, weight='bold')
            
            # Remove grid for clean look
            ax.grid(False)
            
            # Set consistent axis ranges
            ax.set_xlim(0, days)
            ax.set_ylim(0, 105)
            
            # Set labels
            ax.set_xlabel('Time (day)', color='white')
            ax.set_ylabel('Biodegradation (%)', color='white')
            
            # Remove top and right spines for open graph look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout()
            
            # Save individual plot
            individual_filename = os.path.join(save_dir, 'quintic_biodegradation_curves.png')
            plt.savefig(individual_filename, dpi=300, bbox_inches='tight', facecolor='#000000')
            plt.close()
    
            
            return quintic_df
        
    except np.linalg.LinAlgError as e:
        print(f"Warning: Quintic solution failed, falling back to sigmoid")
        print(f"Error: {e}")
        # Fallback to sigmoid if quintic fails
        return generate_sigmoid_curves(
            np.array([max_disintegration]), 
            np.array([t0 * 2.0]), 
            np.array([0.1]), 
            days=400, 
            curve_type='biodegradation',
            save_dir=save_dir,
            actual_thickness=actual_thickness
        ) 