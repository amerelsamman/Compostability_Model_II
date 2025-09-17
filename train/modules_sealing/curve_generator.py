"""
Sealing profile curve generation module for polymer blends.
Generates sealing strength vs temperature curves using cubic polynomial interpolation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple, Optional
from .utils import calculate_boundary_points, validate_sealing_curve

class SealingCurveGenerator:
    """
    Generator for sealing strength vs temperature curves using cubic polynomial interpolation.
    """
    
    def __init__(self, temperature_range: Tuple[float, float] = (0, 300), 
                 num_points: int = 100):
        """
        Initialize the sealing curve generator.
        
        Args:
            temperature_range: (min_temp, max_temp) in Celsius
            num_points: Number of points to generate in the curve
        """
        self.temperature_range = temperature_range
        self.num_points = num_points
        
    def generate_cubic_polynomial(self, boundary_points: Dict[str, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate cubic polynomial curve through 4 boundary points.
        
        Args:
            boundary_points: Dictionary with 4 boundary points (temp, strength)
        
        Returns:
            Tuple of (temperatures, strengths) arrays
        """
        # Extract points in order
        points = [
            boundary_points['initial_sealing'],
            boundary_points['first_polymer_max'], 
            boundary_points['blend_predicted'],
            boundary_points['degradation']
        ]
        
        # Separate temperatures and strengths
        temps = [p[0] for p in points]
        strengths = [p[1] for p in points]
        
        # Solve for cubic polynomial coefficients: y = ax³ + bx² + cx + d
        # We have 4 points, so we can solve for 4 coefficients exactly
        A = np.array([
            [temps[0]**3, temps[0]**2, temps[0], 1],
            [temps[1]**3, temps[1]**2, temps[1], 1], 
            [temps[2]**3, temps[2]**2, temps[2], 1],
            [temps[3]**3, temps[3]**2, temps[3], 1]
        ])
        
        b = np.array(strengths)
        
        try:
            # Solve the linear system
            coefficients = np.linalg.solve(A, b)
            a, b_coeff, c, d = coefficients
        except np.linalg.LinAlgError:
            # Fallback to linear interpolation if cubic fails
            print("Warning: Cubic polynomial failed, using linear interpolation")
            return self._linear_interpolation(temps, strengths)
        
        # Generate temperature range - start from first boundary point to avoid negative values
        temp_min = min(temps)  # Start from first boundary point
        temp_max = max(temps)  # End at last boundary point
        temperatures = np.linspace(temp_min, temp_max, self.num_points)
        
        # Calculate strengths using cubic polynomial
        strengths = (a * temperatures**3 + 
                    b_coeff * temperatures**2 + 
                    c * temperatures + d)
        
        # Ensure physical constraints
        strengths = np.clip(strengths, 0, 100)  # Reasonable strength bounds
        
        return temperatures, strengths
    
    def _linear_interpolation(self, temps: List[float], strengths: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fallback linear interpolation between boundary points.
        
        Args:
            temps: List of temperature values
            strengths: List of strength values
        
        Returns:
            Tuple of (temperatures, strengths) arrays
        """
        temp_min, temp_max = self.temperature_range
        temperatures = np.linspace(temp_min, temp_max, self.num_points)
        strengths = np.interp(temperatures, temps, strengths)
        
        return temperatures, strengths
    
    def generate_sealing_profile(self, polymers: List[Dict], compositions: List[float],
                               predicted_adhesion_strength: float,
                               save_csv: bool = True, save_plot: bool = True,
                               save_dir: str = '.', blend_name: str = 'blend') -> Dict:
        """
        Generate complete sealing profile for a polymer blend.
        
        Args:
            polymers: List of polymer dictionaries with properties
            compositions: Volume fractions of each polymer
            predicted_adhesion_strength: ML-predicted adhesion strength
            save_csv: Whether to save CSV file
            save_plot: Whether to save plot
            save_dir: Directory to save outputs
            blend_name: Name for output files
        
        Returns:
            Dictionary with curve data and metadata
        """
        # Calculate boundary points
        boundary_points = calculate_boundary_points(polymers, compositions, predicted_adhesion_strength)
        
        # Generate cubic polynomial curve
        temperatures, strengths = self.generate_cubic_polynomial(boundary_points)
        
        # Validate curve
        if not validate_sealing_curve(temperatures, strengths):
            print("Warning: Generated curve failed validation")
        
        # Create DataFrame
        curve_df = pd.DataFrame({
            'temperature_c': temperatures,
            'sealing_strength_n_per_15mm': strengths
        })
        
        # Save CSV
        if save_csv:
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f'{blend_name}_sealing_profile.csv')
            curve_df.to_csv(csv_path, index=False)
            print(f"Sealing profile saved to: {csv_path}")
        
        # Save plot
        if save_plot:
            self._save_sealing_plot(temperatures, strengths, boundary_points, 
                                  save_dir, blend_name, polymers, compositions)
        
        return {
            'curve_data': curve_df,
            'boundary_points': boundary_points,
            'temperatures': temperatures,
            'strengths': strengths,
            'is_valid': validate_sealing_curve(temperatures, strengths)
        }
    
    def _save_sealing_plot(self, temperatures: np.ndarray, strengths: np.ndarray,
                          boundary_points: Dict[str, Tuple[float, float]],
                          save_dir: str, blend_name: str, polymers: List[Dict] = None, 
                          compositions: List[float] = None):
        """
        Save sealing profile plot with professional dark theme styling.
        
        Args:
            temperatures: Temperature array
            strengths: Strength array
            boundary_points: Dictionary of boundary points
            save_dir: Directory to save plot
            blend_name: Name for output file
            polymers: List of polymer dictionaries (optional)
            compositions: List of compositions (optional)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Set professional dark theme styling
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'black',
            'axes.facecolor': 'black',
            'axes.edgecolor': 'white',
            'axes.labelcolor': 'white',
            'text.color': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'grid.color': 'gray',
            'grid.alpha': 0.3,
            'axes.grid': False,  # No grid for cleaner look
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.linewidth': 1.5,
            'font.size': 12,
            'font.weight': 'normal',
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.facecolor': 'black',
            'legend.edgecolor': 'white',
            'legend.framealpha': 0.8
        })
        
        plt.figure(figsize=(12, 8))
        
        # Main curve with professional styling
        plt.plot(temperatures, strengths, color='#2E8B57', linewidth=3, 
                label='Sealing Profile', alpha=0.9)
        
        # Mark boundary points with distinct colors and styles
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        markers = ['o', 's', '^', 'D']
        
        for i, (name, (temp, strength)) in enumerate(boundary_points.items()):
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            plt.scatter(temp, strength, c=color, s=120, marker=marker, 
                       edgecolors='white', linewidth=2, zorder=5,
                       label=f'{name.replace("_", " ").title()}: ({temp:.0f}°C, {strength:.1f} N/15mm)')
        
        plt.xlabel('Temperature (°C)', fontweight='bold', fontsize=14)
        plt.ylabel('Sealing Strength (N/15mm)', fontweight='bold', fontsize=14)
        
        # Create title with composition information
        if polymers and compositions:
            composition_text = ', '.join([f"{p['grade']} ({c*100:.0f}%)" for p, c in zip(polymers, compositions)])
            plt.title(f'Sealing Profile: {blend_name}\nComposition: {composition_text}', 
                     fontweight='bold', fontsize=16, pad=20)
        else:
            plt.title(f'Sealing Profile: {blend_name}', 
                     fontweight='bold', fontsize=16, pad=20)
        
        # Professional legend styling
        plt.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                  fontsize=11, framealpha=0.9)
        
        # Set axis limits with some padding
        plt.xlim(min(temperatures) - 10, max(temperatures) + 10)
        plt.ylim(0, max(strengths) * 1.1)
        
        # Add subtle background grid
        plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Tight layout for better appearance
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, f'{blend_name}_sealing_profile.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.close()
        print(f"Sealing profile plot saved to: {plot_path}")

def generate_sealing_profile(polymers: List[Dict], compositions: List[float],
                           predicted_adhesion_strength: float,
                           temperature_range: Tuple[float, float] = (0, 300),
                           num_points: int = 100,
                           save_csv: bool = True, save_plot: bool = True,
                           save_dir: str = '.', blend_name: str = 'blend') -> Dict:
    """
    Convenience function to generate sealing profile for a polymer blend.
    
    Args:
        polymers: List of polymer dictionaries with properties
        compositions: Volume fractions of each polymer
        predicted_adhesion_strength: ML-predicted adhesion strength
        temperature_range: (min_temp, max_temp) in Celsius
        num_points: Number of points to generate
        save_csv: Whether to save CSV file
        save_plot: Whether to save plot
        save_dir: Directory to save outputs
        blend_name: Name for output files
    
    Returns:
        Dictionary with curve data and metadata
    """
    generator = SealingCurveGenerator(temperature_range, num_points)
    return generator.generate_sealing_profile(
        polymers, compositions, predicted_adhesion_strength,
        save_csv, save_plot, save_dir, blend_name
    )
