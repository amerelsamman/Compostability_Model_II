import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .core_model import (
    sus, DAYS, SAMPLE_AREA, THICKNESS_DEFAULT,
    generate_material_curve, parse_thickness, find_material_by_grade,
    parse_blend_input, is_home_compostable_certified, sigmoid, get_max_disintegration_hybrid
)
from .blend_generator import generate_blend

def generate_custom_blend_curves(blend_strings, output_filename, actual_thickness=None):
    """Generate curves for custom blends"""
    print("Generating custom blend curves...")
    
    custom_blend_curves = []
    custom_blend_labels = []
    
    for i, blend_str in enumerate(blend_strings):
        try:
            print(f"\nProcessing custom blend {i+1}: {blend_str}")
            
            # Use the core blend generation function
            material_info, blend_curve = generate_blend(blend_str, actual_thickness=actual_thickness)
            
            custom_blend_curves.append(blend_curve)
            
            # Create label
            label_parts = []
            for material in material_info:
                label_parts.append(f"{material['polymer']} {material['grade']} ({material['vol_frac']:.1%})")
            custom_blend_labels.append(" + ".join(label_parts))
            
            print(f"  Generated blend curve with max disintegration: {np.max(blend_curve):.1f}%")
                
        except ValueError as e:
            print(f"ERROR: {e}")
            return
    
    # Plot custom blends
    if custom_blend_curves:
        import matplotlib as mpl
        plt.close('all')
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='#000000')
        ax.set_facecolor('#000000')
        
        colors = ['#8942E5']  # Use only the specified purple
        for i, (curve, label) in enumerate(zip(custom_blend_curves, custom_blend_labels)):
            color = colors[i % len(colors)]
            x = np.arange(1, DAYS+1)
            y = curve
            ax.plot(x, y, label=label, linewidth=2, color=color)
            print(f"Plotted: {label} (max: {np.max(y):.1f}%, 90d: {y[89]:.1f}%)")
        # Set axis and title colors
        ax.tick_params(colors='white', which='both')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        # Set concise title using only polymer names
        if len(custom_blend_curves) == 1 and len(material_info) > 0:
            polymer_names = [mat['polymer'] for mat in material_info]
            blend_name = "/".join(polymer_names)
            blend_title = f"Rate of Disintegration of {blend_name} blend"
        else:
            blend_title = "Rate of Disintegration of Custom Blend"
        ax.set_title(blend_title, color='white', fontsize=18, weight='bold')
        ax.grid(False)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Time (day)', color='white')
        ax.set_ylabel('Disintegration (%)', color='white')
        # Remove top and right spines for open graph look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        # No legend for clean look
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\nCustom blend plot saved as {output_filename}")
    else:
        print("No custom blend curves to plot!") 