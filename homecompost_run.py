#!/usr/bin/env python3
"""
Homecompost prediction function for polymer blends.
Returns a dictionary of {time: prediction} for disintegration over time.
"""

from homecompost_modules.blend_generator import generate_csv_for_single_blend

def predict_homecompost(blend_str, thickness=0.050):
    """
    Predict homecompost disintegration for a polymer blend.
    
    Args:
        blend_str: Blend string in format "material1,grade1,vol_frac1,material2,grade2,vol_frac2,..."
        thickness: Actual thickness in mm (default: 0.050 mm = 50 μm)
    
    Returns:
        Dictionary of {time: prediction} where time is days and prediction is disintegration percentage
    """
    try:
        # Generate the disintegration profile
        time_prediction_dict = generate_csv_for_single_blend(blend_str, output_path=None, actual_thickness=thickness)
        return time_prediction_dict
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None 

result_dict = predict_homecompost("PLA,Ingeo 4043D,0.5,PBAT,ecoflex® F Blend C1200,0.5", thickness=0.050)
print(result_dict)