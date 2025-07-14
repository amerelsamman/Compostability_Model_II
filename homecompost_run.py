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

# Keep CLI functionality for backward compatibility
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate disintegration curve for a single polymer blend',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python homecompost_run.py "PLA,4032D,0.7,PBAT,ecoflex® F Blend C1200,0.3"
  python homecompost_run.py "PHA,PHACT S1000P,0.5,PHA,PHACT A1000P,0.5" --thickness 0.100
  python homecompost_run.py "PLA,Bioplast GF 106,0.6,PLA,Bioplast 500 A,0.4" --thickness 0.025
        """
    )
    parser.add_argument('blend', help='Blend string in format: "material1,grade1,vol_frac1,material2,grade2,vol_frac2,..."')
    parser.add_argument('--thickness', type=float, default=0.050, help='Actual thickness in mm (default: 0.050 mm = 50 μm)')
    parser.add_argument('--csv', default='blend_data.csv', help='Output CSV filename (default: blend_data.csv)')
    
    args = parser.parse_args()
    
    try:
        print(f"🎯 Generating blend: {args.blend}")
        print(f"📏 Thickness: {args.thickness} mm ({args.thickness*1000:.0f} μm)")
        print(f"📁 Output file: {args.csv} (data)")
        print()
        
        # Generate the CSV and get the dictionary
        time_prediction_dict = generate_csv_for_single_blend(args.blend, args.csv, actual_thickness=args.thickness)
        
        print(f"\n✅ Success! Generated:")
        print(f"   📄 Data: {args.csv}")
        print(f"   📊 Predictions: {len(time_prediction_dict)} time points")
        
        # Show a few sample predictions
        print(f"\n📈 Sample predictions:")
        for day in [1, 7, 14, 28, 56, 84]:
            if day in time_prediction_dict:
                print(f"   Day {day}: {time_prediction_dict[day]:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1) 