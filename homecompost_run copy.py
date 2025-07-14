#!/usr/bin/env python3
"""
Simple CLI for running single polymer blends using the modular model.
Usage: python homecompost_run.py "material1,grade1,vol_frac1,material2,grade2,vol_frac2" [--thickness THICKNESS]
Example: python homecompost_run.py "PLA,4032D,0.7,PBAT,ecoflex® F Blend C1200,0.3" --thickness 0.050
"""

import sys
import argparse
from homecompost_modules.blend_generator import generate_csv_for_single_blend
from homecompost_modules.plotting import generate_custom_blend_curves

def main():
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
    parser.add_argument('--output', default='blend_curve.png', help='Output PNG filename (default: blend_curve.png)')
    parser.add_argument('--csv', default='blend_data.csv', help='Output CSV filename (default: blend_data.csv)')
    
    args = parser.parse_args()
    
    try:
        print(f"🎯 Generating blend: {args.blend}")
        print(f"📏 Thickness: {args.thickness} mm ({args.thickness*1000:.0f} μm)")
        print(f"📁 Output files: {args.output} (plot), {args.csv} (data)")
        print()
        
        # Generate the plot
        generate_custom_blend_curves([args.blend], args.output, actual_thickness=args.thickness)
        
        # Generate the CSV
        generate_csv_for_single_blend(args.blend, args.csv, actual_thickness=args.thickness)
        
        print(f"\n✅ Success! Generated:")
        print(f"   📊 Plot: {args.output}")
        print(f"   📄 Data: {args.csv}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 