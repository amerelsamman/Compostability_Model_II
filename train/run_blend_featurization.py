#!/usr/bin/env python3
"""
Wrapper script to run polymer blend feature extraction.
Usage: python run_blend_featurization.py [input_file] [output_file]
"""

import sys
import os
from modules.blend_feature_extractor import process_blend_features

def main():
    """Main function to run blend feature extraction with command line arguments."""
    # Default values
    input_file = "data/wvtr/training.csv"
    output_file = "training_features.csv"
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    print(f"Processing blend features...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-" * 50)
    
    try:
        result = process_blend_features(input_file, output_file)
        print(f"\n✅ Successfully processed {len(result)} blends!")
        print(f"Output saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 