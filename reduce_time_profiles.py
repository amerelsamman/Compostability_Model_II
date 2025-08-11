#!/usr/bin/env python3
"""
Script to reduce time profiles in blendsdatabase_v3.csv
- Disintegration: sample every 15 days starting from 0
- Biodegradation: sample every 30 days starting from 0
"""

import pandas as pd
import json
import ast
import sys

def reduce_time_profile(profile_str, interval_days):
    """
    Reduce a time profile dictionary to sample every interval_days starting from 0
    
    Args:
        profile_str: String representation of the time profile dictionary
        interval_days: Interval in days to sample (15 for disintegration, 30 for biodegradation)
    
    Returns:
        Reduced time profile dictionary as a string
    """
    try:
        # Parse the profile string to get the dictionary
        if isinstance(profile_str, str):
            # Handle the case where the string might be a JSON string
            try:
                profile_dict = json.loads(profile_str)
            except json.JSONDecodeError:
                # Try ast.literal_eval as fallback
                profile_dict = ast.literal_eval(profile_str)
        else:
            profile_dict = profile_str
        
        # Convert string keys to integers and get the max day
        profile_dict = {int(k): float(v) for k, v in profile_dict.items()}
        max_day = max(profile_dict.keys())
        
        # Create reduced profile with sampling every interval_days
        reduced_profile = {}
        for day in range(0, max_day + 1, interval_days):
            if day in profile_dict:
                reduced_profile[day] = profile_dict[day]
        
        # Always include the last day if it's not already included
        if max_day not in reduced_profile:
            reduced_profile[max_day] = profile_dict[max_day]
        
        return json.dumps(reduced_profile)
    
    except Exception as e:
        print(f"Error processing profile: {e}")
        return profile_str

def main():
    # Read the input file
    input_file = "blendsdatabase_v3.csv"
    output_file = "blendsdatabase_v3_2.csv"
    
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)
    
    print(f"Original file shape: {df.shape}")
    print(f"Processing {len(df)} rows...")
    
    # Process disintegration_time_profile (every 15 days)
    print("Reducing disintegration profiles (every 15 days)...")
    df['disintegration_time_profile'] = df['disintegration_time_profile'].apply(
        lambda x: reduce_time_profile(x, 15)
    )
    
    # Process biodegradation_time_profile (every 30 days)
    print("Reducing biodegradation profiles (every 30 days)...")
    df['biodegradation_time_profile'] = df['biodegradation_time_profile'].apply(
        lambda x: reduce_time_profile(x, 30)
    )
    
    # Save the reduced file
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print(f"✅ Successfully created {output_file}")
    print(f"Final file shape: {df.shape}")
    
    # Show sample of reduced profiles
    print("\nSample of reduced profiles:")
    print("First row disintegration (every 15 days):")
    sample_dis = json.loads(df.iloc[0]['disintegration_time_profile'])
    print(f"  Days: {list(sample_dis.keys())[:10]}...")
    print(f"  Values: {list(sample_dis.values())[:10]}...")
    
    print("\nFirst row biodegradation (every 30 days):")
    sample_bio = json.loads(df.iloc[0]['biodegradation_time_profile'])
    print(f"  Days: {list(sample_bio.keys())[:10]}...")
    print(f"  Values: {list(sample_bio.values())[:10]}...")

if __name__ == "__main__":
    main() 