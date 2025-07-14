#!/usr/bin/env python3
"""
Test script for the new homecompost prediction function.
"""

from homecompost_run import predict_homecompost

def test_homecompost_function():
    print("Testing homecompost prediction function...")
    
    # Test blend string
    blend_str = "PLA,4032D,0.7,PBAT,ecoflex® F Blend C1200,0.3"
    thickness = 0.050  # 50 μm
    
    print(f"Blend: {blend_str}")
    print(f"Thickness: {thickness} mm")
    print()
    
    # Get predictions
    predictions = predict_homecompost(blend_str, thickness)
    
    if predictions is not None:
        print(f"✅ Success! Got {len(predictions)} time points")
        print()
        print("📈 Sample predictions:")
        
        # Show predictions for key time points
        key_days = [1, 7, 14, 28, 56, 84, 112]
        for day in key_days:
            if day in predictions:
                print(f"   Day {day:3d}: {predictions[day]:6.2f}%")
        
        print()
        print("📊 Full prediction dictionary:")
        print(f"   {predictions}")
        
    else:
        print("❌ Failed to get predictions")

if __name__ == "__main__":
    test_homecompost_function() 