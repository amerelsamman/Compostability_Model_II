#!/usr/bin/env python3
"""
Usage Examples for Polymer Blend Property Predictor

This file shows how to use the predict_property() function for various scenarios.
"""

from polymer_blend_predictor import predict_property, predict_all_properties, get_available_materials

def example_1_basic_wvtr():
    """Example 1: Basic WVTR prediction for a PLA/PBAT blend"""
    print("=== Example 1: WVTR Prediction ===")
    
    result = predict_property(
        polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
        property_name="wvtr",
        temperature=25,
        rh=60,
        thickness=100
    )
    
    if result['success']:
        print(f"✅ {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
        print(f"   Blend: {result['polymers']}")
        print(f"   Conditions: {result['environmental_params']}")
    else:
        print(f"❌ {result['error_message']}")

def example_2_tensile_strength():
    """Example 2: Tensile Strength prediction for a PLA/PCL blend"""
    print("\n=== Example 2: Tensile Strength Prediction ===")
    
    result = predict_property(
        polymers=[("PLA", "4032D", 0.7), ("PCL", "Capa 6500", 0.3)],
        property_name="ts",
        thickness=50
    )
    
    if result['success']:
        print(f"✅ {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print(f"❌ {result['error_message']}")

def example_3_single_polymer():
    """Example 3: Single polymer property prediction"""
    print("\n=== Example 3: Single Polymer (Cobb Value) ===")
    
    result = predict_property(
        polymers=[("PLA", "4032D", 1.0)],  # Single polymer, 100% fraction
        property_name="cobb"
    )
    
    if result['success']:
        print(f"✅ {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print(f"❌ {result['error_message']}")

def example_4_three_component_blend():
    """Example 4: Three-component blend"""
    print("\n=== Example 4: Three-Component Blend ===")
    
    result = predict_property(
        polymers=[
            ("PLA", "4032D", 0.5),
            ("PBAT", "Ecoworld", 0.3),
            ("PCL", "Capa 6500", 0.2)
        ],
        property_name="eab",
        thickness=75
    )
    
    if result['success']:
        print(f"✅ {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
    else:
        print(f"❌ {result['error_message']}")

def example_5_all_properties():
    """Example 5: Predict all properties at once"""
    print("\n=== Example 5: All Properties ===")
    
    all_results = predict_all_properties(
        polymers=[("PLA", "4032D", 0.6), ("PBAT", "Ecoworld", 0.4)],
        temperature=23,
        rh=50,
        thickness=75
    )
    
    for prop_name, result in all_results.items():
        if result['success']:
            print(f"✅ {result['property_name']}: {result['prediction']:.2f} {result['unit']}")
        else:
            print(f"❌ {prop_name.upper()}: {result['error_message']}")

def example_6_error_handling():
    """Example 6: Error handling examples"""
    print("\n=== Example 6: Error Handling ===")
    
    # Invalid material
    result = predict_property(
        polymers=[("INVALID", "MATERIAL", 1.0)],
        property_name="wvtr",
        temperature=25,
        rh=60,
        thickness=100
    )
    print(f"Invalid material: {result['error_message']}")
    
    # Volume fractions don't sum to 1.0
    result = predict_property(
        polymers=[("PLA", "4032D", 0.3), ("PBAT", "Ecoworld", 0.3)],  # Only sums to 0.6
        property_name="wvtr",
        temperature=25,
        rh=60,
        thickness=100
    )
    print(f"Invalid fractions: {result['error_message']}")
    
    # Missing required parameter
    result = predict_property(
        polymers=[("PLA", "4032D", 1.0)],
        property_name="wvtr",
        # Missing temperature, rh, thickness
    )
    print(f"Missing parameters: {result['error_message']}")

def show_available_materials():
    """Show available materials and their grades"""
    print("\n=== Available Materials ===")
    materials = get_available_materials()
    
    if materials:
        print(f"Total materials available: {len(materials)}")
        for material, grade in materials:
            print(f"  {material}, {grade}")
    else:
        print("No materials found. Check if material-smiles-dictionary.csv exists.")

def main():
    """Run all examples"""
    print("Polymer Blend Property Predictor - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_1_basic_wvtr()
    example_2_tensile_strength()
    example_3_single_polymer()
    example_4_three_component_blend()
    example_5_all_properties()
    example_6_error_handling()
    show_available_materials()

if __name__ == "__main__":
    main() 