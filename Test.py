from polymer_blend_predictor import predict_property
# Predict WVTR for a PLA/PBAT blend
result = predict_property(
    polymers=[("PLA", "4032D", 0.5), ("PBAT", "Ecoworld", 0.5)],
    property_name="wvtr",
    temperature=25,
    rh=60,
    thickness=100
)

if result['success']:
    print(f"{result['property_name']}: {result['prediction']:.2f} {result['unit']}")
else:
    print(f"Error: {result['error_message']}")


result = predict_property(
    polymers=[("PLA", "4032D", 0.7), ("PCL", "Capa 6500", 0.3)],
    property_name="ts",
    thickness=50
)

if result['success']:
    print(f"{result['property_name']}: {result['prediction']:.2f} {result['unit']}")
else:
    print(f"Error: {result['error_message']}")
    