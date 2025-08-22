#!/usr/bin/env python3
"""
Simple Compostability CLI Script
Follows the exact same pattern as other working properties.
"""

import sys
import pandas as pd
import numpy as np
import os
import joblib
from train.modules.blend_feature_extractor import process_blend_features
from train.modules_home.utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves, generate_quintic_biodegradation_curve

def load_material_dictionary(dict_path='material-smiles-dictionary.csv'):
    """Load the material-SMILES dictionary."""
    try:
        df = pd.read_csv(dict_path)
        material_dict = {}
        for _, row in df.iterrows():
            key = (row['Material'].strip(), row['Grade'].strip())
            material_dict[key] = row['SMILES'].strip()
        print(f"✅ Material dictionary loaded with {len(material_dict)} entries")
        return material_dict
    except Exception as e:
        print(f"❌ Error loading material dictionary: {e}")
        return None

def parse_blend_string(blend_string):
    """Parse blend string like "PLA, Ingeo 4043D, 0.5, PBAT, Ecoworld, 0.5" """
    parts = [part.strip() for part in blend_string.split(',')]
    polymers = []
    
    i = 0
    while i < len(parts):
        if i + 2 < len(parts):
            material = parts[i]
            grade = parts[i + 1]
            try:
                fraction = float(parts[i + 2])
                polymers.append((material, grade, fraction))
                i += 3
            except ValueError:
                i += 3
        else:
            break
    
    return polymers

def create_input_dataframe(polymers, material_dict):
    """Create input DataFrame exactly like the working properties do."""
    try:
        # Convert polymers to SMILES
        smiles_list = []
        for material, grade, vol_fraction in polymers:
            key = (material, grade)
            if key in material_dict:
                smiles = material_dict[key]
                smiles_list.append((smiles, vol_fraction))
            else:
                print(f"❌ Material/Grade combination not found: {material} {grade}")
                return None
        
        # Create the basic structure (same as prediction_utils.py)
        data = {
            'Materials': ', '.join([f"{mat} {grade}" for mat, grade, _ in polymers]),
            'Polymer Grade 1': polymers[0][1] if len(polymers) > 0 else 'Unknown',
            'Polymer Grade 2': polymers[1][1] if len(polymers) > 1 else 'Unknown',
            'Polymer Grade 3': polymers[2][1] if len(polymers) > 2 else 'Unknown',
            'Polymer Grade 4': polymers[3][1] if len(polymers) > 3 else 'Unknown',
            'Polymer Grade 5': polymers[4][1] if len(polymers) > 4 else 'Unknown',
            'SMILES1': smiles_list[0][0] if len(smiles_list) > 0 else '',
            'SMILES2': smiles_list[1][0] if len(smiles_list) > 1 else '',
            'SMILES3': smiles_list[2][0] if len(smiles_list) > 2 else '',
            'SMILES4': smiles_list[3][0] if len(smiles_list) > 3 else '',
            'SMILES5': smiles_list[4][0] if len(smiles_list) > 4 else '',
            'vol_fraction1': smiles_list[0][1] if len(smiles_list) > 0 else 0.0,
            'vol_fraction2': smiles_list[1][1] if len(smiles_list) > 1 else 0.0,
            'vol_fraction3': smiles_list[2][1] if len(smiles_list) > 2 else 0.0,
            'vol_fraction4': smiles_list[3][1] if len(smiles_list) > 3 else 0.0,
            'vol_fraction5': smiles_list[4][1] if len(smiles_list) > 4 else 0.0
        }
        
        # Fill unused columns with proper defaults
        for i in range(len(polymers), 5):
            data[f'Polymer Grade {i+1}'] = 'Unknown'
            data[f'SMILES{i+1}'] = ''
            data[f'vol_fraction{i+1}'] = 0.0
        
        df = pd.DataFrame([data])
        return df
        
    except Exception as e:
        print(f"❌ Error creating input DataFrame: {e}")
        return None

def prepare_features_for_prediction(featurized_df):
    """Prepare features exactly like the working properties do."""
    try:
        # Remove metadata columns (same logic as prediction_utils.py)
        exclude_cols = ['Materials', 'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5']
        X = featurized_df.drop(columns=[col for col in exclude_cols if col in featurized_df.columns])
        
        # Remove target columns if they exist
        target_cols = ['property1', 'property2']
        X = X.drop(columns=[col for col in target_cols if col in X.columns])
        
        # Handle categorical features (same as prediction_utils.py)
        categorical_features = []
        numerical_features = []
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'string':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        # Fill missing values (same as prediction_utils.py)
        for col in categorical_features:
            X[col] = X[col].fillna('Unknown')
        for col in numerical_features:
            X[col] = X[col].fillna(0)
        
        # Convert categorical features to category dtype (same as prediction_utils.py)
        for col in categorical_features:
            X[col] = X[col].astype('category')
        
        return X
        
    except Exception as e:
        print(f"❌ Error preparing features: {e}")
        return None

def predict_compostability(blend_string, output_prefix="cli_prediction", model_dir="train/models/eol/v5/"):
    """Main prediction function using the same pattern as working properties."""
    print("="*60)
    print("SIMPLE COMPOSTABILITY PREDICTION CLI")
    print("="*60)
    print(f"Blend: {blend_string}")
    print(f"Output prefix: {output_prefix}")
    print(f"Model directory: {model_dir}")
    
    try:
        # Step 1: Load material dictionary (same as working properties)
        material_dict = load_material_dictionary()
        if material_dict is None:
            return None
        
        # Step 2: Parse blend string (same as working properties)
        polymers = parse_blend_string(blend_string)
        if not polymers:
            print("❌ Failed to parse blend string")
            return None
        
        print(f"Found {len(polymers)} polymers: {polymers}")
        
        # Step 3: Create input DataFrame (same as working properties)
        input_df = create_input_dataframe(polymers, material_dict)
        if input_df is None:
            return None
        
        print(f"Created input DataFrame with shape: {input_df.shape}")
        
        # Step 4: Save to temp file and featurize (same as working properties)
        temp_input_file = "temp_blend_input.csv"
        temp_features_file = "temp_blend_features.csv"
        input_df.to_csv(temp_input_file, index=False)
        
        featurized_df = process_blend_features(temp_input_file, temp_features_file)
        if featurized_df is None:
            print("❌ Featurization failed")
            return None
        
        print(f"Featurized DataFrame shape: {featurized_df.shape}")
        
        # Clean up temp files
        if os.path.exists(temp_input_file):
            os.remove(temp_input_file)
        if os.path.exists(temp_features_file):
            os.remove(temp_features_file)
        
        # Step 5: Prepare features for prediction (same as working properties)
        features_df = prepare_features_for_prediction(featurized_df)
        if features_df is None:
            return None
        
        print(f"Features DataFrame shape: {features_df.shape}")
        
        # Step 6: Load models (same pattern as working properties)
        model_max_L_path = os.path.join(model_dir, "comprehensive_polymer_model_max_L.pkl")
        model_t0_path = os.path.join(model_dir, "comprehensive_polymer_model_t0.pkl")
        
        if not os.path.exists(model_max_L_path) or not os.path.exists(model_t0_path):
            print(f"❌ Model files not found in {model_dir}")
            return None
        
        model_max_L = joblib.load(model_max_L_path)
        model_t0 = joblib.load(model_t0_path)
        print("✅ Models loaded successfully")
        
        # Step 7: Make predictions (same as working properties)
        print("\nMaking predictions...")
        max_L_pred = model_max_L.predict(features_df)[0]
        t0_pred = model_t0.predict(features_df)[0]
        
        # Convert from log scale (same as working properties)
        max_L_pred = np.exp(max_L_pred) - 1e-6
        t0_pred = np.exp(t0_pred)
        
        print(f"Predicted max_L: {max_L_pred:.2f}")
        print(f"Predicted t0: {t0_pred:.2f}")
        
        # Step 8: Generate curves (added on top)
        print("\nGenerating curves...")
        
        # Calculate k0 values
        k0_disintegration = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred, t_max=200.0, 
                                                           majority_polymer_high_disintegration=True,
                                                           actual_thickness=0.050)
        k0_biodegradation = calculate_k0_from_sigmoid_params(max_L_pred, t0_pred * 2.0, t_max=400.0, 
                                                           majority_polymer_high_disintegration=True,
                                                           actual_thickness=0.050)
        
        print(f"k0 (Disintegration): {k0_disintegration:.4f}")
        print(f"k0 (Biodegradation): {k0_biodegradation:.4f}")
        
        # Create output directory
        output_dir = f"test_results/{output_prefix}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate sigmoid curves
        disintegration_df = generate_sigmoid_curves(
            np.array([max_L_pred]), 
            np.array([t0_pred]), 
            np.array([k0_disintegration]), 
            days=200, 
            curve_type='disintegration',
            save_dir=output_dir,
            actual_thickness=0.050
        )
        
        # Generate biodegradation curves
        biodegradation_df = generate_quintic_biodegradation_curve(
            disintegration_df, 
            t0_pred, 
            max_L_pred, 
            days=400, 
            save_dir=output_dir,
            actual_thickness=0.050
        )
        
        print(f"\n✅ Prediction complete! Results saved to: {output_dir}")
        
        return {
            'max_L': max_L_pred,
            't0': t0_pred,
            'k0_disintegration': k0_disintegration,
            'k0_biodegradation': k0_biodegradation,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python predict_compostability_cli_new.py 'Material1, Grade1, fraction1, Material2, Grade2, fraction2' [output_prefix]")
        sys.exit(1)
    
    blend_string = sys.argv[1]
    output_prefix = sys.argv[2] if len(sys.argv) > 2 else "cli_prediction"
    
    result = predict_compostability(blend_string, output_prefix)
    
    if result:
        print(f"\nPrediction Summary:")
        print(f"Max_L: {result['max_L']:.2f}")
        print(f"t0: {result['t0']:.2f} days")
        print(f"k0 (Disintegration): {result['k0_disintegration']:.4f}")
        print(f"k0 (Biodegradation): {result['k0_biodegradation']:.4f}")
        print(f"Results saved to: {result['output_dir']}")
    else:
        print("❌ Prediction failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
