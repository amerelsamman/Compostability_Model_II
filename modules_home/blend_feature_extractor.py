"""
Polymer Blends Feature Extractor
Processes polymer blends using individual molecular features weighted by volume fractions.
"""

import pandas as pd
import numpy as np
import logging
from .feature_extractor import FeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def process_blend_features(input_file, output_file):
    """
    Process polymer blends and compute weighted features based on volume fractions.
    
    Args:
        input_file (str): Path to input CSV file with blend data
        output_file (str): Path to output CSV file with weighted features
    """
    # Read input data
    df = pd.read_csv(input_file)
    
    # Define the exact feature order from feature_extractor.py
    feature_order = [
        'SP_C', 'SP_N', 'SP2_C', 'SP2_N', 'SP2_O', 'SP2_S', 'SP2_B',
        'SP3_C', 'SP3_N', 'SP3_O', 'SP3_S', 'SP3_P', 'SP3_Si', 'SP3_B',
        'SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I', 'SP3D2_S',
        'phenyls', 'cyclohexanes', 'cyclopentanes', 'cyclopentenes', 'thiophenes',
        'aromatic_rings_with_n', 'aromatic_rings_with_o', 'aromatic_rings_with_n_o',
        'aromatic_rings_with_s', 'aliphatic_rings_with_n', 'aliphatic_rings_with_o',
        'aliphatic_rings_with_n_o', 'aliphatic_rings_with_s', 'other_rings',
        'carboxylic_acid', 'anhydride', 'acyl_halide', 'carbamide', 'urea',
        'carbamate', 'thioamide', 'amide', 'ester', 'sulfonamide', 'sulfone',
        'sulfoxide', 'phosphate', 'nitro', 'acetal', 'ketal', 'isocyanate',
        'thiocyanate', 'azide', 'azo', 'imide', 'sulfonyl_halide', 'phosphonate',
        'thiourea', 'guanidine', 'silicon_4_coord', 'boron_3_coord', 'vinyl',
        'vinyl_halide', 'allene', 'alcohol', 'ether', 'aldehyde', 'ketone',
        'thiol', 'thioether', 'primary_amine', 'secondary_amine', 'tertiary_amine',
        'quaternary_amine', 'imine', 'nitrile', 'primary_carbon', 'secondary_carbon',
        'tertiary_carbon', 'quaternary_carbon',
        'branching_factor', 'tree_depth', 'ethyl_chain', 'propyl_chain', 'butyl_chain', 'long_chain'
    ]
    
    # Process each blend
    blend_features_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        try:
            # Extract SMILES and volume fractions
            smiles_list = []
            vol_fractions = []
            
            # Collect all non-empty SMILES and their corresponding volume fractions
            for i in range(1, 6):  # Handle up to 5 components
                smiles_col = f'SMILES{i}'
                vol_col = f'vol_fraction{i}'
                
                if pd.notna(row[smiles_col]) and str(row[smiles_col]).strip() != '':
                    smiles_list.append(row[smiles_col])
                    # If volume fraction is NaN or empty, treat as single polymer (100% weight)
                    if pd.isna(row[vol_col]) or str(row[vol_col]).strip() == '':
                        vol_fractions.append(1.0 if len(smiles_list) == 1 else 0.0)
                    else:
                        vol_fractions.append(float(row[vol_col]))
            
            # Validate input
            if len(smiles_list) == 0:
                continue
                
            if len(smiles_list) != len(vol_fractions):
                continue
            
            # For single polymers, set volume fraction to 1.0
            if len(smiles_list) == 1 and vol_fractions[0] == 1.0:
                vol_fractions = [1.0]
            # For blends, check if volume fractions sum to 1.0 (with more lenient tolerance)
            elif len(smiles_list) > 1 and not np.isclose(sum(vol_fractions), 1.0, atol=1e-2):
                continue
            
            # Initialize weighted features
            weighted_features = {feature: 0.0 for feature in feature_order}
            
            # Process each polymer in the blend
            for smiles, vol_frac in zip(smiles_list, vol_fractions):
                # Extract features for this polymer
                polymer_features = FeatureExtractor.extract_all_features(smiles)
                
                if polymer_features is None:
                    continue
                
                # Weight the features by volume fraction
                for feature, value in polymer_features.items():
                    if feature in weighted_features:
                        weighted_features[feature] += value * vol_frac
            
            blend_features_list.append(weighted_features)
            valid_indices.append(idx)
            
        except Exception as e:
            continue
    
    # Convert features to DataFrame
    features_df = pd.DataFrame(blend_features_list)
    
    # Get only the valid blends from the original dataframe
    valid_df = df.iloc[valid_indices]
    
    # Define the output column order
    output_columns = [
        'Materials', 'Polymer Grade 1', 'Polymer Grade 2', 'Polymer Grade 3', 'Polymer Grade 4', 'Polymer Grade 5',
        'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5',
        'vol_fraction1', 'vol_fraction2', 'vol_fraction3', 'vol_fraction4', 'vol_fraction5'
    ]
    
    # Add environmental columns if they exist
    env_columns = ['Temperature (C)', 'RH (%)', 'Thickness (um)']
    for col in env_columns:
        if col in valid_df.columns:
            output_columns.append(col)
    
    # Create the final result DataFrame
    result_parts = []
    
    # Add metadata columns (excluding property for now)
    metadata_cols = [col for col in output_columns if col in valid_df.columns]
    result_parts.append(valid_df[metadata_cols].reset_index(drop=True))
    
    # Add weighted features
    result_parts.append(features_df.reset_index(drop=True))
    
    # Combine all parts
    final_result = pd.concat(result_parts, axis=1)
    
    # Add property column if it exists in the original data
    if 'property' in valid_df.columns:
        final_result['property'] = valid_df['property'].values
    
    # Reorder columns to match desired output order
    final_columns = []
    for col in output_columns:
        if col in final_result.columns:
            final_columns.append(col)
    
    # Add feature columns in the correct order
    final_columns.extend(feature_order)
    
    # Add property column last
    if 'property' in final_result.columns:
        final_columns.append('property')
    
    # Reorder the DataFrame
    final_result = final_result[final_columns]
    
    # Save results
    final_result.to_csv(output_file, index=False)
    
    return final_result

def main():
    """Main function to run the blend feature extraction."""
    input_file = "data/wvtr/training.csv"
    output_file = "training_features.csv"
    
    logger.info("Starting polymer blend feature extraction...")
    result = process_blend_features(input_file, output_file)
    logger.info("✅ Blend feature extraction completed successfully!")

if __name__ == "__main__":
    main() 