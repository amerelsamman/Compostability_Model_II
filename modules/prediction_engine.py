#!/usr/bin/env python3
"""
Prediction engine for polymer blend property prediction.
"""

import os
import tempfile
import logging
from modules.prediction_utils import (
    PROPERTY_CONFIGS, get_env_params_for_property, convert_polymers_to_smiles,
    create_input_dataframe, load_model, prepare_features_for_prediction,
    predict_property
)
from modules.blend_feature_extractor import process_blend_features
from modules.error_calculator import ErrorCalculator

# Set up logging
logger = logging.getLogger(__name__)

def predict_single_property(property_type, polymers, available_env_params, material_dict, model_path=None, include_errors=True):
    """
    Predict a single property type with optional error quantification.
    
    Args:
        property_type: property type to predict
        polymers: list of polymer tuples
        available_env_params: available environmental parameters
        material_dict: material dictionary
        model_path: optional custom model path
        include_errors: whether to include error calculations
    
    Returns:
        prediction result dict or None if failed
    """
    config = PROPERTY_CONFIGS[property_type]
    
    # Get environmental parameters for this property (can include missing values)
    env_params = get_env_params_for_property(available_env_params, property_type)
    
    # Convert polymers to SMILES
    smiles_list = convert_polymers_to_smiles(polymers, material_dict)
    if smiles_list is None:
        return None
    
    # Create input DataFrame
    input_df = create_input_dataframe(smiles_list, polymers, env_params)
    if input_df is None:
        return None
    
    # Save input to temporary file for featurization
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        input_df.to_csv(f.name, index=False)
        temp_input_file = f.name
    
    try:
        # Featurize the blend
        temp_output_file = temp_input_file.replace('.csv', '_featurized.csv')
        featurized_df = process_blend_features(temp_input_file, temp_output_file)
        
        if featurized_df is None or len(featurized_df) == 0:
            logger.error("❌ Featurization failed or produced empty result")
            return None
        
        # Load model
        model = load_model(property_type, model_path)
        if model is None:
            return None
        
        # Prepare features for prediction
        features_df = prepare_features_for_prediction(featurized_df, model, property_type)
        if features_df is None:
            return None
        
        # Make prediction
        prediction = predict_property(features_df, model, property_type)
        if prediction is None:
            return None
        
        result = {
            'property_type': property_type,
            'name': config['name'],
            'unit': config['unit'],
            'prediction': prediction,
            'env_params': env_params
        }
        
        # Add error calculations if requested
        if include_errors:
            try:
                error_calc = ErrorCalculator()
                error_bounds = error_calc.calculate_error_bounds(property_type, prediction)
                if error_bounds:
                    result['error_bounds'] = error_bounds
                    result['error_calculator'] = error_calc
            except Exception as e:
                logger.warning(f"⚠️ Error calculation failed for {property_type}: {e}")
        
        return result
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input_file)
            if os.path.exists(temp_input_file.replace('.csv', '_featurized.csv')):
                os.unlink(temp_input_file.replace('.csv', '_featurized.csv'))
        except:
            pass 