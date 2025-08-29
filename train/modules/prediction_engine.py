#!/usr/bin/env python3
"""
Prediction engine for polymer blend property prediction.
"""

import os
import tempfile
import logging
from .prediction_utils import (
    PROPERTY_CONFIGS, get_env_params_for_property, convert_polymers_to_smiles,
    create_input_dataframe, load_model, prepare_features_for_prediction,
    predict_property
)
from .blend_feature_extractor import process_blend_features
from .error_calculator import ErrorCalculator
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

def predict_blend_property(property_type, polymers, available_env_params, material_dict, model_path=None, include_errors=True):
    """
    Predict a property type (single or multiple) with optional error quantification.
    Handles both single-property predictions (WVTR, TS, EAB, Cobb, OTR, Adhesion) 
    and multi-property predictions (compostability with max_L and t0).
    
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
        
        # Prepare features for prediction
        features_df = prepare_features_for_prediction(featurized_df, None, property_type)
        if features_df is None:
            return None
        
        # Handle different property types
        if property_type == 'compost':
            # Multi-property prediction: compostability (max_L and t0)
            model_dir = model_path if model_path else config['model_path']
            model_dir = model_dir if model_dir.endswith('/') else model_dir + '/'
            
            # Load both models
            model_max_L_path = os.path.join(model_dir, "comprehensive_polymer_model_max_L.pkl")
            model_t0_path = os.path.join(model_dir, "comprehensive_polymer_model_t0.pkl")
            
            if not os.path.exists(model_max_L_path) or not os.path.exists(model_t0_path):
                logger.error(f"❌ Compostability model files not found in {model_dir}")
                return None
            
            import joblib
            model_max_L = joblib.load(model_max_L_path)
            model_t0 = joblib.load(model_t0_path)
            
            # Make predictions
            max_L_pred = model_max_L.predict(features_df)[0]
            t0_pred = model_t0.predict(features_df)[0]
            
            # Convert from log scale if needed
            if config['log_scale']:
                max_L_pred = np.exp(max_L_pred) - 1e-6
                t0_pred = np.exp(t0_pred)
            
            # Create blend label
            blend_label = " + ".join([f"{material} {grade} ({vol_fraction:.1%})" for material, grade, vol_fraction in polymers])
            
            result = {
                'property_type': property_type,
                'name': config['name'],
                'unit': config['unit'],
                'prediction': max_L_pred,  # Primary prediction (max disintegration)
                'env_params': env_params,
                'blend_label': blend_label,
                't0_pred': t0_pred,
                'model_path': model_dir,
                'max_L_pred': max_L_pred,
                'thickness': env_params.get('Thickness (um)', 50) / 1000.0  # Convert to mm
            }
            
        elif property_type == 'adhesion' and config.get('is_dual_property', False):
            # Dual-property prediction: adhesion (sealing temperature + adhesion strength)
            model_dir = model_path if model_path else config['model_path']
            model_dir = model_dir if model_dir.endswith('/') else model_dir + '/'
            
            # Load both models
            sealing_temp_model_path = os.path.join(model_dir, "sealing_temperature_model.pkl")
            adhesion_strength_model_path = os.path.join(model_dir, "adhesion_strength_model.pkl")
            
            if not os.path.exists(sealing_temp_model_path) or not os.path.exists(adhesion_strength_model_path):
                logger.error(f"❌ Adhesion model files not found in {model_dir}")
                return None
            
            import joblib
            sealing_temp_model = joblib.load(sealing_temp_model_path)
            adhesion_strength_model = joblib.load(adhesion_strength_model_path)
            
            # Make predictions
            sealing_temp_pred = sealing_temp_model.predict(features_df)[0]
            adhesion_strength_pred = adhesion_strength_model.predict(features_df)[0]
            
            # Convert from log scale if needed
            if config['log_scale']:
                sealing_temp_pred = np.exp(sealing_temp_pred)
                adhesion_strength_pred = np.exp(adhesion_strength_pred)
            
            # Create blend label
            blend_label = " + ".join([f"{material} {grade} ({vol_fraction:.1%})" for material, grade, vol_fraction in polymers])
            
            result = {
                'property_type': property_type,
                'name': config['name'],
                'unit': config['unit'],
                'prediction': adhesion_strength_pred,  # Primary prediction (adhesion strength)
                'env_params': env_params,
                'blend_label': blend_label,
                'sealing_temp_pred': sealing_temp_pred,  # Secondary prediction (sealing temperature)
                'model_path': model_dir,
                'thickness': env_params.get('Thickness (um)', 50) / 1000.0  # Convert to mm
            }
            
        else:
            # Single property prediction (WVTR, TS, EAB, Cobb, OTR, Adhesion)
            # Load model
            model = load_model(property_type, model_path)
            if model is None:
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
                error_bounds = error_calc.calculate_error_bounds(property_type, result['prediction'])
                if error_bounds:
                    result['error_bounds'] = error_bounds
                    result['error_calculator'] = error_calc
            except Exception as e:
                logger.warning(f"⚠️ Error calculation failed for {property_type}: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in {property_type} prediction: {e}")
        return None
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_input_file)
            if os.path.exists(temp_input_file.replace('.csv', '_featurized.csv')):
                os.unlink(temp_input_file.replace('.csv', '_featurized.csv'))
        except:
            pass 