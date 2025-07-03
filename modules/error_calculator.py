#!/usr/bin/env python3
"""
Error calculator for polymer blend property prediction uncertainty quantification.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

# Set up logging
logger = logging.getLogger(__name__)

class ErrorCalculator:
    """Calculate error bars and uncertainty for property predictions."""
    
    def __init__(self, model_errors_file: str = 'modelrrors.csv'):
        """
        Initialize the error calculator.
        
        Args:
            model_errors_file: Path to the CSV file containing model errors
        """
        self.model_errors = self._load_model_errors(model_errors_file)
    
    def _load_model_errors(self, file_path: str) -> Dict[str, Dict[str, float]]:
        """
        Load model errors from CSV file.
        
        Args:
            file_path: Path to the model errors CSV file
            
        Returns:
            Dictionary mapping property types to their error metrics
        """
        try:
            df = pd.read_csv(file_path)
            errors = {}
            
            for _, row in df.iterrows():
                property_type = row['Model'].lower()
                errors[property_type] = {
                    'model_error': float(row['Error']),
                    'experimental_std': float(row['Experimental Standard Deviation'])
                }
            
            logger.info(f"✅ Loaded model errors for {len(errors)} properties")
            return errors
            
        except Exception as e:
            logger.error(f"❌ Failed to load model errors from {file_path}: {e}")
            return {}
    
    def calculate_error_bounds(self, property_type: str, prediction: float) -> Optional[Dict[str, float]]:
        """
        Calculate error bounds for a property prediction.
        
        Args:
            property_type: Type of property (wvtr, ts, eab, cobb)
            prediction: The predicted value
            
        Returns:
            Dictionary containing error bounds and metrics, or None if failed
        """
        if property_type not in self.model_errors:
            logger.warning(f"⚠️ No error data available for property type: {property_type}")
            return None
        
        error_data = self.model_errors[property_type]
        model_error = error_data['model_error']
        experimental_std = error_data['experimental_std']
        
        # Calculate bounds using model error
        upper_bound = prediction + model_error
        lower_bound = max(0.0, prediction - model_error)  # Ensure non-negative
        
        return {
            'prediction': prediction,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'model_error': model_error,
            'experimental_std': experimental_std
        }
    
    def format_error_results(self, property_type: str, error_bounds: Dict[str, float], 
                           property_name: str, unit: str) -> str:
        """
        Format error results for display.
        
        Args:
            property_type: Type of property
            error_bounds: Error bounds dictionary
            property_name: Human-readable property name
            unit: Property unit
            
        Returns:
            Formatted string for display
        """
        if not error_bounds:
            return f"❌ Error calculation failed for {property_name}"
        
        result = f"\n=== {property_name.upper()} PREDICTION RESULTS WITH UNCERTAINTY ===\n"
        result += f"Predicted {property_name}: {error_bounds['prediction']:.2f} {unit}\n"
        result += f"Confidence Interval: [{error_bounds['lower_bound']:.2f}, {error_bounds['upper_bound']:.2f}] {unit}\n"
        result += f"Model Error (±): {error_bounds['model_error']:.2f} {unit}\n"
        result += f"Experimental Standard Deviation: {error_bounds['experimental_std']:.2f} {unit}"
        
        return result
    
    def get_available_properties(self) -> list:
        """Get list of properties with available error data."""
        return list(self.model_errors.keys()) 