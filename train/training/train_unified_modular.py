#!/usr/bin/env python3
"""
Modular unified training script for all polymer blend properties.
This maintains IDENTICAL functionality to the original train_unified.py while providing
a modular structure for maintainability.
"""

import pandas as pd
import numpy as np
import warnings
import argparse
import os
from typing import List, Tuple

# Import our modular components
from training_modules import (
    get_property_config, get_available_properties,
    load_and_preprocess_data, split_data_with_last_n_strategy,
    create_preprocessing_pipeline, train_models, evaluate_models,
    save_models, save_feature_importance,
    create_comprehensive_plots, create_last_n_performance_plots
)

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments - EXACTLY matching original"""
    parser = argparse.ArgumentParser(description='Train XGBoost model for any polymer blend property')
    parser.add_argument('--property', type=str, required=True, 
                       choices=get_available_properties(),
                       help='Property to train (ts, cobb, wvtr, otr, adhesion, eab, eol)')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output directory path')
    parser.add_argument('--last_n_training', type=int, default=None,
                       help='Number of last N blends to put in training (overrides default)')
    parser.add_argument('--last_n_testing', type=int, default=None,
                       help='Number of last N blends to put in testing (overrides default)')
    
    args = parser.parse_args()
    return args

def main():
    """Main training function - EXACTLY matching original"""
    args = parse_arguments()
    
    # Get property configuration
    property_config = get_property_config(args.property)
    print(f"\n🚀 Training {property_config.name} model...")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess data
    df, X, log_y_values, target_cols = load_and_preprocess_data(args.input, property_config, args)
    
    # Split data according to property strategy
    X_train, X_test, log_y_values_train, log_y_values_test = split_data_with_last_n_strategy(
        df, X, log_y_values, property_config, args
    )
    
    # Train models
    print("\n🎯 Training XGBoost models...")
    
    # Identify categorical and numerical features
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    models = train_models(X_train, log_y_values_train, categorical_features, numerical_features)
    
    # Evaluate models
    print("\n📊 Evaluating model performance...")
    results = evaluate_models(models, X_train, X_test, log_y_values_train, log_y_values_test, target_cols)
    
    # Create performance plots
    print("\n📈 Creating performance plots...")
    create_comprehensive_plots(models, X_train, X_test, log_y_values_train, log_y_values_test, 
                             results, target_cols, args.output)
    
    create_last_n_performance_plots(models, df, X, log_y_values, results, 
                                  target_cols, property_config, args, args.output)
    
    # Save feature importance
    save_feature_importance(models, target_cols, args.output)
    
    # Save trained models
    save_models(models, target_cols, args.output)
    
    print(f"\n🎉 {property_config.name} training completed successfully!")
    print(f"📁 Check the {args.output} directory for output files")

if __name__ == "__main__":
    main()
