#!/usr/bin/env python3
"""
Unified training script for all polymer blend properties.
This maintains IDENTICAL functionality to the original scripts while providing
a single interface for training any property.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import os
import joblib
from typing import List, Tuple, Dict, Any

# Import our configuration
from training_config import get_property_config, get_available_properties

warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments"""
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

def load_and_preprocess_data(input_path: str, property_config, args) -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
    """Load and preprocess data according to property configuration"""
    print(f"Loading featurized polymer blends data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle target columns
    if property_config.is_dual_property:
        target_cols = property_config.target_columns
        print(f"Target columns: {target_cols}")
        
        # Separate features and targets
        smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
        excluded_cols = target_cols + smiles_cols + ['Materials']
        X = df.drop(columns=excluded_cols)
        
        # Get target values
        y_values = []
        for col in target_cols:
            if col in df.columns:
                y_values.append(df[col])
            else:
                # Handle case where column names might be different
                if col == 'max_L' and 'property1' in df.columns:
                    y_values.append(df['property1'])
                elif col == 't0' and 'property2' in df.columns:
                    y_values.append(df['property2'])
                else:
                    raise ValueError(f"Target column {col} not found in data")
        
        # Apply log transformation with appropriate offset
        log_y_values = []
        for i, y in enumerate(y_values):
            if property_config.log_offset > 0:
                log_y = np.log(y + property_config.log_offset)
            else:
                log_y = np.log(y)
            log_y_values.append(log_y)
            
        print(f"Log-transformed target ranges:")
        for i, (col, log_y) in enumerate(zip(target_cols, log_y_values)):
            print(f"  {col}: {log_y.min():.4f} to {log_y.max():.4f}")
            
    else:
        # Single property
        target_col = property_config.target_columns[0]
        print(f"Target column: {target_col}")
        
        # Remove rows where the property value is zero if configured
        if property_config.remove_zero_targets:
            initial_shape = df.shape
            zero_rows = df[df[target_col] == 0].shape[0]
            df = df[df[target_col] != 0].reset_index(drop=True)
            print(f"Removed {zero_rows} rows with property == 0. New shape: {df.shape} (was {initial_shape})")
        
        # Separate features and target
        smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
        excluded_cols = [target_col] + smiles_cols + ['Materials']
        X = df.drop(columns=excluded_cols)
        y = df[target_col]
        
        # Apply log transformation
        if property_config.log_offset > 0:
            y = np.log(y + property_config.log_offset)
        else:
            y = np.log(y)
        
        log_y_values = [y]
        target_cols = [target_col]
        
        print(f"Log-transformed target range: {y.min():.4f} to {y.max():.4f}")
    
    # Handle NaN targets if configured (WVTR/OTR style)
    if property_config.handle_nan_targets:
        print("Handling missing values...")
        # Remove rows with NaN or infinite values in target
        valid_mask = ~(log_y_values[0].isna() | np.isinf(log_y_values[0]))
        df = df[valid_mask]
        X = X[valid_mask]
        for i in range(len(log_y_values)):
            log_y_values[i] = log_y_values[i][valid_mask]
        
        # Also check for NaN values in features and fill them
        X = X.fillna(0)  # Fill NaN values in features with 0
        
        print(f"After cleaning: {len(df)} samples remaining")
        print(f"Target NaN count: {log_y_values[0].isna().sum()}")
        print(f"Target infinite count: {np.isinf(log_y_values[0]).sum()}")
    
    # Identify categorical and numerical features
    categorical_features = []
    numerical_features = []
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'string':
            categorical_features.append(col)
        else:
            numerical_features.append(col)
    
    print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    
    # Handle missing values
    print("\nHandling missing values...")
    for col in categorical_features:
        X[col] = X[col].fillna('Unknown')
        
    for col in numerical_features:
        X[col] = X[col].fillna(0)
    
    return df, X, log_y_values, target_cols

def create_preprocessing_pipeline(categorical_features: List[str], numerical_features: List[str]):
    """Create preprocessing pipeline"""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )

def split_data_with_last_n_strategy(df: pd.DataFrame, X: pd.DataFrame, log_y_values: List[pd.Series], 
                                   property_config, args) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Series], List[pd.Series]]:
    """Split data according to property-specific strategy"""
    
    # Determine last N values
    last_n_training = args.last_n_training if args.last_n_training is not None else property_config.default_last_n_training
    last_n_testing = args.last_n_testing if args.last_n_testing is not None else property_config.default_last_n_testing
    
    print(f"Data splitting strategy:")
    print(f"  Last {last_n_training} blends in training")
    print(f"  Last {last_n_testing} blends in testing")
    
    if property_config.is_dual_property:
        # Dual property splitting (TS, EAB, EOL style)
        if last_n_training > 0:
            last_n_indices = list(range(len(df) - last_n_training, len(df)))
            
            # Remove last N from the main pool for train_test_split
            remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
            X_remaining = X.iloc[remaining_indices]
            log_y1_remaining = log_y_values[0].iloc[remaining_indices]
            log_y2_remaining = log_y_values[1].iloc[remaining_indices]
            
            # Use train_test_split on the remaining data
            X_temp_train, X_temp_test, y1_temp_train, y1_temp_test, y2_temp_train, y2_temp_test, temp_train_indices, temp_test_indices = train_test_split(
                X_remaining, log_y1_remaining, log_y2_remaining, remaining_indices, 
                test_size=0.2, random_state=42, shuffle=True
            )
            
            # Combine: last N always in training, rest split by train_test_split
            train_indices = last_n_indices + temp_train_indices
            test_indices = temp_test_indices
            
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            log_y1_train = log_y_values[0].iloc[train_indices]
            log_y1_test = log_y_values[0].iloc[test_indices]
            log_y2_train = log_y_values[1].iloc[train_indices]
            log_y2_test = log_y_values[1].iloc[test_indices]
            
            # Apply oversampling if configured
            if property_config.oversampling_factor > 0:
                print(f"Applying {property_config.oversampling_factor}x oversampling to last {last_n_training} blends...")
                
                last_n_X = X.iloc[last_n_indices]
                last_n_log_y1 = log_y_values[0].iloc[last_n_indices]
                last_n_log_y2 = log_y_values[1].iloc[last_n_indices]
                
                # Repeat the last N blends oversampling_factor times
                oversampled_X = []
                oversampled_y1 = []
                oversampled_y2 = []
                
                # Add original training data (excluding last N)
                other_train_indices = [i for i in train_indices if i not in last_n_indices]
                oversampled_X.append(X.iloc[other_train_indices])
                oversampled_y1.append(log_y_values[0].iloc[other_train_indices])
                oversampled_y2.append(log_y_values[1].iloc[other_train_indices])
                
                # Add last N blends oversampling_factor times
                for _ in range(property_config.oversampling_factor - 1):
                    oversampled_X.append(last_n_X)
                    oversampled_y1.append(last_n_log_y1)
                    oversampled_y2.append(last_n_log_y2)
                
                # Combine all data
                X_train = pd.concat(oversampled_X, ignore_index=True)
                log_y1_train = pd.concat(oversampled_y1, ignore_index=True)
                log_y2_train = pd.concat(oversampled_y2, ignore_index=True)
                
                log_y_values_train = [log_y1_train, log_y2_train]
                log_y_values_test = [log_y1_test, log_y2_test]
            else:
                log_y_values_train = [log_y1_train, log_y2_train]
                log_y_values_test = [log_y1_test, log_y2_test]
                
        else:
            # No special last N strategy, use standard split
            X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
                X, log_y_values[0], log_y_values[1], test_size=0.2, random_state=42, shuffle=True
            )
            log_y_values_train = [y1_train, y2_train]
            log_y_values_test = [y1_test, y2_test]
            
    else:
        # Single property splitting
        if last_n_training > 0:
            # Last N in training (Cobb, WVTR style)
            last_n_indices = list(range(len(df) - last_n_training, len(df)))
            
            # Create masks for train/test split
            train_mask = np.ones(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
            
            # Set last N blends to training
            train_mask[last_n_indices] = True
            
            # Use train_test_split on the remaining data (excluding last N)
            remaining_indices = list(range(len(df) - last_n_training))
            if len(remaining_indices) > 0:
                # Split remaining data with 80/20 ratio
                train_remaining, test_indices = train_test_split(
                    remaining_indices, 
                    test_size=0.2, 
                    random_state=42, 
                    shuffle=True
                )
                
                # Combine: last N always in training, rest split by train_test_split
                train_indices = last_n_indices + train_remaining
                test_indices = test_indices
            else:
                train_indices = last_n_indices
                test_indices = []
                
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = log_y_values[0].iloc[train_indices]
            y_test = log_y_values[0].iloc[test_indices]
            
            log_y_values_train = [y_train]
            log_y_values_test = [y_test]
            
        elif last_n_testing > 0:
            # Last N in testing (Adhesion, OTR style)
            last_n_indices = list(range(len(df) - last_n_testing, len(df)))
            
            # Create masks for train/test split
            train_mask = np.ones(len(df), dtype=bool)
            test_mask = np.zeros(len(df), dtype=bool)
            
            # Set last N blends to testing
            test_mask[last_n_indices] = True
            train_mask[last_n_indices] = False
            
            # Use train_test_split on the remaining data (excluding last N)
            remaining_indices = list(range(len(df) - last_n_testing))
            if len(remaining_indices) > 0:
                # Split remaining data with 80/20 ratio
                train_indices, test_remaining = train_test_split(
                    remaining_indices, 
                    test_size=0.2, 
                    random_state=42, 
                    shuffle=True
                )
                
                # Combine: last N always in testing, rest split by train_test_split
                train_indices = train_indices
                test_indices = last_n_indices + test_remaining
            else:
                train_indices = []
                test_indices = last_n_indices
                
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = log_y_values[0].iloc[train_indices]
            y_test = log_y_values[0].iloc[test_indices]
            
            log_y_values_train = [y_train]
            log_y_values_test = [y_test]
            
        else:
            # Standard split
            X_train, X_test, y_train, y_test = train_test_split(
                X, log_y_values[0], test_size=0.2, random_state=42, shuffle=True
            )
            log_y_values_train = [y_train]
            log_y_values_test = [y_test]
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    return X_train, X_test, log_y_values_train, log_y_values_test

def train_models(X_train: pd.DataFrame, log_y_values_train: List[pd.Series], 
                categorical_features: List[str], numerical_features: List[str]) -> List[Pipeline]:
    """Train XGBoost models"""
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    
    models = []
    for i, y_train in enumerate(log_y_values_train):
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=120,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.2,
                reg_lambda=2.0,
                min_child_weight=1,
                gamma=0.0,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        model.fit(X_train, y_train)
        models.append(model)
        print(f"Trained model {i+1}")
    
    return models

def evaluate_models(models: List[Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame,
                   log_y_values_train: List[pd.Series], log_y_values_test: List[pd.Series],
                   target_cols: List[str]) -> Dict[str, Any]:
    """Evaluate model performance"""
    results = {}
    
    for i, (model, y_train, y_test, target_col) in enumerate(zip(models, log_y_values_train, log_y_values_test, target_cols)):
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics (log scale)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Inverse transform to original scale
        orig_y_train = np.exp(y_train)
        orig_y_test = np.exp(y_test)
        orig_pred_train = np.exp(y_pred_train)
        orig_pred_test = np.exp(y_pred_test)
        
        # Calculate original scale metrics
        orig_train_r2 = r2_score(orig_y_train, orig_pred_train)
        orig_test_r2 = r2_score(orig_y_test, orig_pred_test)
        orig_train_mae = mean_absolute_error(orig_y_train, orig_pred_train)
        orig_test_mae = mean_absolute_error(orig_y_test, orig_pred_test)
        
        results[f'model_{i+1}'] = {
            'target_col': target_col,
            'log_scale': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'original_scale': {
                'train_r2': orig_train_r2,
                'test_r2': orig_test_r2,
                'train_mae': orig_train_mae,
                'test_mae': orig_test_mae
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test,
                'orig_train': orig_pred_train,
                'orig_test': orig_pred_test
            }
        }
        
        print(f"\n=== MODEL {i+1} ({target_col}) PERFORMANCE ===")
        print(f"Training Set:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  Mean Squared Error: {train_mse:.4f}")
        print(f"  Mean Absolute Error: {train_mae:.4f}")
        print(f"  Root Mean Squared Error: {np.sqrt(train_mse):.4f}")
        print(f"\nTest Set:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  Mean Squared Error: {test_mse:.4f}")
        print(f"  Mean Absolute Error: {test_mae:.4f}")
        print(f"  Root Mean Squared Error: {np.sqrt(test_mse):.4f}")
    
    return results

def create_comprehensive_plots(models: List[Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame,
                              log_y_values_train: List[pd.Series], log_y_values_test: List[pd.Series],
                              results: Dict[str, Any], target_cols: List[str], output_dir: str):
    """Create comprehensive performance plots matching the old training scripts"""
    
    n_models = len(models)
    
    # For dual properties, use 3x4 grid (12 subplots)
    # For single properties, use 3x3 grid (9 subplots)
    if n_models == 2:
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        plt.suptitle('Dual Property Model Results', fontsize=16)
    else:
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        plt.suptitle('Single Property Model Results', fontsize=16)
    
    # Ensure axes is always 2D
    if n_models == 1:
        axes = axes.reshape(3, 3)
    
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        # Get predictions
        y_train = log_y_values_train[i]
        y_test = log_y_values_test[i]
        y_pred_train = results[f'model_{i+1}']['predictions']['train']
        y_pred_test = results[f'model_{i+1}']['predictions']['test']
        
        # Calculate residuals
        residuals_train = y_train - y_pred_train
        residuals_test = y_test - y_pred_test
        
        # Get feature importance
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importances = model.named_steps['regressor'].feature_importances_
        
        if n_models == 2:
            # Dual property layout (3x4 grid)
            if i == 0:  # First property
                # 1. Actual vs Predicted
                axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, label='Training', color='blue')
                axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, label='Test', color='red')
                axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[0, 0].set_xlabel(f'Actual Log({target_col})')
                axes[0, 0].set_ylabel(f'Predicted Log({target_col})')
                axes[0, 0].set_title(f'{target_col} Prediction')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # 2. Residuals
                axes[0, 1].scatter(y_pred_train, residuals_train, alpha=0.6, label='Training', color='blue')
                axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', color='red')
                axes[0, 1].axhline(y=0, color='k', linestyle='--')
                axes[0, 1].set_xlabel(f'Predicted Log({target_col})')
                axes[0, 1].set_ylabel('Residuals')
                axes[0, 1].set_title(f'{target_col} Residuals')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # 3. Feature Importance
                top_features = 15
                indices = np.argsort(importances)[::-1]
                axes[0, 2].barh(range(top_features), importances[indices[:top_features]])
                axes[0, 2].set_yticks(range(top_features), [feature_names[i].split('__')[-1] for i in indices[:top_features]])
                axes[0, 2].set_xlabel('Feature Importance')
                axes[0, 2].set_title(f'Top 15 Features - {target_col}')
                axes[0, 2].invert_yaxis()
                
                # 4. Data Distribution
                axes[0, 3].hist(y_train, bins=20, alpha=0.7, label='Training', color='blue')
                axes[0, 3].hist(y_test, bins=20, alpha=0.7, label='Test', color='red')
                axes[0, 3].set_xlabel(f'Log({target_col})')
                axes[0, 3].set_ylabel('Frequency')
                axes[0, 3].set_title(f'{target_col} Distribution')
                axes[0, 3].legend()
                
            else:  # Second property
                # 5. Actual vs Predicted
                axes[1, 0].scatter(y_train, y_pred_train, alpha=0.6, label='Training', color='blue')
                axes[1, 0].scatter(y_test, y_pred_test, alpha=0.6, label='Test', color='red')
                axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                axes[1, 0].set_xlabel(f'Actual Log({target_col})')
                axes[1, 0].set_ylabel(f'Predicted Log({target_col})')
                axes[1, 0].set_title(f'{target_col} Prediction')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 6. Residuals
                axes[1, 1].scatter(y_pred_train, residuals_train, alpha=0.6, label='Training', color='blue')
                axes[1, 1].scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', color='red')
                axes[1, 1].axhline(y=0, color='k', linestyle='--')
                axes[1, 1].set_xlabel(f'Predicted Log({target_col})')
                axes[1, 1].set_ylabel('Residuals')
                axes[1, 1].set_title(f'{target_col} Residuals')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # 7. Feature Importance
                top_features = 15
                indices = np.argsort(importances)[::-1]
                axes[1, 2].barh(range(top_features), importances[indices[:top_features]])
                axes[1, 2].set_yticks(range(top_features), [feature_names[i].split('__')[-1] for i in indices[:top_features]])
                axes[1, 2].set_xlabel('Feature Importance')
                axes[1, 2].set_title(f'Top 15 Features - {target_col}')
                axes[1, 2].invert_yaxis()
                
                # 8. Data Distribution
                axes[1, 3].hist(y_train, bins=20, alpha=0.7, label='Training', color='blue')
                axes[1, 3].hist(y_test, bins=20, alpha=0.7, label='Test', color='red')
                axes[1, 3].set_xlabel(f'Log({target_col})')
                axes[1, 3].set_ylabel('Frequency')
                axes[1, 3].set_title(f'{target_col} Distribution')
                axes[1, 3].legend()
            
            # Bottom row for both properties
            if i == 0:  # First property model summary
                axes[2, 0].text(0.1, 0.9, f'{target_col} Model:', fontsize=12, fontweight='bold')
                axes[2, 0].text(0.1, 0.8, f'R²: {results[f"model_{i+1}"]["log_scale"]["test_r2"]:.3f}', fontsize=10)
                axes[2, 0].text(0.1, 0.7, f'MAE: {results[f"model_{i+1}"]["log_scale"]["test_mae"]:.3f}', fontsize=10)
                axes[2, 0].text(0.1, 0.6, f'RMSE: {np.sqrt(results[f"model_{i+1}"]["log_scale"]["test_mse"]):.3f}', fontsize=10)
                axes[2, 0].text(0.1, 0.5, f'Training: {len(X_train)}', fontsize=10)
                axes[2, 0].text(0.1, 0.4, f'Test: {len(X_test)}', fontsize=10)
                axes[2, 0].text(0.1, 0.3, f'Features: {X_train.shape[1]}', fontsize=10)
                axes[2, 0].set_title('Model Summary')
                axes[2, 0].axis('off')
            else:  # Second property model summary
                axes[2, 1].text(0.1, 0.9, f'{target_col} Model:', fontsize=12, fontweight='bold')
                axes[2, 1].text(0.1, 0.8, f'R²: {results[f"model_{i+1}"]["log_scale"]["test_r2"]:.3f}', fontsize=10)
                axes[2, 1].text(0.1, 0.7, f'MAE: {results[f"model_{i+1}"]["log_scale"]["test_mae"]:.3f}', fontsize=10)
                axes[2, 1].text(0.1, 0.6, f'RMSE: {np.sqrt(results[f"model_{i+1}"]["log_scale"]["test_mse"]):.3f}', fontsize=10)
                axes[2, 1].text(0.1, 0.5, f'Training: {len(X_train)}', fontsize=10)
                axes[2, 1].text(0.1, 0.4, f'Test: {len(X_test)}', fontsize=10)
                axes[2, 1].text(0.1, 0.3, f'Features: {X_train.shape[1]}', fontsize=10)
                axes[2, 1].set_title('Model Summary')
                axes[2, 1].axis('off')
                
                # Performance comparison
                axes[2, 2].bar(target_cols, [results[f'model_{j+1}']['log_scale']['test_r2'] for j in range(n_models)], 
                               color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
                axes[2, 2].set_ylabel('R² Score (Test Set)')
                axes[2, 2].set_title('Model Performance Comparison')
                axes[2, 2].set_ylim(0, 1)
                
                # Data summary
                axes[2, 3].text(0.1, 0.9, 'Overall Summary:', fontsize=12, fontweight='bold')
                axes[2, 3].text(0.1, 0.8, f'Total Training: {len(X_train)}', fontsize=10)
                axes[2, 3].text(0.1, 0.7, f'Total Test: {len(X_test)}', fontsize=10)
                axes[2, 3].text(0.1, 0.6, f'Features: {X_train.shape[1]}', fontsize=10)
                axes[2, 3].text(0.1, 0.5, f'Properties: {n_models}', fontsize=10)
                axes[2, 3].set_title('Data Summary')
                axes[2, 3].axis('off')
        
        else:
            # Single property layout (3x3 grid)
            # 1. Actual vs Predicted
            axes[0, 0].scatter(y_train, y_pred_train, alpha=0.6, label='Training', color='blue')
            axes[0, 0].scatter(y_test, y_pred_test, alpha=0.6, label='Test', color='red')
            axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[0, 0].set_xlabel(f'Actual Log({target_col})')
            axes[0, 0].set_ylabel(f'Predicted Log({target_col})')
            axes[0, 0].set_title(f'{target_col} Prediction')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Residuals
            axes[0, 1].scatter(y_pred_train, residuals_train, alpha=0.6, label='Training', color='blue')
            axes[0, 1].scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', color='red')
            axes[0, 1].axhline(y=0, color='k', linestyle='--')
            axes[0, 1].set_xlabel(f'Predicted Log({target_col})')
            axes[0, 1].set_ylabel('Residuals')
            axes[0, 1].set_title(f'{target_col} Residuals')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Feature Importance
            top_features = 20
            indices = np.argsort(importances)[::-1]
            axes[0, 2].barh(range(top_features), importances[indices[:top_features]])
            axes[0, 2].set_yticks(range(top_features), [feature_names[i] for i in indices[:top_features]])
            axes[0, 2].set_xlabel('Feature Importance')
            axes[0, 2].set_title(f'Top {top_features} Feature Importances')
            axes[0, 2].invert_yaxis()
            
            # 4. Data Distribution
            axes[1, 0].hist(y_train, bins=20, alpha=0.7, label='Training', color='blue')
            axes[1, 0].hist(y_test, bins=20, alpha=0.7, label='Test', color='red')
            axes[1, 0].set_xlabel(f'Log({target_col})')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title(f'{target_col} Distribution')
            axes[1, 0].legend()
            
            # 5. Training vs Test performance
            r2_train = results[f'model_{i+1}']['log_scale']['train_r2']
            r2_test = results[f'model_{i+1}']['log_scale']['test_r2']
            axes[1, 1].bar(['Training', 'Test'], [r2_train, r2_test], color=['blue', 'red'])
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Training vs Test Performance')
            
            # 6. Residuals distribution
            axes[1, 2].hist(residuals_train, bins=30, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 2].set_xlabel('Residuals')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Residuals Distribution')
            
            # 7. Feature importance by category
            axes[2, 0].text(0.1, 0.9, 'Feature Categories:', fontsize=12, fontweight='bold')
            axes[2, 0].text(0.1, 0.8, '• Polymer Grades', fontsize=10)
            axes[2, 0].text(0.1, 0.7, '• Volume Fractions', fontsize=10)
            axes[2, 0].text(0.1, 0.6, '• SP Descriptors', fontsize=10)
            axes[2, 0].text(0.1, 0.5, '• Ring Systems', fontsize=10)
            axes[2, 0].text(0.1, 0.4, '• Carbon Types', fontsize=10)
            axes[2, 0].text(0.1, 0.3, '• Functional Groups', fontsize=10)
            axes[2, 0].set_title('Feature Categories')
            axes[2, 0].axis('off')
            
            # 8. Model complexity
            n_trees = model.named_steps['regressor'].n_estimators
            max_depth = model.named_steps['regressor'].max_depth
            learning_rate = model.named_steps['regressor'].learning_rate
            
            complexity_metrics = ['Trees', 'Max Depth', 'Learning Rate']
            complexity_values = [n_trees, max_depth, learning_rate]
            
            axes[2, 1].bar(complexity_metrics, complexity_values, color=['green', 'orange', 'purple'])
            axes[2, 1].set_title('Model Complexity')
            axes[2, 1].set_ylabel('Value')
            
            # 9. Data summary
            axes[2, 2].text(0.1, 0.9, f'R² Score: {r2_test:.3f}', fontsize=12)
            axes[2, 2].text(0.1, 0.8, f'MAE: {results[f"model_{i+1}"]["log_scale"]["test_mae"]:.3f}', fontsize=12)
            axes[2, 2].text(0.1, 0.7, f'RMSE: {np.sqrt(results[f"model_{i+1}"]["log_scale"]["test_mse"]):.3f}', fontsize=12)
            axes[2, 2].text(0.1, 0.6, f'Training: {len(X_train)}', fontsize=12)
            axes[2, 2].text(0.1, 0.5, f'Test: {len(X_test)}', fontsize=12)
            axes[2, 2].text(0.1, 0.4, f'Features: {X_train.shape[1]}', fontsize=12)
            axes[2, 2].text(0.1, 0.3, f'Target: {target_col}', fontsize=12)
            axes[2, 2].set_title('Model Summary')
            axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_polymer_model_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive performance plot saved")

def create_last_n_performance_plots(models: List[Pipeline], df: pd.DataFrame, X: pd.DataFrame,
                                   log_y_values: List[pd.Series], results: Dict[str, Any],
                                   target_cols: List[str], property_config, args, output_dir: str):
    """Create last N blends performance plots"""
    
    # Determine which blends are in the "last N" category
    last_n_training = args.last_n_training if args.last_n_training is not None else property_config.default_last_n_training
    last_n_testing = args.last_n_testing if args.last_n_testing is not None else property_config.default_last_n_testing
    
    if last_n_training > 0:
        # Last N in training
        last_n_indices = list(range(len(df) - last_n_training, len(df)))
        last_n_X = X.iloc[last_n_indices]
        last_n_log_y_values = [y.iloc[last_n_indices] for y in log_y_values]
        plot_title = f"Last {last_n_training} Blends Performance (Training)"
        filename_prefix = f"last_{last_n_training}_blends_performance"
        
    elif last_n_testing > 0:
        # Last N in testing
        last_n_indices = list(range(len(df) - last_n_testing, len(df)))
        last_n_X = X.iloc[last_n_indices]
        last_n_log_y_values = [y.iloc[last_n_indices] for y in log_y_values]
        plot_title = f"Last {last_n_testing} Blends Performance (Testing)"
        filename_prefix = f"last_{last_n_testing}_blends_performance"
        
    else:
        # No special last N strategy
        return
    
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 12))
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        # Get last N predictions
        last_n_pred_log = model.predict(last_n_X)
        last_n_actual_log = last_n_log_y_values[i]
        
        # Calculate metrics for last N
        last_n_mae_log = mean_absolute_error(last_n_actual_log, last_n_pred_log)
        last_n_r2_log = r2_score(last_n_actual_log, last_n_pred_log)
        
        # Original scale
        last_n_actual = np.exp(last_n_actual_log)
        last_n_pred = np.exp(last_n_pred_log)
        last_n_mae = mean_absolute_error(last_n_actual, last_n_pred)
        last_n_r2 = r2_score(last_n_actual, last_n_pred)
        
        # Plot 1: Log-transformed predictions
        axes[0, i].scatter(last_n_actual_log, last_n_pred_log, color='red', s=100, alpha=0.7)
        axes[0, i].plot([last_n_actual_log.min(), last_n_actual_log.max()], 
                        [last_n_actual_log.min(), last_n_actual_log.max()], 'k--', lw=2)
        axes[0, i].set_xlabel(f'Actual Log({target_col})')
        axes[0, i].set_ylabel(f'Predicted Log({target_col})')
        axes[0, i].set_title(f'{target_col} - Log Scale')
        
        # Add metrics as text box
        metrics_text = f'MAE: {last_n_mae_log:.3f}\nR²: {last_n_r2_log:.3f}'
        axes[0, i].text(0.05, 0.95, metrics_text, transform=axes[0, i].transAxes,
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Original scale predictions
        axes[1, i].scatter(last_n_actual, last_n_pred, color='blue', s=100, alpha=0.7)
        axes[1, i].plot([last_n_actual.min(), last_n_actual.max()], 
                        [last_n_actual.min(), last_n_actual.max()], 'r--', lw=2)
        axes[1, i].set_xlabel(f'Actual {target_col} (Original Scale)')
        axes[1, i].set_ylabel(f'Predicted {target_col} (Original Scale)')
        axes[1, i].set_title(f'{target_col} - Original Scale')
        
        # Add metrics as text box
        metrics_text = f'MAE: {last_n_mae:.3f}\nR²: {last_n_r2:.3f}'
        axes[1, i].text(0.05, 0.95, metrics_text, transform=axes[1, i].transAxes,
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename_prefix}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Last N blends performance plot saved")

def save_feature_importance_csv(models: List[Pipeline], categorical_features: List[str], 
                               numerical_features: List[str], target_cols: List[str], output_dir: str):
    """Save feature importance to CSV"""
    
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        # Get feature names
        feature_names = []
        if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        else:
            # Fallback: create generic feature names
            feature_names = [f"feature_{j}" for j in range(len(model.named_steps['regressor'].feature_importances_))]
        
        # Get feature importances
        importances = model.named_steps['regressor'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, f'feature_importance_{target_col}.csv')
        importance_df.to_csv(csv_path, index=False)
        print(f"✅ Feature importance CSV saved: {csv_path}")

def save_models(models: List[Pipeline], target_cols: List[str], output_dir: str):
    """Save trained models"""
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        model_path = os.path.join(output_dir, f'comprehensive_polymer_model_{target_col}.pkl')
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")

def main():
    """Main training function"""
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
    
    # Create plots
    print("\n📈 Creating performance plots...")
    create_comprehensive_plots(models, X_train, X_test, log_y_values_train, log_y_values_test, 
                              results, target_cols, args.output)
    
    create_last_n_performance_plots(models, df, X, log_y_values, results, target_cols, 
                                   property_config, args, args.output)
    
    # Save feature importance
    print("\n💾 Saving feature importance...")
    save_feature_importance_csv(models, categorical_features, numerical_features, target_cols, args.output)
    
    # Save models
    print("\n💾 Saving trained models...")
    save_models(models, target_cols, args.output)
    
    print(f"\n🎉 {property_config.name} training completed successfully!")
    print(f"📁 Check the {args.output} directory for output files")

if __name__ == "__main__":
    main()
