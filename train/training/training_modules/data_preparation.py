"""
Data preparation module for unified training pipeline.
EXACTLY matches the original train_unified.py functionality.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Tuple

def load_and_preprocess_data(input_path: str, property_config, args) -> Tuple[pd.DataFrame, pd.DataFrame, List, List]:
    """Load and preprocess data according to property configuration - EXACTLY matching original"""
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

def split_data_with_last_n_strategy(df: pd.DataFrame, X: pd.DataFrame, log_y_values: List[pd.Series], 
                                   property_config, args) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Series], List[pd.Series]]:
    """Split data according to property-specific strategy - EXACTLY matching original"""
    
    # Determine last N values
    last_n_training = args.last_n_training if args.last_n_training is not None else property_config.default_last_n_training
    last_n_testing = args.last_n_testing if args.last_n_testing is not None else property_config.default_last_n_testing
    
    # Override oversampling factor if provided via command line
    oversampling_factor = args.oversampling_factor if args.oversampling_factor is not None else property_config.oversampling_factor
    
    print(f"Data splitting strategy:")
    print(f"  Last {last_n_training} blends in training")
    print(f"  Last {last_n_testing} blends in testing")
    
    if property_config.is_dual_property:
        # Dual property splitting (TS, EAB, EOL style)
        if last_n_training > 0:
            last_n_indices = list(range(len(df) - last_n_training, len(df)))
            
            # Handle case where both last_n_training and last_n_testing are specified
            if last_n_testing > 0 and last_n_testing < last_n_training:
                # Split the last N blends: some to training, some to testing
                last_n_training_only = last_n_training - last_n_testing
                last_n_testing_only = last_n_testing
                
                # Indices for blends that go to training only
                last_n_training_indices = list(range(len(df) - last_n_training, len(df) - last_n_testing))
                # Indices for blends that go to testing
                last_n_testing_indices = list(range(len(df) - last_n_testing, len(df)))
                
                print(f"  → Last {last_n_training_only} blends in training only")
                print(f"  → Last {last_n_testing_only} blends in testing")
                
                # Remove all last N from the main pool for train_test_split
                remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
                X_remaining = X.iloc[remaining_indices]
                log_y1_remaining = log_y_values[0].iloc[remaining_indices]
                log_y2_remaining = log_y_values[1].iloc[remaining_indices]
                
                # Use train_test_split on the remaining data
                X_temp_train, X_temp_test, y1_temp_train, y1_temp_test, y2_temp_train, y2_temp_test, temp_train_indices, temp_test_indices = train_test_split(
                    X_remaining, log_y1_remaining, log_y2_remaining, remaining_indices, 
                    test_size=0.2, random_state=42, shuffle=True
                )
                
                # Combine: last N training + temp training, temp testing + last N testing
                train_indices = last_n_training_indices + temp_train_indices
                test_indices = temp_test_indices + last_n_testing_indices
                
            else:
                # Original logic: last N all go to training
                # Remove last N from the main pool for train_test_split
                remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
                X_remaining = X.iloc[remaining_indices]
                log_y1_remaining = log_y_values[0].iloc[remaining_indices]
                log_y2_remaining = log_y_values[1].iloc[remaining_indices]
                
                # Use train_test_split on the remaining data
                X_temp_train, X_temp_test, y1_temp_train, y1_test, y2_temp_train, y2_temp_test, temp_train_indices, temp_test_indices = train_test_split(
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
            if oversampling_factor > 0:
                # Determine which blends to oversample (only the ones actually in training)
                if last_n_testing > 0 and last_n_testing < last_n_training:
                    # Only oversample the blends that are in training (not the ones moved to testing)
                    oversample_indices = last_n_training_indices
                    oversample_count = last_n_training - last_n_testing
                else:
                    # Oversample all last N blends
                    oversample_indices = last_n_indices
                    oversample_count = last_n_training
                
                print(f"Applying {oversampling_factor}x oversampling to last {oversample_count} blends in training...")
                
                oversample_X = X.iloc[oversample_indices]
                oversample_log_y1 = log_y_values[0].iloc[oversample_indices]
                oversample_log_y2 = log_y_values[1].iloc[oversample_indices]
                
                # Repeat the oversample blends oversampling_factor times
                oversampled_X = []
                oversampled_y1 = []
                oversampled_y2 = []
                
                # Add original training data (excluding oversampled blends)
                other_train_indices = [i for i in train_indices if i not in oversample_indices]
                oversampled_X.append(X.iloc[other_train_indices])
                oversampled_y1.append(log_y_values[0].iloc[other_train_indices])
                oversampled_y2.append(log_y_values[1].iloc[other_train_indices])
                
                # Add oversample blends oversampling_factor times
                for _ in range(oversampling_factor - 1):
                    oversampled_X.append(oversample_X)
                    oversampled_y1.append(oversample_log_y1)
                    oversampled_y2.append(oversample_log_y2)
                
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
            
            # Handle case where both last_n_training and last_n_testing are specified
            if last_n_testing > 0 and last_n_testing < last_n_training:
                # Split the last N blends: some to training, some to testing
                last_n_training_only = last_n_training - last_n_testing
                last_n_testing_only = last_n_testing
                
                # Indices for blends that go to training only
                last_n_training_indices = list(range(len(df) - last_n_training, len(df) - last_n_testing))
                # Indices for blends that go to testing
                last_n_testing_indices = list(range(len(df) - last_n_testing, len(df)))
                
                print(f"  → Last {last_n_training_only} blends in training only")
                print(f"  → Last {last_n_testing_only} blends in testing")
                
                # Remove all last N from the main pool for train_test_split
                remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
                X_remaining = X.iloc[remaining_indices]
                y_remaining = log_y_values[0].iloc[remaining_indices]
                
                # Use train_test_split on the remaining data
                X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
                    X_remaining, y_remaining, test_size=0.2, random_state=42, shuffle=True
                )
                
                # Combine: last N training + temp training, temp testing + last N testing
                train_indices = last_n_training_indices + list(X_temp_train.index)
                test_indices = list(X_temp_test.index) + last_n_testing_indices
                
            else:
                # Original logic: last N all go to training
                # Remove last N from the main pool for train_test_split
                remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
                X_remaining = X.iloc[remaining_indices]
                y_remaining = log_y_values[0].iloc[remaining_indices]
                
                # Use train_test_split on the remaining data
                X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
                    X_remaining, y_remaining, test_size=0.2, random_state=42, shuffle=True
                )
                
                # Combine: last N always in training, rest split by train_test_split
                train_indices = last_n_indices + list(X_temp_train.index)
                test_indices = list(X_temp_test.index)
            
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]
            y_train = log_y_values[0].iloc[train_indices]
            y_test = log_y_values[0].iloc[test_indices]
            
            # Apply oversampling if configured
            if oversampling_factor > 0:
                # Determine which blends to oversample (only the ones actually in training)
                if last_n_testing > 0 and last_n_testing < last_n_training:
                    # Only oversample the blends that are in training (not the ones moved to testing)
                    oversample_indices = last_n_training_indices
                    oversample_count = last_n_training - last_n_testing
                else:
                    # Oversample all last N blends
                    oversample_indices = last_n_indices
                    oversample_count = last_n_training
                
                print(f"Applying {oversampling_factor}x oversampling to last {oversample_count} blends in training...")
                
                oversample_X = X.iloc[oversample_indices]
                oversample_y = log_y_values[0].iloc[oversample_indices]
                
                # Repeat the oversample blends oversampling_factor times
                oversampled_X = []
                oversampled_y = []
                
                # Add original training data (excluding oversampled blends)
                other_train_indices = [i for i in train_indices if i not in oversample_indices]
                oversampled_X.append(X.iloc[other_train_indices])
                oversampled_y.append(log_y_values[0].iloc[other_train_indices])
                
                # Add oversample blends oversampling_factor times
                for _ in range(oversampling_factor - 1):
                    oversampled_X.append(oversample_X)
                    oversampled_y.append(oversample_y)
                
                # Combine all data
                X_train = pd.concat(oversampled_X, ignore_index=True)
                y_train = pd.concat(oversampled_y, ignore_index=True)
            
            log_y_values_train = [y_train]
            log_y_values_test = [y_test]
            
        elif last_n_testing > 0:
            # Last N in testing (Adhesion, OTR style)
            last_n_indices = list(range(len(df) - last_n_testing, len(df)))
            
            # Remove last N from the main pool for train_test_split
            remaining_indices = [i for i in range(len(df)) if i not in last_n_indices]
            X_remaining = X.iloc[remaining_indices]
            y_remaining = log_y_values[0].iloc[remaining_indices]
            
            # Use train_test_split on the remaining data
            X_temp_train, X_temp_test, y_temp_train, y_temp_test = train_test_split(
                X_remaining, y_remaining, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Combine: last N always in testing, rest split by train_test_split
            train_indices = list(X_temp_train.index)
            test_indices = list(X_temp_test.index) + last_n_indices
            
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
