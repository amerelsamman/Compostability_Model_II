"""
Model training module for unified training pipeline.
EXACTLY matches the original train_unified.py functionality.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from typing import List, Dict, Any
import joblib

def create_preprocessing_pipeline(categorical_features: List[str], numerical_features: List[str]):
    """Create preprocessing pipeline - EXACTLY matching original"""
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('num', 'passthrough', numerical_features)
        ],
        remainder='drop'
    )

def train_models(X_train: pd.DataFrame, log_y_values_train: List[pd.Series], 
                categorical_features: List[str], numerical_features: List[str]) -> List[Pipeline]:
    """Train XGBoost models - EXACTLY matching original"""
    preprocessor = create_preprocessing_pipeline(categorical_features, numerical_features)
    
    models = []
    for i, y_train in enumerate(log_y_values_train):
        print(f"Training model {i+1}")
        
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                n_estimators=300,
                max_depth=25,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
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
    """Evaluate model performance and return results dictionary - EXACTLY matching original"""
    print("\n📊 Evaluating model performance...")
    
    results = {}
    
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        # Get predictions
        y_train = log_y_values_train[i]
        y_test = log_y_values_test[i]
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Store results
        model_key = f"model_{i+1}"
        results[model_key] = {
            'target_col': target_col,
            'log_scale': {
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'train_mae': train_mae,
                'test_mae': test_mae
            },
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            }
        }
        
        # Print performance
        print(f"\n=== MODEL {i+1} ({target_col}) PERFORMANCE ===")
        print("Training Set:")
        print(f"  R² Score: {train_r2:.4f}")
        print(f"  Mean Squared Error: {train_mse:.4f}")
        print(f"  Mean Absolute Error: {train_mae:.4f}")
        print(f"  Root Mean Squared Error: {np.sqrt(train_mse):.4f}")
        
        print("Test Set:")
        print(f"  R² Score: {test_r2:.4f}")
        print(f"  Mean Squared Error: {test_mse:.4f}")
        print(f"  Mean Absolute Error: {test_mae:.4f}")
        print(f"  Root Mean Squared Error: {np.sqrt(test_mse):.4f}")
    
    return results

def save_models(models: List[Pipeline], target_cols: List[str], output_dir: str) -> None:
    """Save trained models - EXACTLY matching original"""
    print("\n💾 Saving trained models...")
    
    for i, (model, target_col) in enumerate(zip(models, target_cols)):
        model_path = f"{output_dir}/comprehensive_polymer_model_{target_col}.pkl"
        joblib.dump(model, model_path)
        print(f"✅ Model saved: {model_path}")

def save_feature_importance(models: List[Pipeline], target_cols: List[str], output_dir: str) -> None:
    """Save feature importance to CSV - EXACTLY matching original"""
    print("\n💾 Saving feature importance...")
    
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
        csv_path = f"{output_dir}/feature_importance_{target_col}.csv"
        importance_df.to_csv(csv_path, index=False)
        print(f"✅ Feature importance CSV saved: {csv_path}")
