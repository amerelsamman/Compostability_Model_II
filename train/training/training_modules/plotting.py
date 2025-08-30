"""
Plotting module for unified training pipeline.
EXACTLY matches the original train_unified.py functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from typing import List, Dict, Any
import os

def create_comprehensive_plots(models: List[Pipeline], X_train: pd.DataFrame, X_test: pd.DataFrame,
                              log_y_values_train: List[pd.Series], log_y_values_test: List[pd.Series],
                              results: Dict[str, Any], target_cols: List[str], output_dir: str):
    """Create comprehensive performance plots - EXACTLY matching original"""
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
                r2_scores = [results[f'model_{j+1}']['log_scale']['test_r2'] for j in range(n_models)]
                colors = ['#FF6B6B', '#4ECDC4']
                axes[2, 2].bar(target_cols, r2_scores, color=colors[:n_models], alpha=0.7)
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
            axes[0, 2].set_yticks(range(top_features), [feature_names[i].split('__')[-1] for i in indices[:top_features]])
            axes[0, 2].set_xlabel('Feature Importance')
            axes[0, 2].set_title(f'Top 20 Features - {target_col}')
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
            axes[1, 2].hist(residuals_train, bins=20, alpha=0.7, color='blue')
            axes[1, 2].set_xlabel('Residuals')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Residuals Distribution')
            
            # 7. Model summary
            axes[2, 0].text(0.1, 0.9, f'{target_col} Model:', fontsize=12, fontweight='bold')
            axes[2, 0].text(0.1, 0.8, f'R²: {results[f"model_{i+1}"]["log_scale"]["test_r2"]:.3f}', fontsize=10)
            axes[2, 0].text(0.1, 0.7, f'MAE: {results[f"model_{i+1}"]["log_scale"]["test_mae"]:.3f}', fontsize=10)
            axes[2, 0].text(0.1, 0.6, f'RMSE: {np.sqrt(results[f"model_{i+1}"]["log_scale"]["test_mse"]):.3f}', fontsize=10)
            axes[2, 0].text(0.1, 0.5, f'Training: {len(X_train)}', fontsize=10)
            axes[2, 0].text(0.1, 0.4, f'Test: {len(X_test)}', fontsize=10)
            axes[2, 0].text(0.1, 0.3, f'Features: {X_train.shape[1]}', fontsize=10)
            axes[2, 0].set_title('Model Summary')
            axes[2, 0].axis('off')
            
            # 8. Performance comparison
            r2_scores = [results[f'model_{j+1}']['log_scale']['test_r2'] for j in range(n_models)]
            colors = ['#FF6B6B']
            axes[2, 1].bar(target_cols, r2_scores, color=colors[:n_models], alpha=0.7)
            axes[2, 1].set_ylabel('R² Score (Test Set)')
            axes[2, 1].set_title('Model Performance Comparison')
            axes[2, 1].set_ylim(0, 1)
            
            # 9. Data summary
            axes[2, 2].text(0.1, 0.9, 'Overall Summary:', fontsize=12, fontweight='bold')
            axes[2, 2].text(0.1, 0.8, f'Total Training: {len(X_train)}', fontsize=10)
            axes[2, 2].text(0.1, 0.7, f'Total Test: {len(X_test)}', fontsize=10)
            axes[2, 2].text(0.1, 0.6, f'Features: {X_train.shape[1]}', fontsize=10)
            axes[2, 2].text(0.1, 0.5, f'Properties: {n_models}', fontsize=10)
            axes[2, 2].set_title('Data Summary')
            axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_polymer_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Comprehensive performance plot saved")

def create_last_n_performance_plots(models: List[Pipeline], df: pd.DataFrame, X: pd.DataFrame, 
                                   log_y_values: List[pd.Series], results: Dict[str, Any], 
                                   target_cols: List[str], property_config, args, output_dir: str):
    """Create last N blends performance plots - EXACTLY matching original"""
    # Determine last N values
    last_n_training = args.last_n_training if args.last_n_training is not None else property_config.default_last_n_training
    last_n_testing = args.last_n_testing if args.last_n_testing is not None else property_config.default_last_n_testing
    
    if last_n_training == 0 and last_n_testing == 0:
        print("No last N strategy specified, skipping last N performance plots")
        return
    
    # Get last N data
    if last_n_training > 0:
        last_n_indices = list(range(len(df) - last_n_training, len(df)))
        last_n_X = X.iloc[last_n_indices]
        last_n_log_y_values = [log_y.iloc[last_n_indices] for log_y in log_y_values]
        plot_title = f'Last {last_n_training} Blends Performance'
        filename_prefix = f'last_{last_n_training}_blends_performance'
    else:
        last_n_indices = list(range(len(df) - last_n_testing, len(df)))
        last_n_X = X.iloc[last_n_indices]
        last_n_log_y_values = [log_y.iloc[last_n_indices] for log_y in log_y_values]
        plot_title = f'Last {last_n_testing} Blends Performance'
        filename_prefix = f'last_{last_n_testing}_blends_performance'
    
    # Create plot
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    
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
        
        # Add blend labels (B1, B2, B3, etc.) for log scale
        for j, (actual, pred) in enumerate(zip(last_n_actual_log, last_n_pred_log)):
            blend_label = f'B{j+1}'
            axes[0, i].annotate(blend_label, (actual, pred), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='darkred')
        
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
        
        # Add blend labels (B1, B2, B3, etc.) for original scale
        for j, (actual, pred) in enumerate(zip(last_n_actual, last_n_pred)):
            blend_label = f'B{j+1}'
            axes[1, i].annotate(blend_label, (actual, pred), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold', color='darkblue')
        
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
