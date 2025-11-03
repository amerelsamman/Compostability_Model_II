"""
Visualization Module
Functions for creating optimization and performance plots
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def create_optimization_plots(results_df: pd.DataFrame, property_name: str, output_dir: str = "."):
    """Create comprehensive optimization visualization plots."""
    print("📈 Creating optimization visualization plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'text.color': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': True,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9
    })
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Blend Optimization Results - {property_name.upper()}', fontsize=16, fontweight='bold')
    
    # 1. MAE vs Iterations
    axes[0, 0].plot(results_df['iteration'], results_df['mae'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('MAE')
    axes[0, 0].set_title('Optimization Progress')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add improvement annotation
    initial_mae = results_df['mae'].iloc[0]
    final_mae = results_df['mae'].iloc[-1]
    improvement = initial_mae - final_mae
    axes[0, 0].annotate(f'Improvement: {improvement:.3f}', 
                       xy=(len(results_df), final_mae), 
                       xytext=(len(results_df)*0.7, initial_mae*0.8),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')
    
    # 2. Improvement per Iteration
    axes[0, 1].plot(results_df['iteration'][1:], results_df['improvement'][1:], 'g-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('MAE Improvement')
    axes[0, 1].set_title('Improvement per Iteration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate Distribution (if available)
    if 'blend_lrs' in results_df.columns:
        # Extract learning rates from the last iteration
        last_lrs = results_df['blend_lrs'].iloc[-1]
        if isinstance(last_lrs, str):
            # Parse the string representation of the array
            lr_values = eval(last_lrs)
        else:
            lr_values = last_lrs
        
        # Ensure lr_values is not None and is iterable
        if lr_values is None or len(lr_values) == 0:
            lr_values = [0.001]  # Default fallback
        
        axes[0, 2].hist(lr_values, bins=10, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Learning Rate')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Learning Rate Distribution (Final)')
        axes[0, 2].grid(True, alpha=0.3)
    else:
        # Fallback: show iteration count
        axes[0, 2].bar(['Total Iterations'], [len(results_df)], alpha=0.7, color='blue')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Optimization Summary')
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. MAE Distribution
    axes[1, 0].hist(results_df['mae'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(initial_mae, color='red', linestyle='--', linewidth=2, label=f'Initial: {initial_mae:.3f}')
    axes[1, 0].axvline(final_mae, color='green', linestyle='--', linewidth=2, label=f'Final: {final_mae:.3f}')
    axes[1, 0].set_xlabel('MAE')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('MAE Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Improvement Distribution
    improvements = results_df['improvement'][results_df['improvement'] != 0]
    if len(improvements) > 0:
        axes[1, 1].hist(improvements, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('MAE Improvement')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Improvement Distribution')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No improvements\nrecorded', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Improvement Distribution')
    
    # 6. Summary Statistics
    axes[1, 2].axis('off')
    summary_text = f"""
OPTIMIZATION SUMMARY

Property: {property_name.upper()}
Total Iterations: {len(results_df)}
Initial MAE: {initial_mae:.4f}
Final MAE: {final_mae:.4f}
Total Improvement: {improvement:.4f}
Improvement %: {(improvement/initial_mae)*100:.2f}%

Convergence:
• Total Iterations: {len(results_df)}
• Improved Iterations: {len(results_df[results_df['improvement'] > 0])}
• No Change Iterations: {len(results_df[results_df['improvement'] == 0])}
• Worsened Iterations: {len(results_df[results_df['improvement'] < 0])}

Best Performance:
• Best MAE: {results_df['mae'].min():.4f}
• Best Iteration: {results_df.loc[results_df['mae'].idxmin(), 'iteration']}
    """
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'blend_optimization_results_{property_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Optimization plots saved to: {plot_path}")


def create_validation_performance_plots(validation_df: pd.DataFrame, property_name: str, 
                                      initial_mae: float, final_mae: float, 
                                      initial_predictions: List[float], final_predictions: List[float],
                                      output_dir: str = ".", plot_suffix: str = ""):
    """Create validation blend performance plots."""
    print("📊 Creating validation performance plots...")
    
    # Set up the dark theme
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'font.size': 12,
        'font.weight': 'normal',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': 'black',
        'legend.framealpha': 0.8
    })
    
    # Property abbreviation mapping
    property_abbreviations = {
        'tensile': 'TS (MPa)',
        'tensile strength': 'TS (MPa)',
        'wvtr': 'WVTR (g/m²/day)',
        'water vapor transmission rate': 'WVTR (g/m²/day)',
        'eab': 'EaB (%)',
        'elongation at break': 'EaB (%)',
        'cobb': 'Cobb (g/m²)',
        'cobb angle': 'Cobb (g/m²)',
        'seal': 'Max Seal Strength (N/15mm)',
        'sealing': 'Max Seal Strength (N/15mm)',
        'adhesion': 'Max Seal Strength (N/15mm)',
        'compost': 'Compost (%)',
        'compostability': 'Compost (%)',
        'otr': 'OTR (cc/m²/day)',
        'oxygen transmission rate': 'OTR (cc/m²/day)'
    }
    
    # Get property abbreviation
    prop_abbrev = property_abbreviations.get(property_name.lower(), property_name.upper())
    
    # Get actual values
    property_cols = [col for col in validation_df.columns if col.startswith('property')]
    if property_cols:
        actual_values = validation_df[property_cols[0]].values
    else:
        actual_values = np.zeros(len(validation_df))
    
    # Check if this is a dual property (TS or EAB)
    is_dual_property = len(property_cols) == 2 and property_name.lower() in ['ts', 'eab']
    
    # Create plot
    if is_dual_property:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 1, figsize=(6, 10))
    
    if not is_dual_property:
        axes = axes.reshape(2, 1)
    
    # Get blend labels
    if 'Materials' in validation_df.columns:
        blend_labels = validation_df['Materials'].astype(str).values
    else:
        blend_labels = [f'Blend {i+1}' for i in range(len(validation_df))]
    
    # Plot for each property
    properties_to_plot = property_cols if is_dual_property else [property_cols[0]]
    
    for i, prop_col in enumerate(properties_to_plot):
        if is_dual_property:
            actual_vals = validation_df[prop_col].values
            # For dual properties, use the same predictions for both properties
            # since we only have single predictions (property1)
            initial_preds = initial_predictions
            final_preds = final_predictions
            
            # For dual properties, use specific labels
            if property_name.lower() == 'ts':
                current_prop_label = 'TS-MD (MPa)' if i == 0 else 'TS-TD (MPa)'
            elif property_name.lower() == 'eab':
                current_prop_label = 'EaB-MD (%)' if i == 0 else 'EaB-TD (%)'
            else:
                current_prop_label = f'{prop_abbrev} - {prop_col}'
        else:
            actual_vals = actual_values
            initial_preds = initial_predictions
            final_preds = final_predictions
            current_prop_label = prop_abbrev
        
        # Filter arrays to match length
        actual_vals = actual_vals[:len(initial_preds)]
        blend_labels_filtered = blend_labels[:len(initial_preds)]
        
        # Calculate R² for this specific property
        initial_r2 = 1 - (np.sum((actual_vals - initial_preds)**2) / np.sum((actual_vals - np.mean(actual_vals))**2))
        final_r2 = 1 - (np.sum((actual_vals - final_preds)**2) / np.sum((actual_vals - np.mean(actual_vals))**2))
        
        # Plot 1: Before vs After Optimization
        if is_dual_property:
            ax1 = axes[0, i]
            ax2 = axes[1, i]
        else:
            ax1 = axes[0, 0]
            ax2 = axes[1, 0]
        
        # Before optimization
        ax1.scatter(actual_vals, initial_preds, alpha=0.7, s=60, color='red', edgecolors='white', linewidth=1)
        ax1.plot([actual_vals.min(), actual_vals.max()], [actual_vals.min(), actual_vals.max()], 'w--', alpha=0.8, linewidth=2)
        ax1.set_xlabel(f'Actual {current_prop_label}')
        ax1.set_ylabel(f'Predicted {current_prop_label}')
        ax1.set_title(f'Before Optimization (R² = {initial_r2:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Add metrics text box for before optimization
        initial_mae = np.mean(np.abs(actual_vals - initial_preds))
        metrics_text = f'MAE: {initial_mae:.3f}\nR²: {initial_r2:.3f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # After optimization
        ax2.scatter(actual_vals, final_preds, alpha=0.7, s=60, color='green', edgecolors='white', linewidth=1)
        ax2.plot([actual_vals.min(), actual_vals.max()], [actual_vals.min(), actual_vals.max()], 'w--', alpha=0.8, linewidth=2)
        ax2.set_xlabel(f'Actual {current_prop_label}')
        ax2.set_ylabel(f'Predicted {current_prop_label}')
        ax2.set_title(f'After Optimization (R² = {final_r2:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Add metrics text box for after optimization
        final_mae = np.mean(np.abs(actual_vals - final_preds))
        metrics_text = f'MAE: {final_mae:.3f}\nR²: {final_r2:.3f}'
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add blend labels as annotations
        for j, (actual, pred, label) in enumerate(zip(actual_vals, final_preds, blend_labels_filtered)):
            if j % 2 == 0:  # Show every other label to avoid crowding
                ax2.annotate(label, (actual, pred), xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='white', alpha=0.8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f'last_{len(validation_df)}_blends_performance{plot_suffix}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Validation performance plots saved to: {plot_path}")


def create_testing_performance_plot(testing_df: pd.DataFrame, property_name: str, 
                                  final_mae: float, actual_values: List[float], 
                                  final_predictions: List[float], output_dir: str = ".", 
                                  plot_suffix: str = ""):
    """Create testing performance plot showing only final predictions vs actual values."""
    print("📊 Creating testing performance plot...")
    
    # Ensure predictions and actual values have the same length
    if len(final_predictions) != len(actual_values):
        print(f"⚠️  Warning: Mismatch in prediction lengths - actual: {len(actual_values)}, final: {len(final_predictions)}")
        min_len = min(len(final_predictions), len(actual_values))
        final_predictions = final_predictions[:min_len]
        actual_values = actual_values[:min_len]
    
    if len(final_predictions) == 0:
        print("⚠️  No valid predictions available for plotting")
        return
    
    # Set up the dark theme EXACTLY like XGBoost training plots
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'black',
        'axes.facecolor': 'black',
        'axes.edgecolor': 'white',
        'axes.labelcolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'gray',
        'grid.alpha': 0.3,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'font.size': 12,
        'font.weight': 'normal',
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.facecolor': 'black',
        'legend.framealpha': 0.8
    })
    
    # Property abbreviation mapping with units
    property_abbreviations = {
        'tensile': 'TS (MPa)',
        'tensile strength': 'TS (MPa)',
        'wvtr': 'WVTR (g/m²/day)',
        'water vapor transmission rate': 'WVTR (g/m²/day)',
        'eab': 'EaB (%)',
        'elongation at break': 'EaB (%)',
        'cobb': 'Cobb (g/m²)',
        'cobb angle': 'Cobb (g/m²)',
        'seal': 'Max Seal Strength (N/15mm)',
        'sealing': 'Max Seal Strength (N/15mm)',
        'adhesion': 'Max Seal Strength (N/15mm)',
        'compost': 'Compost (%)',
        'compostability': 'Compost (%)',
        'otr': 'OTR (cc/m²/day)',
        'oxygen transmission rate': 'OTR (cc/m²/day)'
    }
    
    # Get property abbreviation
    prop_abbrev = property_abbreviations.get(property_name.lower(), property_name.upper())
    
    # Get actual values (assuming they're in the first property column)
    property_cols = [col for col in testing_df.columns if col.startswith('property')]
    if property_cols:
        actual_values = testing_df[property_cols[0]].values
    else:
        actual_values = np.zeros(len(testing_df))
    
    # Check if this is a dual property (TS or EAB)
    is_dual_property = len(property_cols) == 2 and property_name.lower() in ['ts', 'eab']
    
    # Create plot
    if is_dual_property:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 6))
        axes = [axes]
    
    # Get blend labels from Materials column (filtered to match predictions length)
    if 'Materials' in testing_df.columns:
        blend_labels = testing_df['Materials'].astype(str).values[:len(final_predictions)]
    else:
        blend_labels = [f'Blend {i+1}' for i in range(len(final_predictions))]
    
    # Calculate R2
    actual_array = np.array(actual_values)
    pred_array = np.array(final_predictions)
    r2 = 1 - (np.sum((actual_array - pred_array)**2) / np.sum((actual_array - np.mean(actual_array))**2))
    
    # Plot for each property (single or dual)
    properties_to_plot = property_cols if is_dual_property else [property_cols[0]]
    
    for i, prop_col in enumerate(properties_to_plot):
        if is_dual_property:
            actual_vals = testing_df[prop_col].values[:len(final_predictions)]
            
            # For dual properties, use specific labels
            if property_name.lower() == 'ts':
                current_prop_label = 'TS-MD (MPa)' if i == 0 else 'TS-TD (MPa)'
            elif property_name.lower() == 'eab':
                current_prop_label = 'EaB-MD (%)' if i == 0 else 'EaB-TD (%)'
            else:
                current_prop_label = prop_abbrev
        else:
            actual_vals = actual_values
            current_prop_label = prop_abbrev
        
        # For dual properties, we need to handle the fact that we only have single predictions
        # For property2 (TD), we'll use the same predictions as property1 (MD) for now
        # This is a limitation - ideally we'd need separate simulations for each property
        if is_dual_property and i == 1:
            # For property2 (TD), use the same predictions as property1
            pred_vals = final_predictions
        else:
            pred_vals = final_predictions
        
        # Plot: Final Predictions vs Actual
        axes[i].scatter(actual_vals, pred_vals, color='#6BFF6B', s=100, alpha=0.7)
        axes[i].plot([actual_vals.min(), actual_vals.max()], 
                    [actual_vals.min(), actual_vals.max()], 'w--', lw=3, alpha=0.8, label='y=x')
        axes[i].set_xlabel(f'Actual {current_prop_label}', fontweight='bold')
        axes[i].set_ylabel(f'Predicted {current_prop_label}', fontweight='bold')
        axes[i].set_title(f'{current_prop_label} - Testing Set (Optimized Predictions)', fontweight='bold')
        axes[i].legend(loc='best', framealpha=0.8)
        axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Add blend labels
        for j, (actual, pred) in enumerate(zip(actual_vals, pred_vals)):
            blend_label = str(blend_labels[j])
            axes[i].annotate(blend_label, (actual, pred), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8, color='white')
        
        # Calculate R2 for this specific property
        if len(actual_vals) > 0 and len(pred_vals) > 0:
            r2_prop = 1 - (np.sum((actual_vals - pred_vals)**2) / np.sum((actual_vals - np.mean(actual_vals))**2))
        else:
            r2_prop = 0.0
        
        # Add MAE and R2 metrics
        metrics_text = f'MAE: {final_mae:.3f}\nR²: {r2_prop:.3f}'
        axes[i].text(0.05, 0.95, metrics_text, transform=axes[i].transAxes,
                    fontsize=12, verticalalignment='top', fontweight='bold', color='red',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    plot_filename = f'last_{len(testing_df)}_blends_performance{plot_suffix}.png'
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Testing performance plot saved to: {plot_path}")
