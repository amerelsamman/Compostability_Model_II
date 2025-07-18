import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Random Forest model for tensile strength prediction with thickness')
parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output', type=str, required=True, help='Output directory path')
args = parser.parse_args()

print("Loading featurized polymer blends data with thickness...")
df = pd.read_csv(args.input)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check thickness column
thickness_col = 'Thickness (um)'
thickness_count = 0
thickness_range = "N/A"
if thickness_col in df.columns:
    thickness_count = df[thickness_col].notna().sum()
    thickness_range = f"{df[thickness_col].min():.2f} - {df[thickness_col].max():.2f}"
    print(f"Thickness column found with {thickness_count} non-null values")
    print(f"Thickness range: {thickness_range}")
else:
    print("Thickness column not found!")

# Define target column
target_col = 'property'

# Separate features and target
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
X = df.drop(columns=[target_col, 'Materials'] + smiles_cols)
y = df[target_col]

# Apply log transformation to target values
log_y = np.log(y)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Log-transformed target range: {log_y.min():.4f} to {log_y.max():.4f}")

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

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop'
)

# Create the full pipeline with Random Forest
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,  # Allow full depth for overfitting
        min_samples_split=2,  # Minimum allowed by scikit-learn
        min_samples_leaf=1,   # Allow leaf nodes with single samples
        n_jobs=-1  # Use all CPU cores
    ))
])

# Split the data using train_test_split while ensuring last 4 blends are in testing
last_4_indices = list(range(len(df) - 4, len(df)))
print(f"Last 4 blend indices: {last_4_indices}")

# Remove last 4 from the main pool for train_test_split
remaining_indices = [i for i in range(len(df)) if i not in last_4_indices]
X_remaining = X.iloc[remaining_indices]
log_y_remaining = log_y.iloc[remaining_indices]

# Use train_test_split on the remaining data
X_temp_train, X_temp_test, y_temp_train, y_temp_test, temp_train_indices, temp_test_indices = train_test_split(
    X_remaining, log_y_remaining, remaining_indices, 
    test_size=0.2, random_state=42, shuffle=True
)

# Combine: last 4 always in testing, rest split by train_test_split
train_indices = temp_train_indices
test_indices = last_4_indices + temp_test_indices

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
log_y_train = log_y.iloc[train_indices]
log_y_test = log_y.iloc[test_indices]

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Last 4 blends in testing: {all(i in test_indices for i in last_4_indices)}")

# Check if thickness values are in training set
if thickness_col in X.columns:
    train_thickness = X_train[thickness_col].unique()
    print(f"Unique thickness values in training set: {train_thickness}")

# Oversample the last 16-4 blends (last 12 blends) 20x
print("\nOversampling last 12 blends 20x...")
last_16_indices = list(range(len(df) - 16, len(df)))
last_12_indices = list(range(len(df) - 16, len(df) - 4))  # Last 12 blends (16-4)
print(f"Last 16 blend indices: {last_16_indices}")
print(f"Last 12 blend indices (for oversampling): {last_12_indices}")

# Create oversampling weights - set all to 1.0, then 20x for last 12
weights = pd.Series(1.0, index=log_y_train.index)
for idx in last_12_indices:
    if idx in log_y_train.index:
        weights.iloc[log_y_train.index.get_loc(idx)] = weights.iloc[log_y_train.index.get_loc(idx)] * 20

print(f"Updated sampling weights range: {weights.min():.2f} - {weights.max():.2f}")

# Perform oversampling
n_samples = len(X_train)
oversampled_indices = []

for i in range(n_samples):
    # Add each sample with probability proportional to its weight
    n_copies = max(1, int(weights.iloc[i]))
    oversampled_indices.extend([i] * n_copies)

print(f"Original training samples: {len(X_train)}")
print(f"Oversampled training samples: {len(oversampled_indices)}")
print(f"Oversampling factor: {len(oversampled_indices) / len(X_train):.2f}x")

# Create oversampled datasets
X_train_oversampled = X_train.iloc[oversampled_indices].reset_index(drop=True)
log_y_train_oversampled = log_y_train.iloc[oversampled_indices].reset_index(drop=True)

# Use oversampled data for training
X_train = X_train_oversampled
log_y_train = log_y_train_oversampled

# Save original data for plotting (before oversampling)
X_original = X.copy()
log_y_original = log_y.copy()

# Train the model
print("\nTraining the comprehensive model (log scale)...")
model.fit(X_train, log_y_train)

# Make predictions (log scale)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model (log scale)
print("\n=== MODEL PERFORMANCE (LOG SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(log_y_train, y_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y_train, y_pred_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y_train, y_pred_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y_train, y_pred_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(log_y_test, y_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y_test, y_pred_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y_test, y_pred_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y_test, y_pred_test)):.4f}")

# Inverse transform predictions and targets to original scale
orig_y_train = np.exp(log_y_train)
orig_y_test = np.exp(log_y_test)
orig_pred_train = np.exp(y_pred_train)
orig_pred_test = np.exp(y_pred_test)

# Evaluate the model (original scale)
print("\n=== MODEL PERFORMANCE (ORIGINAL SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(orig_y_train, orig_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y_train, orig_pred_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y_train, orig_pred_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y_train, orig_pred_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(orig_y_test, orig_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y_test, orig_pred_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y_test, orig_pred_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y_test, orig_pred_test)):.4f}")

# Feature importance analysis
print("\n=== FEATURE IMPORTANCE ===")
feature_names = []
if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
    feature_names = model.named_steps['preprocessor'].get_feature_names_out()
else:
    # Fallback for older sklearn versions
    feature_names = [f"feature_{i}" for i in range(len(model.named_steps['regressor'].feature_importances_))]

importances = model.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]

print("Top 30 most important features:")
for i in range(min(30, len(indices))):
    print(f"{i+1:2d}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

# Create visualizations
plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted plot
plt.subplot(3, 3, 1)
plt.scatter(orig_y_train, orig_pred_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(orig_y_train, orig_pred_train):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y_train, orig_pred_train)):.2f})', color='blue')
plt.scatter(orig_y_test, orig_pred_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(orig_y_test, orig_pred_test):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y_test, orig_pred_test)):.2f})', color='red')
plt.plot([orig_y_train.min(), orig_y_train.max()], [orig_y_train.min(), orig_y_train.max()], 'k--', lw=2)
plt.xlabel('Actual Tensile Strength (MPa)')
plt.ylabel('Predicted Tensile Strength (MPa)')
plt.title('Actual vs Predicted Tensile Strength')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Residuals plot
plt.subplot(3, 3, 2)
residuals_train = orig_y_train - orig_pred_train
residuals_test = orig_y_test - orig_pred_test
plt.scatter(orig_pred_train, residuals_train, alpha=0.6, label='Training', color='blue')
plt.scatter(orig_pred_test, residuals_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Tensile Strength (MPa)')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Feature importance plot
plt.subplot(3, 3, 3)
top_features = 20
top_indices = indices[:top_features]
plt.barh(range(top_features), importances[top_indices])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance')
plt.title(f'Top {top_features} Feature Importances')
plt.gca().invert_yaxis()

# 4. Distribution of target variable
plt.subplot(3, 3, 4)
plt.hist(orig_y_train, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Tensile Strength (MPa)')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable')
plt.grid(True, alpha=0.3)

# 5. Training vs Test performance comparison
plt.subplot(3, 3, 5)
metrics = ['R²', 'MAE', 'RMSE']
train_scores = [
    r2_score(orig_y_train, orig_pred_train),
    mean_absolute_error(orig_y_train, orig_pred_train),
    np.sqrt(mean_squared_error(orig_y_train, orig_pred_train))
]
test_scores = [
    r2_score(orig_y_test, orig_pred_test),
    mean_absolute_error(orig_y_test, orig_pred_test),
    np.sqrt(mean_squared_error(orig_y_test, orig_pred_test))
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Training', color='blue', alpha=0.7)
plt.bar(x + width/2, test_scores, width, label='Test', color='red', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Training vs Test Performance')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Prediction error distribution
plt.subplot(3, 3, 6)
plt.hist(residuals_train, bins=15, alpha=0.7, label='Training', color='blue')
plt.hist(residuals_test, bins=15, alpha=0.7, label='Test', color='red')
plt.xlabel('Prediction Error (MPa)')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

# 7. Feature importance by category
plt.subplot(3, 3, 7)
def get_importance_sum(features, feature_names, importances):
    """Calculate importance sum for features containing a specific string"""
    total_importance = 0
    for i, feature in enumerate(feature_names):
        if any(f in feature for f in features):
            total_importance += importances[i]
    return total_importance

# Calculate importance by category
categories = {
    'Polymer Grades': ['Polymer Grade'],
    'Volume Fractions': ['vol_fraction'],
    'SP Descriptors': ['SP_'],
    'Chemical Groups': [
        'alcohol', 'ether', 'ester', 'amide', 'ketone', 'aldehyde',
        'vinyl', 'allene', 'nitrile', 'imine', 'thiol', 'thioether', 'amine',
        'azide', 'azo', 'imide', 'sulfone', 'sulfoxide', 'phosphate', 'nitro'
    ],
    'Ring Systems': ['phenyls', 'cyclohexanes', 'cyclopentanes', 'cyclopentenes', 'thiophenes', 'aromatic_rings', 'aliphatic_rings', 'other_rings'],
    'Carbon Types': ['primary_carbon', 'secondary_carbon', 'tertiary_carbon', 'quaternary_carbon'],
    'Halogens': ['SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I'],
    'Thickness': ['thickness']
}

category_importance = {}
for category, features in categories.items():
    category_importance[category] = get_importance_sum(features, feature_names, importances)

# Plot category importance
plt.bar(category_importance.keys(), category_importance.values(), color='skyblue')
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title('Feature Importance by Category (TS v5 - Random Forest Model with Thickness)')
plt.xticks(rotation=45)

# 8. Model complexity analysis
plt.subplot(3, 3, 8)
n_trees = model.named_steps['regressor'].n_estimators
max_depth = model.named_steps['regressor'].max_depth
learning_rate = 0  # Random Forest doesn't use learning rate

# Handle None max_depth (unlimited depth)
if max_depth is None:
    max_depth_display = "Unlimited"
    max_depth_value = 20  # Use a reasonable value for visualization
else:
    max_depth_display = str(max_depth)
    max_depth_value = max_depth

complexity_metrics = ['Number of Trees', 'Max Depth', 'Learning Rate']
complexity_values = [n_trees, max_depth_value, learning_rate]

plt.bar(complexity_metrics, complexity_values, color=['green', 'orange', 'purple'])
plt.title('Model Complexity Metrics (TS v5 - Random Forest Model with Thickness)')
plt.ylabel('Value')
plt.xticks(rotation=45)

# Add text annotation for unlimited depth if applicable
if max_depth is None:
    plt.text(1, max_depth_value + 1, 'Unlimited', ha='center', va='bottom', fontweight='bold')

# 9. Data summary
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'Total Features: {len(feature_names)}', fontsize=12)
plt.text(0.1, 0.7, f'Categorical: {len(categorical_features)}', fontsize=12)
plt.text(0.1, 0.6, f'Numerical: {len(numerical_features)}', fontsize=12)
plt.text(0.1, 0.5, f'Training Samples: {len(orig_y_train)}', fontsize=12)
plt.text(0.1, 0.4, f'Test Samples: {len(orig_y_test)}', fontsize=12)
plt.text(0.1, 0.3, f'Target Range: {orig_y_train.min():.1f} - {orig_y_train.max():.1f} MPa', fontsize=12)
plt.text(0.1, 0.2, f'Target Mean: {orig_y_train.mean():.1f} MPa', fontsize=12)
plt.text(0.1, 0.1, f'Target Std: {orig_y_train.std():.1f} MPa', fontsize=12)
plt.title('Dataset Summary')
plt.axis('off')

plt.tight_layout()

# Error-to-noise analysis for test set (original scale)
data_std = np.std(orig_y_test)
data_mean = np.mean(orig_y_test)
data_cv = data_std / data_mean  # Coefficient of variation
relative_errors = np.abs(orig_y_test - orig_pred_test) / orig_y_test
relative_mae = np.mean(relative_errors)
relative_rmse = np.sqrt(np.mean(relative_errors**2))
error_to_std_ratio = relative_mae / data_cv
rmse_to_std_ratio = relative_rmse / data_cv
print(f"\n=== ERROR vs DATA VARIABILITY ANALYSIS ===")
print(f"Data Statistics:")
print(f"  Mean TS: {data_mean:.2f}")
print(f"  Std TS: {data_std:.2f}")
print(f"  Coefficient of Variation (CV): {data_cv:.3f} ({data_cv*100:.1f}%)")
print(f"\nError Metrics:")
print(f"  Relative MAE: {relative_mae:.3f} ({relative_mae*100:.1f}%)")
print(f"  Relative RMSE: {relative_rmse:.3f} ({relative_rmse*100:.1f}%)")
print(f"\nError-to-Noise Ratios:")
print(f"  Relative MAE / CV: {error_to_std_ratio:.3f}")
print(f"  Relative RMSE / CV: {rmse_to_std_ratio:.3f}")
if error_to_std_ratio < 1:
    print(f"  ✅ Model error ({relative_mae*100:.1f}%) is LESS than data variability ({data_cv*100:.1f}%)")
    print(f"  ✅ Model is performing well relative to inherent data noise")
elif error_to_std_ratio < 2:
    print(f"  ⚠️  Model error ({relative_mae*100:.1f}%) is similar to data variability ({data_cv*100:.1f}%)")
    print(f"  ⚠️  Model performance is acceptable but could be improved")
else:
    print(f"  ❌ Model error ({relative_mae*100:.1f}%) is GREATER than data variability ({data_cv*100:.1f}%)")
    print(f"  ❌ Model is not performing well relative to inherent data noise")

# Last 4 blends performance (log and original scale)
print("\n=== LAST 4 BLENDS PERFORMANCE ===")
last_4_X = X.iloc[last_4_indices]
last_4_log_y = log_y.iloc[last_4_indices]
last_4_pred_log = model.predict(last_4_X)
last_4_mae_log = mean_absolute_error(last_4_log_y, last_4_pred_log)
last_4_rmse_log = np.sqrt(mean_squared_error(last_4_log_y, last_4_pred_log))
last_4_r2_log = r2_score(last_4_log_y, last_4_pred_log)
print(f"Last 4 Blends (Log Scale):")
print(f"R² Score: {last_4_r2_log:.4f}")
print(f"Mean Absolute Error: {last_4_mae_log:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse_log:.4f}")
# Original scale
last_4_actual = np.exp(last_4_log_y)
last_4_pred = np.exp(last_4_pred_log)
last_4_mae = mean_absolute_error(last_4_actual, last_4_pred)
last_4_rmse = np.sqrt(mean_squared_error(last_4_actual, last_4_pred))
last_4_r2 = r2_score(last_4_actual, last_4_pred)
print(f"Last 4 Blends (Original Scale):")
print(f"R² Score: {last_4_r2:.4f}")
print(f"Mean Absolute Error: {last_4_mae:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse:.4f}")

# Create separate plot for last 4 blends performance
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y, last_4_pred_log, color='red', s=100, alpha=0.7, label=f'Last 4 Blends (MAE={last_4_mae_log:.2f}, RMSE={last_4_rmse_log:.2f})')
plt.plot([last_4_log_y.min(), last_4_log_y.max()], [last_4_log_y.min(), last_4_log_y.max()], 'k--', lw=2)
plt.xlabel('Actual Log(Tensile Strength)')
plt.ylabel('Predicted Log(Tensile Strength)')
plt.title('Last 4 Blends: Log-Transformed Predictions (TS v11 - Random Forest Model with Thickness)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_log_y, last_4_pred_log)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions
plt.subplot(1, 2, 2)
# Calculate relative errors
relative_errors = np.abs(last_4_actual - last_4_pred) / last_4_actual
relative_mae = np.mean(relative_errors)
relative_rmse = np.sqrt(np.mean(relative_errors**2))

# Calculate data statistics
data_std = np.std(last_4_actual)
data_mean = np.mean(last_4_actual)
data_cv = data_std / data_mean  # Coefficient of variation

# Calculate error-to-noise ratio
error_to_std_ratio = relative_mae / data_cv

plt.scatter(last_4_actual, last_4_pred, color='blue', s=100, alpha=0.7)
plt.plot([last_4_actual.min(), last_4_actual.max()], [last_4_actual.min(), last_4_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Tensile Strength (Original Scale)')
plt.ylabel('Predicted Tensile Strength (Original Scale)')
plt.title('Tensile Strength Predictions - Original Scale (TS v11 - Random Forest Model with Thickness)')

# Add metrics as text box on the plot
metrics_text = (
    f'Rel MAE: {relative_mae:.2f}\n'
    f'Rel RMSE: {relative_rmse:.2f}\n'
    f'CV: {data_cv:.2f}\n'
    f'Rel MAE / CV: {error_to_std_ratio:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, alpha=0.3)

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_actual, last_4_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nLast 4 Blends - Original Scale Performance:")
print(f"R² Score: {last_4_r2:.4f}")
print(f"Mean Absolute Error: {last_4_mae:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse:.4f}")
print(f"Relative MAE: {relative_mae:.4f} ({relative_mae*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse:.4f} ({relative_rmse*100:.2f}%)")
print("✅ Last 4 blends performance plot saved as 'last_4_blends_performance.png'")



# Save the model
import joblib
os.makedirs(args.output, exist_ok=True)
model_path = os.path.join(args.output, 'comprehensive_polymer_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved as '{model_path}'")

# Save the main plot
plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results.png'), dpi=300, bbox_inches='tight')
plt.show()

# Example prediction
print("\n=== EXAMPLE PREDICTION ===")
example_data = X.iloc[0:1].copy()
actual_value_log = log_y.iloc[0]
predicted_value_log = model.predict(example_data)[0]
print(f"Actual Log(TS): {actual_value_log:.2f}")
print(f"Predicted Log(TS): {predicted_value_log:.2f}")
print(f"Prediction Error: {abs(actual_value_log - predicted_value_log):.2f}")
# Convert back to original scale for comparison
original_actual = np.exp(actual_value_log)
original_predicted = np.exp(predicted_value_log)
print(f"Actual TS (original scale): {original_actual:.2f}")
print(f"Predicted TS (original scale): {original_predicted:.2f}")
print(f"Original Scale Error: {abs(original_actual - original_predicted):.2f}")

# Prediction function
def predict_tensile_strength(input_data):
    """
    Predict tensile strength for polymer blend data (Model - High Training Performance)
    Parameters:
    - input_data: pandas DataFrame with all features (excluding SMILES columns)
    Returns:
    - float: predicted tensile strength in MPa
    """
    # Handle missing values
    for col in categorical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('Unknown')
    for col in numerical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna(0)
    prediction_log = model.predict(input_data)[0]
    return np.exp(prediction_log)

print("\n=== MODEL SUMMARY ===")
print("✅ Random Forest model successfully trained")
print("✅ All features included (categorical + numerical + featurized + thickness)")
print("✅ One-hot encoding applied to categorical features")
print("✅ Model allowed to overfit as requested")
print("✅ Feature importance analysis completed")
print("✅ Comprehensive visualizations saved as 'comprehensive_polymer_model_results.png'")
print(f"✅ Model saved as '{model_path}'")
print(f"✅ All files saved to: {args.output}")
print("✅ Prediction function created for new data")

# Save training explanation
with open(os.path.join(args.output, 'training_explanation.txt'), 'w') as f:
    f.write("Random Forest Model Training Explanation (TS v5 with Thickness)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} samples with thickness information\n")
    f.write(f"Features: {len(feature_names)} (including thickness)\n")
    f.write(f"Training set: {len(orig_y_train)} samples\n")
    f.write(f"Test set: {len(orig_y_test)} samples\n\n")
    
    f.write("Model Configuration:\n")
    f.write(f"- Algorithm: Random Forest\n")
    f.write(f"- Estimators: 200\n")
    f.write(f"- Max depth: Unlimited\n")
    f.write(f"- Min samples split: 2\n")
    f.write(f"- Min samples leaf: 1\n\n")
    
    f.write("Training Set Performance:\n")
    f.write(f"- R² Score (log scale): {r2_score(log_y_train, y_pred_train):.4f}\n")
    f.write(f"- R² Score (original scale): {r2_score(orig_y_train, orig_pred_train):.4f}\n")
    f.write(f"- RMSE (original scale): {np.sqrt(mean_squared_error(orig_y_train, orig_pred_train)):.4f} MPa\n")
    f.write(f"- MAE (original scale): {mean_absolute_error(orig_y_train, orig_pred_train):.4f} MPa\n\n")
    
    f.write("Test Set Performance:\n")
    f.write(f"- R² Score (log scale): {r2_score(log_y_test, y_pred_test):.4f}\n")
    f.write(f"- R² Score (original scale): {r2_score(orig_y_test, orig_pred_test):.4f}\n")
    f.write(f"- RMSE (original scale): {np.sqrt(mean_squared_error(orig_y_test, orig_pred_test)):.4f} MPa\n")
    f.write(f"- MAE (original scale): {mean_absolute_error(orig_y_test, orig_pred_test):.4f} MPa\n\n")
    
    f.write("Last 4 Blends Performance:\n")
    f.write(f"- R² Score (log scale): {last_4_r2_log:.4f}\n")
    f.write(f"- R² Score (original scale): {last_4_r2:.4f}\n")
    f.write(f"- RMSE (original scale): {last_4_rmse:.4f} MPa\n")
    f.write(f"- MAE (original scale): {last_4_mae:.4f} MPa\n")
    f.write(f"- Relative MAE: {relative_mae:.4f} ({relative_mae*100:.2f}%)\n")
    f.write(f"- Relative RMSE: {relative_rmse:.4f} ({relative_rmse*100:.2f}%)\n\n")
    
    f.write("Top 10 Feature Importances:\n")
    for i in range(min(10, len(indices))):
        f.write(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}\n")
    
    f.write(f"\nThickness Information:\n")
    f.write(f"- Thickness values in dataset: {thickness_count}\n")
    f.write(f"- Thickness range: {thickness_range}\n")
    f.write(f"- Thickness values in training set: {train_thickness}\n")
    # Find thickness importance with correct prefix
    thickness_feature_name = None
    for feature_name in feature_names:
        if 'thickness' in feature_name:
            thickness_feature_name = feature_name
            break
    if thickness_feature_name:
        thickness_importance = importances[list(feature_names).index(thickness_feature_name)]
        f.write(f"- Thickness importance: {thickness_importance:.4f}\n")
    else:
        f.write(f"- Thickness importance: Not found in feature names\n")

print(f"✅ Training explanation saved to {os.path.join(args.output, 'training_explanation.txt')}") 