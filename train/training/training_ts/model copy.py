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

# Define target columns for both MD and TD
target_col1 = 'property1'  # MD (Machine Direction)
target_col2 = 'property2'  # TD (Transverse Direction)

# Separate features and targets
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
X = df.drop(columns=[target_col1, target_col2, 'Materials'] + smiles_cols)
y1 = df[target_col1]  # MD property
y2 = df[target_col2]  # TD property

# Apply log transformation to target values
log_y1 = np.log(y1)
log_y2 = np.log(y2)

print(f"\nFeature matrix shape: {X.shape}")
print(f"MD target vector shape: {y1.shape}")
print(f"TD target vector shape: {y2.shape}")
print(f"Log-transformed MD range: {log_y1.min():.4f} to {log_y1.max():.4f}")
print(f"Log-transformed TD range: {log_y2.min():.4f} to {log_y2.max():.4f}")

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

# Split the data using train_test_split while ensuring last 4 blends are in training
last_4_indices = list(range(len(df) - 4, len(df)))
print(f"Last 4 blend indices: {last_4_indices}")

# Remove last 4 from the main pool for train_test_split
remaining_indices = [i for i in range(len(df)) if i not in last_4_indices]
X_remaining = X.iloc[remaining_indices]
log_y1_remaining = log_y1.iloc[remaining_indices]
log_y2_remaining = log_y2.iloc[remaining_indices]

# Use train_test_split on the remaining data
X_temp_train, X_temp_test, y1_temp_train, y1_temp_test, y2_temp_train, y2_temp_test, temp_train_indices, temp_test_indices = train_test_split(
    X_remaining, log_y1_remaining, log_y2_remaining, remaining_indices, 
    test_size=0.2, random_state=42, shuffle=True
)

# Combine: last 4 always in training, rest split by train_test_split
train_indices = last_4_indices + temp_train_indices
test_indices = temp_test_indices

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
log_y1_train = log_y1.iloc[train_indices]
log_y1_test = log_y1.iloc[test_indices]
log_y2_train = log_y2.iloc[train_indices]
log_y2_test = log_y2.iloc[test_indices]

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Last 4 blends in training: {all(i in train_indices for i in last_4_indices)}")

# Oversample the last 4 blends by 10x
print("\nOversampling last 4 blends by 10x...")
last_4_X = X.iloc[last_4_indices]
last_4_log_y1 = log_y1.iloc[last_4_indices]
last_4_log_y2 = log_y2.iloc[last_4_indices]

# Repeat the last 4 blends 9 more times (total 10x)
oversampled_X = []
oversampled_y1 = []
oversampled_y2 = []

# Add original training data (excluding last 4)
other_train_indices = [i for i in train_indices if i not in last_4_indices]
oversampled_X.append(X.iloc[other_train_indices])
oversampled_y1.append(log_y1.iloc[other_train_indices])
oversampled_y2.append(log_y2.iloc[other_train_indices])

# Add last 4 blends 2 times
for _ in range(1):
    oversampled_X.append(last_4_X)
    oversampled_y1.append(last_4_log_y1)
    oversampled_y2.append(last_4_log_y2)

# Combine all data
X_train_oversampled = pd.concat(oversampled_X, ignore_index=True)
log_y1_train_oversampled = pd.concat(oversampled_y1, ignore_index=True)
log_y2_train_oversampled = pd.concat(oversampled_y2, ignore_index=True)

print(f"Original training set size: {X_train.shape[0]}")
print(f"Oversampled training set size: {X_train_oversampled.shape[0]}")
print(f"Last 4 blends repeated 2x: {last_4_X.shape[0] * 2} samples")

# Use oversampled data for training
X_train = X_train_oversampled
log_y1_train = log_y1_train_oversampled
log_y2_train = log_y2_train_oversampled

# Check if thickness values are in training set
if thickness_col in X.columns:
    train_thickness = X_train[thickness_col].unique()
    print(f"Unique thickness values in training set: {train_thickness}")

# Create two separate models for MD and TD
print("\nTraining the comprehensive model for MD (property1)...")
model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,  # Allow full depth for overfitting
        min_samples_split=2,  # Minimum allowed by scikit-learn
        min_samples_leaf=1,   # Allow leaf nodes with single samples
        n_jobs=-1  # Use all CPU cores
    ))
])

print("Training the comprehensive model for TD (property2)...")
model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=200,
        max_depth=None,  # Allow full depth for overfitting
        min_samples_split=2,  # Minimum allowed by scikit-learn
        min_samples_leaf=1,   # Allow leaf nodes with single samples
        n_jobs=-1  # Use all CPU cores
    ))
])

# Train both models
model1.fit(X_train, log_y1_train)
model2.fit(X_train, log_y2_train)

# Make predictions (log scale)
y1_pred_train = model1.predict(X_train)
y1_pred_test = model1.predict(X_test)
y2_pred_train = model2.predict(X_train)
y2_pred_test = model2.predict(X_test)

# Evaluate MD model (log scale)
print("\n=== MD MODEL PERFORMANCE (LOG SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(log_y1_train, y1_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y1_train, y1_pred_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y1_train, y1_pred_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y1_train, y1_pred_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(log_y1_test, y1_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y1_test, y1_pred_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y1_test, y1_pred_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y1_test, y1_pred_test)):.4f}")

# Evaluate TD model (log scale)
print("\n=== TD MODEL PERFORMANCE (LOG SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(log_y2_train, y2_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y2_train, y2_pred_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y2_train, y2_pred_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y2_train, y2_pred_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(log_y2_test, y2_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(log_y2_test, y2_pred_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(log_y2_test, y2_pred_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(log_y2_test, y2_pred_test)):.4f}")

# Inverse transform predictions and targets to original scale
orig_y1_train = np.exp(log_y1_train)
orig_y1_test = np.exp(log_y1_test)
orig_pred1_train = np.exp(y1_pred_train)
orig_pred1_test = np.exp(y1_pred_test)

orig_y2_train = np.exp(log_y2_train)
orig_y2_test = np.exp(log_y2_test)
orig_pred2_train = np.exp(y2_pred_train)
orig_pred2_test = np.exp(y2_pred_test)

# Evaluate MD model (original scale)
print("\n=== MD MODEL PERFORMANCE (ORIGINAL SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(orig_y1_train, orig_pred1_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y1_train, orig_pred1_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y1_train, orig_pred1_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(orig_y1_test, orig_pred1_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y1_test, orig_pred1_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y1_test, orig_pred1_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test)):.4f}")

# Evaluate TD model (original scale)
print("\n=== TD MODEL PERFORMANCE (ORIGINAL SCALE) ===")
print("Training Set:")
print(f"R² Score: {r2_score(orig_y2_train, orig_pred2_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y2_train, orig_pred2_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y2_train, orig_pred2_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y2_train, orig_pred2_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(orig_y2_test, orig_pred2_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(orig_y2_test, orig_pred2_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(orig_y2_test, orig_pred2_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(orig_y2_test, orig_pred2_test)):.4f}")

# Feature importance analysis for both models
print("\n=== FEATURE IMPORTANCE ===")
feature_names = []
if hasattr(model1.named_steps['preprocessor'], 'get_feature_names_out'):
    feature_names = model1.named_steps['preprocessor'].get_feature_names_out()
else:
    # Fallback for older sklearn versions
    feature_names = [f"feature_{i}" for i in range(len(model1.named_steps['regressor'].feature_importances_))]

importances1 = model1.named_steps['regressor'].feature_importances_
importances2 = model2.named_steps['regressor'].feature_importances_
indices1 = np.argsort(importances1)[::-1]
indices2 = np.argsort(importances2)[::-1]

print("Top 30 most important features for MD (property1):")
for i in range(min(30, len(indices1))):
    print(f"{i+1:2d}. {feature_names[indices1[i]]}: {importances1[indices1[i]]:.4f}")

print("\nTop 30 most important features for TD (property2):")
for i in range(min(30, len(indices2))):
    print(f"{i+1:2d}. {feature_names[indices2[i]]}: {importances2[indices2[i]]:.4f}")

# Create visualizations for MD (property1)
plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted plot for MD
plt.subplot(3, 3, 1)
plt.scatter(orig_y1_train, orig_pred1_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(orig_y1_train, orig_pred1_train):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train)):.2f})', color='blue')
plt.scatter(orig_y1_test, orig_pred1_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(orig_y1_test, orig_pred1_test):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test)):.2f})', color='red')
plt.plot([orig_y1_train.min(), orig_y1_train.max()], [orig_y1_train.min(), orig_y1_train.max()], 'k--', lw=2)
plt.xlabel('Actual MD Tensile Strength (MPa)')
plt.ylabel('Predicted MD Tensile Strength (MPa)')
plt.title('MD Tensile Strength Predictions')
plt.legend()

# 2. Residuals plot for MD
plt.subplot(3, 3, 2)
residuals1_train = orig_y1_train - orig_pred1_train
residuals1_test = orig_y1_test - orig_pred1_test
plt.scatter(orig_pred1_train, residuals1_train, alpha=0.6, label='Training', color='blue')
plt.scatter(orig_pred1_test, residuals1_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted MD Tensile Strength (MPa)')
plt.ylabel('Residuals')
plt.title('MD Residuals')
plt.legend()

# 3. Feature importance plot for MD
plt.subplot(3, 3, 3)
top_features = 20
top_indices1 = indices1[:top_features]
plt.barh(range(top_features), importances1[top_indices1])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices1])
plt.xlabel('Feature Importance')
plt.title('MD Feature Importance')
plt.gca().invert_yaxis()

# 4. Distribution of MD target variable
plt.subplot(3, 3, 4)
plt.hist(orig_y1_train, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('MD Tensile Strength (MPa)')
plt.ylabel('Frequency')
plt.title('MD Target Distribution')

# 5. Training vs Test performance comparison for MD
plt.subplot(3, 3, 5)
metrics = ['R²', 'MAE', 'RMSE']
train_scores1 = [
    r2_score(orig_y1_train, orig_pred1_train),
    mean_absolute_error(orig_y1_train, orig_pred1_train),
    np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train))
]
test_scores1 = [
    r2_score(orig_y1_test, orig_pred1_test),
    mean_absolute_error(orig_y1_test, orig_pred1_test),
    np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test))
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores1, width, label='Training', color='blue', alpha=0.7)
plt.bar(x + width/2, test_scores1, width, label='Test', color='red', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('MD Performance Comparison')
plt.xticks(x, metrics)
plt.legend()

# 6. Prediction error distribution for MD
plt.subplot(3, 3, 6)
plt.hist(residuals1_train, bins=15, alpha=0.7, label='Training', color='blue')
plt.hist(residuals1_test, bins=15, alpha=0.7, label='Test', color='red')
plt.xlabel('Prediction Error (MPa)')
plt.ylabel('Frequency')
plt.title('MD Error Distribution')
plt.legend()

# 7. Feature importance by category for MD
plt.subplot(3, 3, 7)
def get_importance_sum(features, feature_names, importances):
    """Calculate importance sum for features containing a specific string"""
    total_importance = 0
    for i, feature in enumerate(feature_names):
        if any(f in feature for f in features):
            total_importance += importances[i]
    return total_importance

# Calculate importance by category for MD
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

category_importance1 = {}
for category, features in categories.items():
    category_importance1[category] = get_importance_sum(features, feature_names, importances1)

# Plot category importance for MD
plt.bar(category_importance1.keys(), category_importance1.values(), color='skyblue')
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title('MD Category Importance')
plt.xticks(rotation=45)

# 8. Model complexity analysis for MD
plt.subplot(3, 3, 8)
n_trees = model1.named_steps['regressor'].n_estimators
max_depth = model1.named_steps['regressor'].max_depth
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
plt.title('MD Model Complexity')
plt.ylabel('Value')
plt.xticks(rotation=45)

# Add text annotation for unlimited depth if applicable
if max_depth is None:
    plt.text(1, max_depth_value + 1, 'Unlimited', ha='center', va='bottom', fontweight='bold')

# 9. Data summary for MD
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'Total Features: {len(feature_names)}', fontsize=12)
plt.text(0.1, 0.7, f'Categorical: {len(categorical_features)}', fontsize=12)
plt.text(0.1, 0.6, f'Numerical: {len(numerical_features)}', fontsize=12)
plt.text(0.1, 0.5, f'Training Samples: {len(orig_y1_train)}', fontsize=12)
plt.text(0.1, 0.4, f'Test Samples: {len(orig_y1_test)}', fontsize=12)
plt.text(0.1, 0.3, f'Target Range: {orig_y1_train.min():.1f} - {orig_y1_train.max():.1f} MPa', fontsize=12)
plt.text(0.1, 0.2, f'Target Mean: {orig_y1_train.mean():.1f} MPa', fontsize=12)
plt.text(0.1, 0.1, f'Target Std: {orig_y1_train.std():.1f} MPa', fontsize=12)
plt.title('MD Data Summary')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results_MD.png'), dpi=300, bbox_inches='tight')
plt.show()

# Error-to-noise analysis for test set (original scale) - MD
data_std1 = np.std(orig_y1_test)
data_mean1 = np.mean(orig_y1_test)
data_cv1 = data_std1 / data_mean1  # Coefficient of variation
relative_errors1 = np.abs(orig_y1_test - orig_pred1_test) / orig_y1_test
relative_mae1 = np.mean(relative_errors1)
relative_rmse1 = np.sqrt(np.mean(relative_errors1**2))
error_to_std_ratio1 = relative_mae1 / data_cv1
rmse_to_std_ratio1 = relative_rmse1 / data_cv1

# Error-to-noise analysis for test set (original scale) - TD
data_std2 = np.std(orig_y2_test)
data_mean2 = np.mean(orig_y2_test)
data_cv2 = data_std2 / data_mean2  # Coefficient of variation
relative_errors2 = np.abs(orig_y2_test - orig_pred2_test) / orig_y2_test
relative_mae2 = np.mean(relative_errors2)
relative_rmse2 = np.sqrt(np.mean(relative_errors2**2))
error_to_std_ratio2 = relative_mae2 / data_cv2
rmse_to_std_ratio2 = relative_rmse2 / data_cv2

print(f"\n=== ERROR vs DATA VARIABILITY ANALYSIS ===")
print(f"MD Data Statistics:")
print(f"  Mean TS: {data_mean1:.2f}")
print(f"  Std TS: {data_std1:.2f}")
print(f"  Coefficient of Variation (CV): {data_cv1:.3f} ({data_cv1*100:.1f}%)")
print(f"\nMD Error Metrics:")
print(f"  Relative MAE: {relative_mae1:.3f} ({relative_mae1*100:.1f}%)")
print(f"  Relative RMSE: {relative_rmse1:.3f} ({relative_rmse1*100:.1f}%)")
print(f"\nMD Error-to-Noise Ratios:")
print(f"  Relative MAE / CV: {error_to_std_ratio1:.3f}")
print(f"  Relative RMSE / CV: {rmse_to_std_ratio1:.3f}")

print(f"\nTD Data Statistics:")
print(f"  Mean TS: {data_mean2:.2f}")
print(f"  Std TS: {data_std2:.2f}")
print(f"  Coefficient of Variation (CV): {data_cv2:.3f} ({data_cv2*100:.1f}%)")
print(f"\nTD Error Metrics:")
print(f"  Relative MAE: {relative_mae2:.3f} ({relative_mae2*100:.1f}%)")
print(f"  Relative RMSE: {relative_rmse2:.3f} ({relative_rmse2*100:.1f}%)")
print(f"\nTD Error-to-Noise Ratios:")
print(f"  Relative MAE / CV: {error_to_std_ratio2:.3f}")
print(f"  Relative RMSE / CV: {rmse_to_std_ratio2:.3f}")

# Last 4 blends performance for MD
print("\n=== LAST 4 BLENDS PERFORMANCE (MD) ===")
last_4_X = X.iloc[last_4_indices]
last_4_log_y1 = log_y1.iloc[last_4_indices]
last_4_pred_log1 = model1.predict(last_4_X)
last_4_mae_log1 = mean_absolute_error(last_4_log_y1, last_4_pred_log1)
last_4_rmse_log1 = np.sqrt(mean_squared_error(last_4_log_y1, last_4_pred_log1))
last_4_r2_log1 = r2_score(last_4_log_y1, last_4_pred_log1)
print(f"Last 4 Blends MD (Log Scale):")
print(f"R² Score: {last_4_r2_log1:.4f}")
print(f"Mean Absolute Error: {last_4_mae_log1:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse_log1:.4f}")
# Original scale
last_4_actual1 = np.exp(last_4_log_y1)
last_4_pred1 = np.exp(last_4_pred_log1)
last_4_mae1 = mean_absolute_error(last_4_actual1, last_4_pred1)
last_4_rmse1 = np.sqrt(mean_squared_error(last_4_actual1, last_4_pred1))
last_4_r21 = r2_score(last_4_actual1, last_4_pred1)
print(f"Last 4 Blends MD (Original Scale):")
print(f"R² Score: {last_4_r21:.4f}")
print(f"Mean Absolute Error: {last_4_mae1:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse1:.4f}")

# Last 4 blends performance for TD
print("\n=== LAST 4 BLENDS PERFORMANCE (TD) ===")
last_4_log_y2 = log_y2.iloc[last_4_indices]
last_4_pred_log2 = model2.predict(last_4_X)
last_4_mae_log2 = mean_absolute_error(last_4_log_y2, last_4_pred_log2)
last_4_rmse_log2 = np.sqrt(mean_squared_error(last_4_log_y2, last_4_pred_log2))
last_4_r2_log2 = r2_score(last_4_log_y2, last_4_pred_log2)
print(f"Last 4 Blends TD (Log Scale):")
print(f"R² Score: {last_4_r2_log2:.4f}")
print(f"Mean Absolute Error: {last_4_mae_log2:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse_log2:.4f}")
# Original scale
last_4_actual2 = np.exp(last_4_log_y2)
last_4_pred2 = np.exp(last_4_pred_log2)
last_4_mae2 = mean_absolute_error(last_4_actual2, last_4_pred2)
last_4_rmse2 = np.sqrt(mean_squared_error(last_4_actual2, last_4_pred2))
last_4_r22 = r2_score(last_4_actual2, last_4_pred2)
print(f"Last 4 Blends TD (Original Scale):")
print(f"R² Score: {last_4_r22:.4f}")
print(f"Mean Absolute Error: {last_4_mae2:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse2:.4f}")

# Create separate plot for last 4 blends performance - MD
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for MD
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y1, last_4_pred_log1, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y1.min(), last_4_log_y1.max()], [last_4_log_y1.min(), last_4_log_y1.max()], 'k--', lw=2)
plt.xlabel('Actual Log(MD Tensile Strength)')
plt.ylabel('Predicted Log(MD Tensile Strength)')
plt.title('MD Tensile Strength Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae_log1:.2f}\n'
    f'R²: {last_4_r2_log1:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_log_y1, last_4_pred_log1)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions for MD
plt.subplot(1, 2, 2)
# Calculate relative errors
relative_errors1 = np.abs(last_4_actual1 - last_4_pred1) / last_4_actual1
relative_mae1 = np.mean(relative_errors1)
relative_rmse1 = np.sqrt(np.mean(relative_errors1**2))

# Calculate data statistics
data_std1 = np.std(last_4_actual1)
data_mean1 = np.mean(last_4_actual1)
data_cv1 = data_std1 / data_mean1  # Coefficient of variation

# Calculate error-to-noise ratio
error_to_std_ratio1 = relative_mae1 / data_cv1

plt.scatter(last_4_actual1, last_4_pred1, color='blue', s=100, alpha=0.7)
plt.plot([last_4_actual1.min(), last_4_actual1.max()], [last_4_actual1.min(), last_4_actual1.max()], 'r--', lw=2)
plt.xlabel('Actual MD Tensile Strength (Original Scale)')
plt.ylabel('Predicted MD Tensile Strength (Original Scale)')
plt.title('MD Tensile Strength Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae1:.2f}\n'
    f'R²: {last_4_r21:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_actual1, last_4_pred1)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_MD.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nLast 4 Blends MD - Original Scale Performance:")
print(f"R² Score: {last_4_r21:.4f}")
print(f"Mean Absolute Error: {last_4_mae1:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse1:.4f}")
print(f"Relative MAE: {relative_mae1:.4f} ({relative_mae1*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse1:.4f} ({relative_rmse1*100:.2f}%)")
print("✅ Last 4 blends MD performance plot saved as 'last_4_blends_performance_MD.png'")

# Create separate plot for last 4 blends performance - TD
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for TD
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y2, last_4_pred_log2, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y2.min(), last_4_log_y2.max()], [last_4_log_y2.min(), last_4_log_y2.max()], 'k--', lw=2)
plt.xlabel('Actual Log(TD Tensile Strength)')
plt.ylabel('Predicted Log(TD Tensile Strength)')
plt.title('TD Tensile Strength Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae_log2:.2f}\n'
    f'R²: {last_4_r2_log2:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_log_y2, last_4_pred_log2)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions for TD
plt.subplot(1, 2, 2)
# Calculate relative errors
relative_errors2 = np.abs(last_4_actual2 - last_4_pred2) / last_4_actual2
relative_mae2 = np.mean(relative_errors2)
relative_rmse2 = np.sqrt(np.mean(relative_errors2**2))

# Calculate data statistics
data_std2 = np.std(last_4_actual2)
data_mean2 = np.mean(last_4_actual2)
data_cv2 = data_std2 / data_mean2  # Coefficient of variation

# Calculate error-to-noise ratio
error_to_std_ratio2 = relative_mae2 / data_cv2

plt.scatter(last_4_actual2, last_4_pred2, color='blue', s=100, alpha=0.7)
plt.plot([last_4_actual2.min(), last_4_actual2.max()], [last_4_actual2.min(), last_4_actual2.max()], 'r--', lw=2)
plt.xlabel('Actual TD Tensile Strength (Original Scale)')
plt.ylabel('Predicted TD Tensile Strength (Original Scale)')
plt.title('TD Tensile Strength Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae2:.2f}\n'
    f'R²: {last_4_r22:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_actual2, last_4_pred2)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_TD.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nLast 4 Blends TD - Original Scale Performance:")
print(f"R² Score: {last_4_r22:.4f}")
print(f"Mean Absolute Error: {last_4_mae2:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse2:.4f}")
print(f"Relative MAE: {relative_mae2:.4f} ({relative_mae2*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse2:.4f} ({relative_rmse2*100:.2f}%)")
print("✅ Last 4 blends TD performance plot saved as 'last_4_blends_performance_TD.png'")

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae1:.2f}\n'
    f'R²: {last_4_r21:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_actual1, last_4_pred1)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_MD.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nLast 4 Blends MD - Original Scale Performance:")
print(f"R² Score: {last_4_r21:.4f}")
print(f"Mean Absolute Error: {last_4_mae1:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse1:.4f}")
print(f"Relative MAE: {relative_mae1:.4f} ({relative_mae1*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse1:.4f} ({relative_rmse1*100:.2f}%)")
print("✅ Last 4 blends MD performance plot saved as 'last_4_blends_performance_MD.png'")



# Save the models
import joblib
os.makedirs(args.output, exist_ok=True)
model1_path = os.path.join(args.output, 'comprehensive_polymer_model_MD.pkl')
model2_path = os.path.join(args.output, 'comprehensive_polymer_model_TD.pkl')
joblib.dump(model1, model1_path)
joblib.dump(model2, model2_path)
print(f"\nMD Model saved as '{model1_path}'")
print(f"TD Model saved as '{model2_path}'")

# Create comprehensive visualization for MD (property1)
plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted plot for MD
plt.subplot(3, 3, 1)
plt.scatter(orig_y1_train, orig_pred1_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(orig_y1_train, orig_pred1_train):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train)):.2f})', color='blue')
plt.scatter(orig_y1_test, orig_pred1_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(orig_y1_test, orig_pred1_test):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test)):.2f})', color='red')
plt.plot([orig_y1_train.min(), orig_y1_train.max()], [orig_y1_train.min(), orig_y1_train.max()], 'k--', lw=2)
plt.xlabel('Actual MD Tensile Strength (MPa)')
plt.ylabel('Predicted MD Tensile Strength (MPa)')
plt.title('MD Tensile Strength Predictions')
plt.legend()

# 2. Residuals plot for MD
plt.subplot(3, 3, 2)
residuals1_train = orig_y1_train - orig_pred1_train
residuals1_test = orig_y1_test - orig_pred1_test
plt.scatter(orig_pred1_train, residuals1_train, alpha=0.6, label='Training', color='blue')
plt.scatter(orig_pred1_test, residuals1_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted MD Tensile Strength (MPa)')
plt.ylabel('Residuals')
plt.title('MD Residuals')
plt.legend()

# 3. Feature importance plot for MD
plt.subplot(3, 3, 3)
top_features = 20
top_indices1 = indices1[:top_features]
plt.barh(range(top_features), importances1[top_indices1])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices1])
plt.xlabel('Feature Importance')
plt.title('MD Feature Importance')
plt.gca().invert_yaxis()

# 4. Distribution of MD target variable
plt.subplot(3, 3, 4)
plt.hist(orig_y1_train, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('MD Tensile Strength (MPa)')
plt.ylabel('Frequency')
plt.title('MD Target Distribution')

# 5. Training vs Test performance comparison for MD
plt.subplot(3, 3, 5)
metrics = ['R²', 'MAE', 'RMSE']
train_scores1 = [
    r2_score(orig_y1_train, orig_pred1_train),
    mean_absolute_error(orig_y1_train, orig_pred1_train),
    np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train))
]
test_scores1 = [
    r2_score(orig_y1_test, orig_pred1_test),
    mean_absolute_error(orig_y1_test, orig_pred1_test),
    np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test))
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores1, width, label='Training', color='blue', alpha=0.7)
plt.bar(x + width/2, test_scores1, width, label='Test', color='red', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('MD Performance Comparison')
plt.xticks(x, metrics)
plt.legend()

# 6. Prediction error distribution for MD
plt.subplot(3, 3, 6)
plt.hist(residuals1_train, bins=15, alpha=0.7, label='Training', color='blue')
plt.hist(residuals1_test, bins=15, alpha=0.7, label='Test', color='red')
plt.xlabel('Prediction Error (MPa)')
plt.ylabel('Frequency')
plt.title('MD Error Distribution')
plt.legend()

# 7. Feature importance by category for MD
plt.subplot(3, 3, 7)
category_importance1 = {}
for category, features in categories.items():
    category_importance1[category] = get_importance_sum(features, feature_names, importances1)

# Plot category importance for MD
plt.bar(category_importance1.keys(), category_importance1.values(), color='skyblue')
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title('MD Category Importance')
plt.xticks(rotation=45)

# 8. Model complexity analysis for MD
plt.subplot(3, 3, 8)
n_trees = model1.named_steps['regressor'].n_estimators
max_depth = model1.named_steps['regressor'].max_depth
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
plt.title('MD Model Complexity')
plt.ylabel('Value')
plt.xticks(rotation=45)

# Add text annotation for unlimited depth if applicable
if max_depth is None:
    plt.text(1, max_depth_value + 1, 'Unlimited', ha='center', va='bottom', fontweight='bold')

# 9. Data summary for MD
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'Total Features: {len(feature_names)}', fontsize=12)
plt.text(0.1, 0.7, f'Categorical: {len(categorical_features)}', fontsize=12)
plt.text(0.1, 0.6, f'Numerical: {len(numerical_features)}', fontsize=12)
plt.text(0.1, 0.5, f'Training Samples: {len(orig_y1_train)}', fontsize=12)
plt.text(0.1, 0.4, f'Test Samples: {len(orig_y1_test)}', fontsize=12)
plt.text(0.1, 0.3, f'Target Range: {orig_y1_train.min():.1f} - {orig_y1_train.max():.1f} MPa', fontsize=12)
plt.text(0.1, 0.2, f'Target Mean: {orig_y1_train.mean():.1f} MPa', fontsize=12)
plt.text(0.1, 0.1, f'Target Std: {orig_y1_train.std():.1f} MPa', fontsize=12)
plt.title('MD Data Summary')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results_MD.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create comprehensive visualization for TD (property2)
plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted plot for TD
plt.subplot(3, 3, 1)
plt.scatter(orig_y2_train, orig_pred2_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(orig_y2_train, orig_pred2_train):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y2_train, orig_pred2_train)):.2f})', color='blue')
plt.scatter(orig_y2_test, orig_pred2_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(orig_y2_test, orig_pred2_test):.2f}, RMSE={np.sqrt(mean_squared_error(orig_y2_test, orig_pred2_test)):.2f})', color='red')
plt.plot([orig_y2_train.min(), orig_y2_train.max()], [orig_y2_train.min(), orig_y2_train.max()], 'k--', lw=2)
plt.xlabel('Actual TD Tensile Strength (MPa)')
plt.ylabel('Predicted TD Tensile Strength (MPa)')
plt.title('TD Tensile Strength Predictions')
plt.legend()

# 2. Residuals plot for TD
plt.subplot(3, 3, 2)
residuals2_train = orig_y2_train - orig_pred2_train
residuals2_test = orig_y2_test - orig_pred2_test
plt.scatter(orig_pred2_train, residuals2_train, alpha=0.6, label='Training', color='blue')
plt.scatter(orig_pred2_test, residuals2_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted TD Tensile Strength (MPa)')
plt.ylabel('Residuals')
plt.title('TD Residuals')
plt.legend()

# 3. Feature importance plot for TD
plt.subplot(3, 3, 3)
top_features = 20
top_indices2 = indices2[:top_features]
plt.barh(range(top_features), importances2[top_indices2])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices2])
plt.xlabel('Feature Importance')
plt.title('TD Feature Importance')
plt.gca().invert_yaxis()

# 4. Distribution of TD target variable
plt.subplot(3, 3, 4)
plt.hist(orig_y2_train, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('TD Tensile Strength (MPa)')
plt.ylabel('Frequency')
plt.title('TD Target Distribution')

# 5. Training vs Test performance comparison for TD
plt.subplot(3, 3, 5)
metrics = ['R²', 'MAE', 'RMSE']
train_scores2 = [
    r2_score(orig_y2_train, orig_pred2_train),
    mean_absolute_error(orig_y2_train, orig_pred2_train),
    np.sqrt(mean_squared_error(orig_y2_train, orig_pred2_train))
]
test_scores2 = [
    r2_score(orig_y2_test, orig_pred2_test),
    mean_absolute_error(orig_y2_test, orig_pred2_test),
    np.sqrt(mean_squared_error(orig_y2_test, orig_pred2_test))
]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores2, width, label='Training', color='blue', alpha=0.7)
plt.bar(x + width/2, test_scores2, width, label='Test', color='red', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('TD Performance Comparison')
plt.xticks(x, metrics)
plt.legend()

# 6. Prediction error distribution for TD
plt.subplot(3, 3, 6)
plt.hist(residuals2_train, bins=15, alpha=0.7, label='Training', color='blue')
plt.hist(residuals2_test, bins=15, alpha=0.7, label='Test', color='red')
plt.xlabel('Prediction Error (MPa)')
plt.ylabel('Frequency')
plt.title('TD Error Distribution')
plt.legend()

# 7. Feature importance by category for TD
plt.subplot(3, 3, 7)
category_importance2 = {}
for category, features in categories.items():
    category_importance2[category] = get_importance_sum(features, feature_names, importances2)

# Plot category importance for TD
plt.bar(category_importance2.keys(), category_importance2.values(), color='skyblue')
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title('TD Category Importance')
plt.xticks(rotation=45)

# 8. Model complexity analysis for TD
plt.subplot(3, 3, 8)
n_trees = model2.named_steps['regressor'].n_estimators
max_depth = model2.named_steps['regressor'].max_depth
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
plt.title('TD Model Complexity')
plt.ylabel('Value')
plt.xticks(rotation=45)

# Add text annotation for unlimited depth if applicable
if max_depth is None:
    plt.text(1, max_depth_value + 1, 'Unlimited', ha='center', va='bottom', fontweight='bold')

# 9. Data summary for TD
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'Total Features: {len(feature_names)}', fontsize=12)
plt.text(0.1, 0.7, f'Categorical: {len(categorical_features)}', fontsize=12)
plt.text(0.1, 0.6, f'Numerical: {len(numerical_features)}', fontsize=12)
plt.text(0.1, 0.5, f'Training Samples: {len(orig_y2_train)}', fontsize=12)
plt.text(0.1, 0.4, f'Test Samples: {len(orig_y2_test)}', fontsize=12)
plt.text(0.1, 0.3, f'Target Range: {orig_y2_train.min():.1f} - {orig_y2_train.max():.1f} MPa', fontsize=12)
plt.text(0.1, 0.2, f'Target Mean: {orig_y2_train.mean():.1f} MPa', fontsize=12)
plt.text(0.1, 0.1, f'Target Std: {orig_y2_train.std():.1f} MPa', fontsize=12)
plt.title('TD Data Summary')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results_TD.png'), dpi=300, bbox_inches='tight')
plt.show()

# Example prediction
print("\n=== EXAMPLE PREDICTION ===")
example_data = X.iloc[0:1].copy()
actual_value1_log = log_y1.iloc[0]
actual_value2_log = log_y2.iloc[0]
predicted_value1_log = model1.predict(example_data)[0]
predicted_value2_log = model2.predict(example_data)[0]
print(f"Actual Log(MD TS): {actual_value1_log:.2f}")
print(f"Predicted Log(MD TS): {predicted_value1_log:.2f}")
print(f"MD Prediction Error: {abs(actual_value1_log - predicted_value1_log):.2f}")
print(f"Actual Log(TD TS): {actual_value2_log:.2f}")
print(f"Predicted Log(TD TS): {predicted_value2_log:.2f}")
print(f"TD Prediction Error: {abs(actual_value2_log - predicted_value2_log):.2f}")
# Convert back to original scale for comparison
original_actual1 = np.exp(actual_value1_log)
original_predicted1 = np.exp(predicted_value1_log)
original_actual2 = np.exp(actual_value2_log)
original_predicted2 = np.exp(predicted_value2_log)
print(f"Actual MD TS (original scale): {original_actual1:.2f}")
print(f"Predicted MD TS (original scale): {original_predicted1:.2f}")
print(f"MD Original Scale Error: {abs(original_actual1 - original_predicted1):.2f}")
print(f"Actual TD TS (original scale): {original_actual2:.2f}")
print(f"Predicted TD TS (original scale): {original_predicted2:.2f}")
print(f"TD Original Scale Error: {abs(original_actual2 - original_predicted2):.2f}")

# Prediction function for both properties
def predict_tensile_strength(input_data):
    """
    Predict both MD and TD tensile strength for polymer blend data
    Parameters:
    - input_data: pandas DataFrame with all features (excluding SMILES columns)
    Returns:
    - tuple: (md_prediction, td_prediction) in MPa
    """
    # Handle missing values
    for col in categorical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('Unknown')
    for col in numerical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna(0)
    prediction1_log = model1.predict(input_data)[0]
    prediction2_log = model2.predict(input_data)[0]
    return np.exp(prediction1_log), np.exp(prediction2_log)

print("\n=== MODEL SUMMARY ===")
print("✅ Two Random Forest models successfully trained (MD and TD)!")
print("✅ All features included (categorical + numerical + featurized + thickness)")
print("✅ One-hot encoding applied to categorical features")
print("✅ Models allowed to overfit as requested")
print("✅ Feature importance analysis completed for both models")
print("✅ Comprehensive visualizations saved for both MD and TD")
print(f"✅ MD Model saved as '{model1_path}'")
print(f"✅ TD Model saved as '{model2_path}'")
print(f"✅ All files saved to: {args.output}")
print("✅ Prediction function created for new data (returns both MD and TD)")

# Save models
model1_path = os.path.join(args.output, 'comprehensive_polymer_model_MD.pkl')
model2_path = os.path.join(args.output, 'comprehensive_polymer_model_TD.pkl')

import joblib
joblib.dump(model1, model1_path)
joblib.dump(model2, model2_path)

print(f"✅ MD Model saved as '{model1_path}'")
print(f"✅ TD Model saved as '{model2_path}'")

# Save training explanation
with open(os.path.join(args.output, 'training_explanation.txt'), 'w') as f:
    f.write("Random Forest Model Training Explanation (TS v5 with Thickness)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} samples with thickness information\n")
    f.write(f"Features: {len(feature_names)} (including thickness)\n")
    f.write(f"Training set: {len(orig_y1_train)} samples (including last 4 blends oversampled 10x)\n")
    f.write(f"Test set: {len(orig_y1_test)} samples\n\n")
    
    f.write("Model Configuration:\n")
    f.write(f"- Algorithm: Random Forest\n")
    f.write(f"- Estimators: 200\n")
    f.write(f"- Max depth: Unlimited\n")
    f.write(f"- Min samples split: 2\n")
    f.write(f"- Min samples leaf: 1\n\n")
    
    f.write("Training Set Performance:\n")
    f.write(f"- R² Score (log scale): {r2_score(log_y1_train, y1_pred_train):.4f}\n")
    f.write(f"- R² Score (original scale): {r2_score(orig_y1_train, orig_pred1_train):.4f}\n")
    f.write(f"- RMSE (original scale): {np.sqrt(mean_squared_error(orig_y1_train, orig_pred1_train)):.4f} MPa\n")
    f.write(f"- MAE (original scale): {mean_absolute_error(orig_y1_train, orig_pred1_train):.4f} MPa\n\n")
    
    f.write("Test Set Performance:\n")
    f.write(f"- R² Score (log scale): {r2_score(log_y1_test, y1_pred_test):.4f}\n")
    f.write(f"- R² Score (original scale): {r2_score(orig_y1_test, orig_pred1_test):.4f}\n")
    f.write(f"- RMSE (original scale): {np.sqrt(mean_squared_error(orig_y1_test, orig_pred1_test)):.4f} MPa\n")
    f.write(f"- MAE (original scale): {mean_absolute_error(orig_y1_test, orig_pred1_test):.4f} MPa\n\n")
    
    f.write("Last 4 Blends Performance:\n")
    f.write(f"- R² Score (log scale): {last_4_r2_log1:.4f}\n")
    f.write(f"- R² Score (original scale): {last_4_r21:.4f}\n")
    f.write(f"- RMSE (original scale): {last_4_rmse1:.4f} MPa\n")
    f.write(f"- MAE (original scale): {last_4_mae1:.4f} MPa\n")
    f.write(f"- Relative MAE: {relative_mae1:.4f} ({relative_mae1*100:.2f}%)\n")
    f.write(f"- Relative RMSE: {relative_rmse1:.4f} ({relative_rmse1*100:.2f}%)\n\n")
    
    f.write("Top 10 Feature Importances (MD):\n")
    indices1 = np.argsort(importances1)[::-1]
    for i in range(min(10, len(indices1))):
        f.write(f"{i+1}. {feature_names[indices1[i]]}: {importances1[indices1[i]]:.4f}\n")
    
    f.write(f"\nThickness Information:\n")
    f.write(f"- Thickness values in dataset: {thickness_count}\n")
    f.write(f"- Thickness range: {thickness_range}\n")
    if thickness_col in X.columns:
        train_thickness = X_train[thickness_col].unique()
        f.write(f"- Thickness values in training set: {train_thickness}\n")
    else:
        f.write(f"- Thickness values in training set: Not available\n")
    # Find thickness importance with correct prefix
    thickness_feature_name = None
    for feature_name in feature_names:
        if 'thickness' in feature_name.lower():
            thickness_feature_name = feature_name
            break
    if thickness_feature_name:
        thickness_importance = importances1[list(feature_names).index(thickness_feature_name)]
        f.write(f"- Thickness importance: {thickness_importance:.4f}\n")
    else:
        f.write(f"- Thickness importance: Not found in feature names\n")

print(f"✅ Training explanation saved to {os.path.join(args.output, 'training_explanation.txt')}") 