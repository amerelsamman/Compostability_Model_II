import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
import os
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train XGBoost model for Cobb polymer blends')
parser.add_argument('--input', type=str, default='data/cobb/polymer_blends_for_ml_featurized.csv',
                    help='Input CSV file path (default: data/cobb/polymer_blends_for_ml_featurized.csv)')
parser.add_argument('--output', type=str, default='../models/cobb/v1',
                    help='Output directory path (default: ../models/cobb/v1)')
parser.add_argument('--target', type=str, default='property',
                    help='Target column name (default: property)')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

print(f"Loading featurized polymer blends data from: {args.input}")
df = pd.read_csv(args.input)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Remove rows where the property value is zero
initial_shape = df.shape
zero_rows = df[df[args.target] == 0].shape[0]
df = df[df[args.target] != 0].reset_index(drop=True)
print(f"Removed {zero_rows} rows with property == 0. New shape: {df.shape} (was {initial_shape})")

# Separate features and target
target = args.target
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
# EXCLUDE Materials column from features (v4 change)
excluded_cols = [target] + smiles_cols + ['Materials']
X = df.drop(columns=excluded_cols)
y = df[target]

# Apply log transformation to target values
y = np.log(y + 1e-10)  # log(x + small_offset) to handle zero values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Log-transformed target range: {y.min():.4f} to {y.max():.4f}")

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

# Split data ensuring last 1 blend is in testing, last 9 blends in training
print("Splitting data ensuring last 1 blend in testing, last 9 blends in training...")
last_10_indices = list(range(len(df) - 10, len(df)))
last_9_indices = list(range(len(df) - 10, len(df) - 1))  # Last 9 blends
last_1_index = [len(df) - 1]  # Last 1 blend

# Create masks for train/test split
train_mask = np.ones(len(df), dtype=bool)
test_mask = np.zeros(len(df), dtype=bool)

# Set last 9 blends to training
train_mask[last_9_indices] = True
train_mask[last_1_index] = False

# Set last 1 blend to testing
test_mask[last_1_index] = True

# Use train_test_split on the remaining data (excluding last 10)
remaining_indices = list(range(len(df) - 10))
if len(remaining_indices) > 0:
    # Split remaining data with 80/20 ratio
    remaining_train_indices, remaining_test_indices = train_test_split(
        remaining_indices, 
        test_size=0.2, 
        random_state=42
    )
    
    # Add remaining train indices to training mask
    train_mask[remaining_train_indices] = True
    # Add remaining test indices to testing mask
    test_mask[remaining_test_indices] = True

# Apply masks to get final train/test split
X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print(f"Last 10 blend indices: {last_10_indices}")
print(f"Last 9 blend indices (training): {last_9_indices}")
print(f"Last 1 blend index (testing): {last_1_index}")

print(f"Training set size: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)")
print(f"Last 9 blends in training: {all(i in np.where(train_mask)[0] for i in last_9_indices)}")
print(f"Last 1 blend in testing: {all(i in np.where(test_mask)[0] for i in last_1_index)}")

print("\nTraining the XGBoost model for Cobb prediction...")

# Create pipeline with XGBoost parameters optimized for 
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=80,       # Moderate number of trees
        max_depth=5,           # Higher depth for better 
        learning_rate=0.1,     # Higher learning rate for faster convergence
        subsample=0.8,         # Higher subsample ratio
        colsample_bytree=0.8,  # Higher column sample ratio
        reg_alpha=0.2,         # Lower L1 regularization
        reg_lambda=2.0,        # Lower L2 regularization
        min_child_weight=1,    # Lower minimum child weight
        gamma=0.0,             # No minimum loss reduction requirement
        random_state=42,
        n_jobs=-1
    ))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
print("\n=== MODEL PERFORMANCE ===")
print("Training Set:")
print(f"R² Score: {r2_score(y_train, y_pred_train):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")

print("\nTest Set:")
print(f"R² Score: {r2_score(y_test, y_pred_test):.4f}")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_test):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred_test):.4f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")

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
fig = plt.figure(figsize=(20, 15))
plt.suptitle('', fontsize=16)

# 1. Actual vs Predicted plot
plt.subplot(3, 3, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(y_train, y_pred_train):.2f}, RMSE={np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f})', color='blue')
plt.scatter(y_test, y_pred_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(y_test, y_pred_test):.2f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f})', color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Residuals plot
plt.subplot(3, 3, 2)
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test
plt.scatter(y_pred_train, residuals_train, alpha=0.6, label='Training', color='blue')
plt.scatter(y_pred_test, residuals_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Log(Property)')
plt.ylabel('Residuals')
plt.title('')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Feature importance plot
plt.subplot(3, 3, 3)
top_features = 20
top_indices = np.argsort(importances)[-top_features:]
plt.barh(range(top_features), importances[top_indices])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance')
plt.title(f'Top {top_features} Feature Importances')
plt.gca().invert_yaxis()

# 4. Distribution of target variable
plt.subplot(3, 3, 4)
plt.hist(y_train, bins=20, alpha=0.7, label='Training', color='blue')
plt.hist(y_test, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Log(Property)')
plt.ylabel('Frequency')
plt.title('')
plt.legend()

# 5. Training vs Test performance comparison
plt.subplot(3, 3, 5)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
plt.bar(['Training', 'Test'], [r2_train, r2_test], color=['blue', 'red'])
plt.ylabel('R² Score')
plt.title('')

# 6. Prediction error distribution
plt.subplot(3, 3, 6)
plt.hist(residuals_train, bins=30, alpha=0.7, color='orange', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('')

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
    'Polymer Grades': ['Polymer Grade 1', 'Polymer Grade 2', 'Polymer Grade 3', 'Polymer Grade 4', 'Polymer Grade 5'],
    'Volume Fractions': ['vol_fraction1', 'vol_fraction2', 'vol_fraction3', 'vol_fraction4', 'vol_fraction5'],
    'SP Descriptors': ['SP_C', 'SP_N', 'SP2_C', 'SP2_N', 'SP2_O', 'SP2_S', 'SP2_B', 'SP3_C', 'SP3_N', 'SP3_O', 'SP3_S', 'SP3_P', 'SP3_Si', 'SP3_B', 'SP3_F', 'SP3_Cl', 'SP3_Br', 'SP3_I', 'SP3D2_S'],
    'Ring Systems': ['phenyls', 'cyclohexanes', 'cyclopentanes', 'cyclopentenes', 'thiophenes', 'aromatic_rings_with_n', 'aromatic_rings_with_o', 'aromatic_rings_with_n_o', 'aromatic_rings_with_s', 'aliphatic_rings_with_n', 'aliphatic_rings_with_o', 'aliphatic_rings_with_n_o', 'aliphatic_rings_with_s', 'other_rings'],
    'Carbon Types': ['primary_carbon', 'secondary_carbon', 'tertiary_carbon', 'quaternary_carbon'],
    'Functional Groups': ['carboxylic_acid', 'anhydride', 'acyl_halide', 'carbamide', 'urea', 'carbamate', 'thioamide', 'amide', 'ester', 'sulfonamide', 'sulfone', 'sulfoxide', 'phosphate', 'nitro', 'acetal', 'ketal', 'isocyanate', 'thiocyanate', 'azide', 'azo', 'imide', 'sulfonyl_halide', 'phosphonate', 'thiourea', 'guanidine', 'silicon_4_coord', 'boron_3_coord', 'vinyl', 'vinyl_halide', 'allene', 'alcohol', 'ether', 'aldehyde', 'ketone', 'thiol', 'thioether', 'primary_amine', 'secondary_amine', 'tertiary_amine', 'quaternary_amine', 'imine', 'nitrile']
}

category_importance = {}
for category, features in categories.items():
    category_importance[category] = get_importance_sum(features, feature_names, importances)

# Plot category importance
plt.bar(category_importance.keys(), category_importance.values(), color='skyblue')
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title(f'')
plt.xticks(rotation=45)

# 8. Model complexity analysis
plt.subplot(3, 3, 8)
n_trees = model.named_steps['regressor'].n_estimators
avg_depth = model.named_steps['regressor'].max_depth
learning_rate = model.named_steps['regressor'].learning_rate

complexity_metrics = ['Number of Trees', 'Max Depth', 'Learning Rate']
complexity_values = [n_trees, avg_depth, learning_rate]

plt.bar(complexity_metrics, complexity_values, color=['green', 'orange', 'purple'])
plt.title('')
plt.ylabel('Value')
plt.xticks(rotation=45)

# 9. Data summary
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'R² Score: {r2_score(y_train, y_pred_train):.3f}', fontsize=12)
plt.text(0.1, 0.7, f'MAE: {mean_absolute_error(y_train, y_pred_train):.3f}', fontsize=12)
plt.text(0.1, 0.6, f'RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.3f}', fontsize=12)
plt.text(0.1, 0.5, f'MAPE: {mean_absolute_percentage_error(y_train, y_pred_train)*100:.1f}%', fontsize=12)
plt.text(0.1, 0.4, f'Training Samples: {len(X_train)}', fontsize=12)
plt.text(0.1, 0.3, f'Test Samples: {len(X_test)}', fontsize=12)
plt.text(0.1, 0.2, f'Features: {X_train.shape[1]}', fontsize=12)
plt.text(0.1, 0.1, f'Target: {target}', fontsize=12)
plt.title('')
plt.axis('off')

plt.tight_layout()
plt.savefig(f'{args.output}/comprehensive_polymer_model_results.png', dpi=300, bbox_inches='tight')
plt.show()

# Create specific plot for polymer grade features importance
print("\n=== POLYMER GRADE FEATURES ANALYSIS ===")
polymer_grade_features = ['Polymer Grade 1', 'Polymer Grade 2']
polymer_grade_importance = []

for feature in polymer_grade_features:
    for i, feature_name in enumerate(feature_names):
        if feature in feature_name:
            polymer_grade_importance.append(importances[i])
            break

if len(polymer_grade_importance) >= 1:
    plt.figure(figsize=(12, 5))
    
    # Polymer grade features importance
    plt.subplot(1, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4']  # Red for grade 1, teal for grade 2
    bars = plt.bar(polymer_grade_features[:len(polymer_grade_importance)], polymer_grade_importance, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, importance in zip(bars, polymer_grade_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Polymer Grade Features')
    plt.ylabel('Feature Importance')
    plt.title('Polymer Grade Features Importance')
    plt.xticks(rotation=45)
    
    # Top 10 features with polymer grade highlight
    plt.subplot(1, 2, 2)
    top_10_indices = np.argsort(importances)[-10:]
    top_10_features = [feature_names[i] for i in top_10_indices]
    top_10_importances = [importances[i] for i in top_10_indices]
    
    # Color polymer grade features differently
    colors = []
    for feature in top_10_features:
        if any(grade in feature for grade in ['Polymer Grade']):
            colors.append('#FF6B6B')  # Red for polymer grade features
        else:
            colors.append('#4ECDC4')  # Teal for other features
    
    bars = plt.barh(range(len(top_10_features)), top_10_importances, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_10_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', ha='left', va='center', fontsize=8)
    
    plt.yticks(range(len(top_10_features)), [f.split('__')[-1] for f in top_10_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Features with Polymer Grade Highlight')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'polymer_grade_features_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Polymer Grade Features Importance:")
    for feature, importance in zip(polymer_grade_features[:len(polymer_grade_importance)], polymer_grade_importance):
        print(f"  {feature}: {importance:.4f}")
    print(f"  Combined Polymer Grade Importance: {sum(polymer_grade_importance):.4f}")
    print(f"  Polymer grade features saved to: polymer_grade_features_importance.png")
else:
    print("Warning: Could not find polymer grade features in the dataset")

# Create separate plot for last 10 blends performance
print("\n=== LAST 10 BLENDS PERFORMANCE ===")
# Get predictions for last 10 blends
last_10_X = X.iloc[last_10_indices]
last_10_y = y.iloc[last_10_indices]
last_10_pred = model.predict(last_10_X)

last_10_mae = mean_absolute_error(last_10_y, last_10_pred)
last_10_rmse = np.sqrt(mean_squared_error(last_10_y, last_10_pred))
last_10_r2 = r2_score(last_10_y, last_10_pred)

print(f"Last 10 Blends Performance:")
print(f"R² Score: {last_10_r2:.4f}")
print(f"Mean Absolute Error: {last_10_mae:.4f}")
print(f"Root Mean Squared Error: {last_10_rmse:.4f}")

# Create separate plot for last 10 blends - Log Scale
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions
plt.subplot(1, 2, 1)
plt.scatter(last_10_y, last_10_pred, color='red', s=100, alpha=0.7, label=f'Last 10 Blends (MAE={last_10_mae:.2f}, RMSE={last_10_rmse:.2f})')
plt.plot([last_10_y.min(), last_10_y.max()], [last_10_y.min(), last_10_y.max()], 'k--', lw=2)
plt.xlabel('Actual Log(Property)')
plt.ylabel('Predicted Log(Property)')
plt.title('Last 10 Blends: Log-Transformed Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_10_y, last_10_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions (inverse log)
plt.subplot(1, 2, 2)
# Convert back to original scale
original_actual = np.exp(last_10_y)
original_pred = np.exp(last_10_pred)

# Calculate better metrics for log-transformed data
relative_errors = np.abs(original_actual - original_pred) / original_actual
geometric_mae = np.exp(last_10_mae) - 1  # This gives the multiplicative factor
relative_mae = np.mean(relative_errors)
relative_rmse = np.sqrt(np.mean(relative_errors**2))

# Calculate standard deviation of the data for comparison
data_std = np.std(original_actual)
data_mean = np.mean(original_actual)
data_cv = data_std / data_mean  # Coefficient of variation

# Calculate error-to-noise ratio
error_to_std_ratio = relative_mae / data_cv
rmse_to_std_ratio = relative_rmse / data_cv

# Calculate original scale metrics
original_mae = mean_absolute_error(original_actual, original_pred)
original_rmse = np.sqrt(mean_squared_error(original_actual, original_pred))
original_r2 = r2_score(original_actual, original_pred)

print(f"\n=== ERROR vs DATA VARIABILITY ANALYSIS ===")
print(f"Data Statistics:")
print(f"  Mean Cobb: {data_mean:.2f}")
print(f"  Std Cobb: {data_std:.2f}")
print(f"  Coefficient of Variation (CV): {data_cv:.3f} ({data_cv*100:.1f}%)")
print(f"\nError Metrics:")
print(f"  Relative MAE: {relative_mae:.3f} ({relative_mae*100:.1f}%)")
print(f"  Relative RMSE: {relative_rmse:.3f} ({relative_rmse*100:.1f}%)")
print(f"\nError-to-Noise Ratios:")
print(f"  Relative MAE / CV: {error_to_std_ratio:.3f}")
print(f"  Relative RMSE / CV: {rmse_to_std_ratio:.3f}")
print(f"\nInterpretation:")
if error_to_std_ratio < 1:
    print(f"  ✅ Model error ({relative_mae*100:.1f}%) is LESS than data variability ({data_cv*100:.1f}%)")
    print(f"  ✅ Model is performing well relative to inherent data noise")
elif error_to_std_ratio < 2:
    print(f"  ⚠️  Model error ({relative_mae*100:.1f}%) is similar to data variability ({data_cv*100:.1f}%)")
    print(f"  ⚠️  Model performance is acceptable but could be improved")
else:
    print(f"  ❌ Model error ({relative_mae*100:.1f}%) is GREATER than data variability ({data_cv*100:.1f}%)")
    print(f"  ❌ Model is not performing well relative to inherent data noise")

plt.scatter(original_actual, original_pred, color='blue', s=100, alpha=0.7)
plt.plot([original_actual.min(), original_actual.max()], [original_actual.min(), original_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Cobb (Original Scale)')
plt.ylabel('Predicted Cobb (Original Scale)')
plt.title('Cobb Predictions - Original Scale')

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
for i, (actual, pred) in enumerate(zip(original_actual, original_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_10_blends_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nLast 10 Blends - Original Scale Performance:")
print(f"R² Score: {original_r2:.4f}")
print(f"Mean Absolute Error: {original_mae:.4f}")
print(f"Root Mean Squared Error: {original_rmse:.4f}")
print(f"Relative MAE: {relative_mae:.4f} ({relative_mae*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse:.4f} ({relative_rmse*100:.2f}%)")
print(f"Geometric MAE: {geometric_mae:.4f} ({geometric_mae*100:.2f}% multiplicative error)")

# Create separate plot for the very last blend performance
print("\n=== LAST BLEND PERFORMANCE ===")
# Get predictions for the very last blend
last_blend_index = len(df) - 1
last_blend_X = X.iloc[[last_blend_index]]
last_blend_y = y.iloc[last_blend_index]
last_blend_pred = model.predict(last_blend_X)[0]

last_blend_mae = abs(last_blend_y - last_blend_pred)
last_blend_rmse = np.sqrt((last_blend_y - last_blend_pred)**2)

print(f"Last Blend Performance:")
print(f"Actual Log(Property): {last_blend_y:.4f}")
print(f"Predicted Log(Property): {last_blend_pred:.4f}")
print(f"Absolute Error: {last_blend_mae:.4f}")
print(f"Root Mean Squared Error: {last_blend_rmse:.4f}")

# Create separate plot for the very last blend - Log Scale
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed prediction
plt.subplot(1, 2, 1)
plt.scatter(last_blend_y, last_blend_pred, color='darkred', s=200, alpha=0.8, label=f'Last Blend (Error={last_blend_mae:.2f})')
plt.plot([last_blend_y - 0.5, last_blend_y + 0.5], [last_blend_y - 0.5, last_blend_y + 0.5], 'k--', lw=2)
plt.xlabel('Actual Log(Property)')
plt.ylabel('Predicted Log(Property)')
plt.title('Last Blend: Log-Transformed Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# Add data point label
plt.annotate(f'Actual: {last_blend_y:.2f}\nPred: {last_blend_pred:.2f}', 
             (last_blend_y, last_blend_pred), textcoords="offset points", xytext=(0,20), 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Original scale prediction (inverse log)
plt.subplot(1, 2, 2)
# Convert back to original scale
original_last_actual = np.exp(last_blend_y)
original_last_pred = np.exp(last_blend_pred)

# Calculate relative error
relative_error = abs(original_last_actual - original_last_pred) / original_last_actual

print(f"\nLast Blend - Original Scale Performance:")
print(f"Actual Property: {original_last_actual:.2f}")
print(f"Predicted Property: {original_last_pred:.2f}")
print(f"Absolute Error: {abs(original_last_actual - original_last_pred):.2f}")
print(f"Relative Error: {relative_error:.4f} ({relative_error*100:.2f}%)")

plt.scatter(original_last_actual, original_last_pred, color='darkblue', s=200, alpha=0.8)
plt.plot([original_last_actual - 10, original_last_actual + 10], [original_last_actual - 10, original_last_actual + 10], 'r--', lw=2)
plt.xlabel('Actual Cobb (Original Scale)')
plt.ylabel('Predicted Cobb (Original Scale)')
plt.title('Last Blend: Original Scale Prediction')

# Add metrics as text box on the plot
metrics_text = (
    f'Actual: {original_last_actual:.1f}\n'
    f'Predicted: {original_last_pred:.1f}\n'
    f'Abs Error: {abs(original_last_actual - original_last_pred):.1f}\n'
    f'Rel Error: {relative_error*100:.1f}%'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.grid(True, alpha=0.3)

# Add data point label
plt.annotate(f'Actual: {original_last_actual:.1f}\nPred: {original_last_pred:.1f}', 
             (original_last_actual, original_last_pred), textcoords="offset points", xytext=(0,20), 
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_blend_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Last blend performance plot saved to: last_blend_performance.png")

# Save the model
import joblib
model_path = os.path.join(args.output, 'comprehensive_polymer_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved as '{model_path}'")

# Create a prediction function
def predict_cobb_property(input_data):
    """
    Predict Cobb property for polymer blend data
    
    Parameters:
    - input_data: pandas DataFrame with all features (excluding Materials column)
    
    Returns:
    - float: predicted Cobb property
    """
    # Handle missing values
    for col in categorical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna('Unknown')
    
    for col in numerical_features:
        if col in input_data.columns:
            input_data[col] = input_data[col].fillna(0)
    
    prediction = model.predict(input_data)[0]
    return prediction

# Example usage with the first row
print("\n=== EXAMPLE PREDICTION ===")
example_data = X.iloc[0:1].copy()
actual_value = y.iloc[0]
predicted_value = predict_cobb_property(example_data)
print(f"Actual Log(Property): {actual_value:.2f}")
print(f"Predicted Log(Property): {predicted_value:.2f}")
print(f"Prediction Error: {abs(actual_value - predicted_value):.2f}")

# Convert back to original scale for comparison
original_actual = np.exp(actual_value)
original_predicted = np.exp(predicted_value)
print(f"Actual Property (original scale): {original_actual:.2f}")
print(f"Predicted Property (original scale): {original_predicted:.2f}")
print(f"Original Scale Error: {abs(original_actual - original_predicted):.2f}")

print("\n=== MODEL SUMMARY ===")
print("✅ XGBoost model successfully trained ")
print("✅ Materials column excluded from features as requested")
print("✅ All other features included (categorical + numerical + featurized)")
print("✅ One-hot encoding applied to categorical features")
print("✅ XGBoost algorithm with gentle regularization")
print("✅ Feature importance analysis completed")
print("✅ Comprehensive visualizations saved as 'comprehensive_polymer_model_results.png'")
print("✅ Model saved as 'comprehensive_polymer_model.pkl'")
print(f"✅ All files saved to: {args.output}")
print("✅ Prediction function created for new data") 