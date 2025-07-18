import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
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
parser = argparse.ArgumentParser(description='Train XGBoost model for WVTR polymer blends (Generalization Model - Better Test Performance)')
parser.add_argument('--input', type=str, default='data/wvtr/polymerblends_for_ml_featurized.csv',
                    help='Input CSV file path (default: data/wvtr/polymerblends_for_ml_featurized.csv)')
parser.add_argument('--output', type=str, default='models/wvtr/v3/wvtr_general',
                    help='Output directory path (default: models/wvtr/v3/wvtr_general)')
parser.add_argument('--target', type=str, default='property',
                    help='Target column name (default: property)')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

print(f"Loading featurized polymer blends data from: {args.input}")
df = pd.read_csv(args.input)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Separate features and target
target = args.target
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
# EXCLUDE Materials column from features (v4 change)
excluded_cols = [target] + smiles_cols + ['Materials']
X = df.drop(columns=excluded_cols)
y = df[target]

# Apply log transformation to target values
y = np.log(y)  # log(x) without +1 offset

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

# Split data ensuring last 21 blends are in training AND all samples with environmental conditions
print("Splitting data with 80/20 split, ensuring last 21 blends AND all environmental samples in training...")

# Find indices of samples with environmental conditions (non-NaN values)
env_condition_mask = df[['Temperature (C)', 'RH (%)', 'Thickness (um)']].notna().any(axis=1)
env_indices = df[env_condition_mask].index.tolist()
print(f"Found {len(env_indices)} samples with environmental conditions at indices: {env_indices}")

last_21_indices = list(range(len(df) - 21, len(df)))

# Create a mask for the remaining data (excluding last 21 and environmental samples)
remaining_indices = [i for i in range(len(df) - 21) if i not in env_indices]

# Use 80% of remaining data for training, 20% for testing
from sklearn.model_selection import train_test_split
train_remaining, test_indices = train_test_split(
    remaining_indices, 
    test_size=0.2, 
    random_state=42
)

# Combine training indices: remaining training data + last 21 blends + environmental samples
train_indices = train_remaining + last_21_indices + env_indices
# Remove duplicates (in case environmental samples overlap with last 21)
train_indices = list(set(train_indices))

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

print(f"Last 21 blend indices: {last_21_indices}")
print(f"Environmental sample indices: {env_indices}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train/test ratio: {len(X_train)/(len(X_train)+len(X_test)):.1%}/{len(X_test)/(len(X_train)+len(X_test)):.1%}")
print(f"Last 21 blends in training: {all(i in train_indices for i in last_21_indices)}")
print(f"All environmental samples in training: {all(i in train_indices for i in env_indices)}")

# Oversample last 4 blends in training set by 15x
# Since indices are reset after split, we need to identify the last 4 blends differently
# Get the last 4 samples from the training set (which should be the last 4 blends)
last_4_train_indices = list(range(len(X_train) - 4, len(X_train)))
print(f"Oversampling last 4 training samples (indices {last_4_train_indices}) by 15x")

for idx in last_4_train_indices:
    for _ in range(15):
        X_train = pd.concat([X_train, X_train.iloc[[idx]]], ignore_index=True)
        y_train = pd.concat([y_train, y_train.iloc[[idx]]], ignore_index=True)

print(f"Training set size after oversampling: {len(X_train)}")

# Add cross-validation for better generalization assessment
print("\n=== CROSS-VALIDATION ASSESSMENT ===")

# Create base model for CV
base_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        colsample_bylevel=0.8,
        reg_alpha=0.5,
        reg_lambda=3.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    ))
])

# Perform 5-fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Hyperparameter optimization for better generalization
print("\n=== HYPERPARAMETER OPTIMIZATION ===")

# Define parameter grid for optimization
param_grid = {
    'regressor__max_depth': [2, 3, 4],
    'regressor__learning_rate': [0.03, 0.05, 0.07],
    'regressor__reg_alpha': [0.3, 0.5, 0.7],
    'regressor__reg_lambda': [2.0, 3.0, 4.0],
    'regressor__subsample': [0.6, 0.7, 0.8],
    'regressor__colsample_bytree': [0.6, 0.7, 0.8]
}

# Create model for grid search
grid_model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100,
        colsample_bylevel=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        n_jobs=-1
    ))
])

# Perform grid search with cross-validation
print("Performing grid search for optimal hyperparameters...")
grid_search = GridSearchCV(
    grid_model, 
    param_grid, 
    cv=3, 
    scoring='r2', 
    n_jobs=-1, 
    verbose=1
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

print("\nTraining the XGBoost model for WVTR prediction (Generalization Model - Better Test Performance)...")

# Create pipeline with optimized XGBoost parameters
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100,      # More trees for better ensemble
        max_depth=grid_search.best_params_['regressor__max_depth'],
        learning_rate=grid_search.best_params_['regressor__learning_rate'],
        subsample=grid_search.best_params_['regressor__subsample'],
        colsample_bytree=grid_search.best_params_['regressor__colsample_bytree'],
        colsample_bylevel=0.8, # Column sampling by level
        reg_alpha=grid_search.best_params_['regressor__reg_alpha'],
        reg_lambda=grid_search.best_params_['regressor__reg_lambda'],
        min_child_weight=3,    # Increased minimum child weight
        gamma=0.1,             # Minimum loss reduction for split
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
plt.figure(figsize=(20, 15))

# 1. Actual vs Predicted plot
plt.subplot(3, 3, 1)
plt.scatter(y_train, y_pred_train, alpha=0.6, label=f'Training (MAE={mean_absolute_error(y_train, y_pred_train):.2f}, RMSE={np.sqrt(mean_squared_error(y_train, y_pred_train)):.2f})', color='blue')
plt.scatter(y_test, y_pred_test, alpha=0.6, label=f'Test (MAE={mean_absolute_error(y_test, y_pred_test):.2f}, RMSE={np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f})', color='red')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Log(Property)')
plt.ylabel('Predicted Log(Property)')
plt.title('Actual vs Predicted Log-Transformed Property (Generalization Model - Better Test Performance)')
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
plt.title('Residuals Plot (Generalization Model - Better Test Performance)')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Feature importance plot
plt.subplot(3, 3, 3)
top_features = 20
top_indices = indices[:top_features]
plt.barh(range(top_features), importances[top_indices])
plt.yticks(range(top_features), [feature_names[i] for i in top_indices])
plt.xlabel('Feature Importance')
plt.title(f'Top {top_features} Feature Importances (Generalization Model - Better Test Performance)')
plt.gca().invert_yaxis()

# 4. Distribution of target variable
plt.subplot(3, 3, 4)
plt.hist(y, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Log(Property)')
plt.ylabel('Frequency')
plt.title('Distribution of Target Variable')
plt.grid(True, alpha=0.3)

# 5. Training vs Test performance comparison
plt.subplot(3, 3, 5)
metrics = ['R²', 'MSE', 'MAE', 'RMSE']
train_scores = [r2_score(y_train, y_pred_train), 
                mean_squared_error(y_train, y_pred_train),
                mean_absolute_error(y_train, y_pred_train),
                np.sqrt(mean_squared_error(y_train, y_pred_train))]
test_scores = [r2_score(y_test, y_pred_test),
               mean_squared_error(y_test, y_pred_test),
               mean_absolute_error(y_test, y_pred_test),
               np.sqrt(mean_squared_error(y_test, y_pred_test))]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Training', color='blue', alpha=0.7)
plt.bar(x + width/2, test_scores, width, label='Test', color='red', alpha=0.7)
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Training vs Test Performance (Generalization Model - Better Test Performance)')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Prediction error distribution
plt.subplot(3, 3, 6)
plt.hist(residuals_train, bins=15, alpha=0.7, label='Training', color='blue')
plt.hist(residuals_test, bins=15, alpha=0.7, label='Test', color='red')
plt.xlabel('Prediction Error (Log(Property))')
plt.ylabel('Frequency')
plt.title('Prediction Error Distribution (Generalization Model - Better Test Performance)')
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
    'Environmental': ['Temperature', 'RH'],
    'Polymer Grades': ['Polymer Grade'],
    'Volume Fractions': ['vol_fraction'],
    'SP Descriptors': ['SP_'],
    'Chemical Groups': ['alcohol', 'ether', 'ester', 'amide', 'ketone', 'aldehyde'],
    'Ring Systems': ['phenyl', 'cyclo', 'aromatic', 'aliphatic'],
    'Carbon Types': ['carbon']
}

category_importance = {}
for category, features in categories.items():
    category_importance[category] = get_importance_sum(features, feature_names, importances)

plt.bar(category_importance.keys(), category_importance.values())
plt.xlabel('Feature Categories')
plt.ylabel('Total Importance')
plt.title('Feature Importance by Category (Generalization Model - Better Test Performance)')
plt.xticks(rotation=45)

# 8. Model complexity analysis
plt.subplot(3, 3, 8)
n_trees = model.named_steps['regressor'].n_estimators
avg_depth = model.named_steps['regressor'].max_depth
learning_rate = model.named_steps['regressor'].learning_rate

complexity_metrics = ['Number of Trees', 'Max Depth', 'Learning Rate']
complexity_values = [n_trees, avg_depth, learning_rate]

plt.bar(complexity_metrics, complexity_values, color=['green', 'orange', 'purple'])
plt.title('Model Complexity Metrics (Generalization Model - Better Test Performance)')
plt.ylabel('Value')
plt.xticks(rotation=45)

# 9. Data summary
plt.subplot(3, 3, 9)
plt.text(0.1, 0.8, f'Total Features: {len(feature_names)}', fontsize=12)
plt.text(0.1, 0.7, f'Categorical: {len(categorical_features)}', fontsize=12)
plt.text(0.1, 0.6, f'Numerical: {len(numerical_features)}', fontsize=12)
plt.text(0.1, 0.5, f'Training Samples: {len(y_train)}', fontsize=12)
plt.text(0.1, 0.4, f'Test Samples: {len(y_test)}', fontsize=12)
plt.text(0.1, 0.3, f'Target Range: {y.min():.1f} - {y.max():.1f} Log(Property)', fontsize=12)
plt.text(0.1, 0.2, f'Target Mean: {y.mean():.1f} Log(Property)', fontsize=12)
plt.text(0.1, 0.1, f'Target Std: {y.std():.1f} Log(Property)', fontsize=12)
plt.title('Dataset Summary (Generalization Model - Better Test Performance)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results.png'), dpi=300, bbox_inches='tight')
plt.show()

# Create specific plot for environmental features importance
print("\n=== ENVIRONMENTAL FEATURES ANALYSIS ===")
env_features = ['Temperature (C)', 'RH (%)']
env_importance = []

for feature in env_features:
    for i, feature_name in enumerate(feature_names):
        if feature in feature_name:
            env_importance.append(importances[i])
            break

if len(env_importance) == 2:
    plt.figure(figsize=(10, 6))
    
    # Environmental features importance
    plt.subplot(1, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4']  # Red for temperature, teal for humidity
    bars = plt.bar(env_features, env_importance, color=colors, alpha=0.8)
    plt.xlabel('Environmental Features')
    plt.ylabel('Feature Importance')
    plt.title('Environmental Features Importance (Generalization Model - Better Test Performance)')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, importance in zip(bars, env_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Environmental features vs other top features
    plt.subplot(1, 2, 2)
    top_10_features = []
    top_10_importance = []
    
    for i in range(min(10, len(indices))):
        feature_name = feature_names[indices[i]]
        importance = importances[indices[i]]
        top_10_features.append(feature_name)
        top_10_importance.append(importance)
    
    # Color environmental features differently
    colors = []
    for feature in top_10_features:
        if 'Temperature' in feature or 'RH' in feature:
            colors.append('#FF6B6B')  # Red for environmental
        else:
            colors.append('#4ECDC4')  # Teal for others
    
    bars = plt.barh(range(len(top_10_features)), top_10_importance, color=colors, alpha=0.8)
    plt.yticks(range(len(top_10_features)), [f.split('__')[-1] for f in top_10_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Features with Environmental Highlight (Generalization Model - Better Test Performance)')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Environmental'),
                      Patch(facecolor='#4ECDC4', label='Other Features')]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'environmental_features_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Environmental Features Importance:")
    print(f"  Temperature (C): {env_importance[0]:.4f}")
    print(f"  RH (%): {env_importance[1]:.4f}")
    print(f"  Combined Environmental Importance: {sum(env_importance):.4f}")
    print(f"  Environmental features saved to: environmental_features_importance.png")

# Create separate plot for last 21 blends - Log Scale
print("\n=== LAST 21 BLENDS PERFORMANCE ===")
# Get predictions for last 21 blends
last_21_X = X.iloc[last_21_indices]
last_21_y = y.iloc[last_21_indices]
last_21_pred = model.predict(last_21_X)

last_21_mae = mean_absolute_error(last_21_y, last_21_pred)
last_21_rmse = np.sqrt(mean_squared_error(last_21_y, last_21_pred))
last_21_r2 = r2_score(last_21_y, last_21_pred)

print(f"Last 21 Blends Performance:")
print(f"R² Score: {last_21_r2:.4f}")
print(f"Mean Absolute Error: {last_21_mae:.4f}")
print(f"Root Mean Squared Error: {last_21_rmse:.4f}")

# Get predictions for last 4 blends
last_4_indices = last_21_indices[-4:]
last_4_X = X.iloc[last_4_indices]
last_4_y = y.iloc[last_4_indices]
last_4_pred = model.predict(last_4_X)

last_4_mae = mean_absolute_error(last_4_y, last_4_pred)
last_4_rmse = np.sqrt(mean_squared_error(last_4_y, last_4_pred))
last_4_r2 = r2_score(last_4_y, last_4_pred)

print(f"\nLast 4 Blends Performance:")
print(f"R² Score: {last_4_r2:.4f}")
print(f"Mean Absolute Error: {last_4_mae:.4f}")
print(f"Root Mean Squared Error: {last_4_rmse:.4f}")

# Create separate plot for last 21 blends and last 4 blends - Log Scale
plt.figure(figsize=(20, 8))

# Plot 1: Last 21 blends - Log-transformed predictions
plt.subplot(2, 3, 1)
plt.scatter(last_21_y, last_21_pred, color='red', s=100, alpha=0.7, label=f'Last 21 Blends (MAE={last_21_mae:.2f}, RMSE={last_21_rmse:.2f})')
plt.plot([last_21_y.min(), last_21_y.max()], [last_21_y.min(), last_21_y.max()], 'k--', lw=2)
plt.xlabel('Actual Log(Property)')
plt.ylabel('Predicted Log(Property)')
plt.title('Last 21 Blends: Log-Transformed Predictions (Generalization Model)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_21_y, last_21_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Last 4 blends - Log-transformed predictions
plt.subplot(2, 3, 2)
plt.scatter(last_4_y, last_4_pred, color='blue', s=150, alpha=0.8, label=f'Last 4 Blends (MAE={last_4_mae:.2f}, RMSE={last_4_rmse:.2f})')
plt.plot([last_4_y.min(), last_4_y.max()], [last_4_y.min(), last_4_y.max()], 'k--', lw=2)
plt.xlabel('Actual Log(Property)')
plt.ylabel('Predicted Log(Property)')
plt.title('Last 4 Blends: Log-Transformed Predictions (Generalization Model)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_y, last_4_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, weight='bold')

# Plot 3: Last 21 blends - Original scale predictions
plt.subplot(2, 3, 3)
original_actual_21 = np.exp(last_21_y)
original_pred_21 = np.exp(last_21_pred)
relative_errors_21 = np.abs(original_actual_21 - original_pred_21) / original_actual_21
relative_mae_21 = np.mean(relative_errors_21)
relative_rmse_21 = np.sqrt(np.mean(relative_errors_21**2))

original_mae_21 = mean_absolute_error(original_actual_21, original_pred_21)
original_rmse_21 = np.sqrt(mean_squared_error(original_actual_21, original_pred_21))

plt.scatter(original_actual_21, original_pred_21, color='red', s=100, alpha=0.7, 
           label=f'Last 21 Blends\nAbs MAE: {original_mae_21:.2f}\nAbs RMSE: {original_rmse_21:.2f}\nRel MAE: {relative_mae_21:.1%}\nRel RMSE: {relative_rmse_21:.1%}')
plt.plot([original_actual_21.min(), original_actual_21.max()], [original_actual_21.min(), original_actual_21.max()], 'k--', lw=2)
plt.xlabel('Actual Property (original scale)')
plt.ylabel('Predicted Property (original scale)')
plt.title('Last 21 Blends: Original Scale Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Last 4 blends - Original scale predictions
plt.subplot(2, 3, 4)
original_actual_4 = np.exp(last_4_y)
original_pred_4 = np.exp(last_4_pred)
relative_errors_4 = np.abs(original_actual_4 - original_pred_4) / original_actual_4
relative_mae_4 = np.mean(relative_errors_4)
relative_rmse_4 = np.sqrt(np.mean(relative_errors_4**2))

# Calculate absolute metrics for last 4 blends
original_mae_4 = mean_absolute_error(original_actual_4, original_pred_4)
original_rmse_4 = np.sqrt(mean_squared_error(original_actual_4, original_pred_4))

plt.scatter(original_actual_4, original_pred_4, color='blue', s=150, alpha=0.8, 
           label=f'Last 4 Blends\nAbs MAE: {original_mae_4:.2f}\nAbs RMSE: {original_rmse_4:.2f}\nRel MAE: {relative_mae_4:.1%}\nRel RMSE: {relative_rmse_4:.1%}')
plt.plot([original_actual_4.min(), original_actual_4.max()], [original_actual_4.min(), original_actual_4.max()], 'k--', lw=2)
plt.xlabel('Actual Property (original scale)')
plt.ylabel('Predicted Property (original scale)')
plt.title('Last 4 Blends: Original Scale Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 5: Performance comparison
plt.subplot(2, 3, 5)
metrics = ['R²', 'MAE', 'RMSE']
last_21_metrics = [last_21_r2, last_21_mae, last_21_rmse]
last_4_metrics = [last_4_r2, last_4_mae, last_4_rmse]

x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width/2, last_21_metrics, width, label='Last 21 Blends', color='red', alpha=0.7)
plt.bar(x + width/2, last_4_metrics, width, label='Last 4 Blends', color='blue', alpha=0.8)
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Performance Comparison')
plt.xticks(x, metrics)
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Relative error comparison
plt.subplot(2, 3, 6)
rel_metrics = ['Relative MAE', 'Relative RMSE']
rel_last_21 = [relative_mae_21, relative_rmse_21]
rel_last_4 = [relative_mae_4, relative_rmse_4]

x = np.arange(len(rel_metrics))
plt.bar(x - width/2, rel_last_21, width, label='Last 21 Blends', color='red', alpha=0.7)
plt.bar(x + width/2, rel_last_4, width, label='Last 4 Blends', color='blue', alpha=0.8)
plt.xlabel('Relative Error Metrics')
plt.ylabel('Relative Error (%)')
plt.title('Relative Error Comparison')
plt.xticks(x, rel_metrics)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_21_and_4_blends_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

# Calculate original scale metrics
original_mae_21 = mean_absolute_error(original_actual_21, original_pred_21)
original_rmse_21 = np.sqrt(mean_squared_error(original_actual_21, original_pred_21))
original_r2_21 = r2_score(original_actual_21, original_pred_21)

original_mae_4 = mean_absolute_error(original_actual_4, original_pred_4)
original_rmse_4 = np.sqrt(mean_squared_error(original_actual_4, original_pred_4))
original_r2_4 = r2_score(original_actual_4, original_pred_4)

print(f"\nLast 21 Blends - Original Scale Performance:")
print(f"R² Score: {original_r2_21:.4f}")
print(f"Mean Absolute Error: {original_mae_21:.4f}")
print(f"Root Mean Squared Error: {original_rmse_21:.4f}")
print(f"Relative MAE: {relative_mae_21:.4f} ({relative_mae_21*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse_21:.4f} ({relative_rmse_21*100:.2f}%)")

print(f"\nLast 4 Blends - Original Scale Performance:")
print(f"R² Score: {original_r2_4:.4f}")
print(f"Mean Absolute Error: {original_mae_4:.4f}")
print(f"Root Mean Squared Error: {original_rmse_4:.4f}")
print(f"Relative MAE: {relative_mae_4:.4f} ({relative_mae_4*100:.2f}%)")
print(f"Relative RMSE: {relative_rmse_4:.4f} ({relative_rmse_4*100:.2f}%)")

# Save the model
import joblib
model_path = os.path.join(args.output, 'comprehensive_polymer_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved as '{model_path}'")

# Create a prediction function
def predict_tensile_strength(input_data):
    """
    Predict tensile strength for polymer blend data (Generalization Model - Better Test Performance)
    
    Parameters:
    - input_data: pandas DataFrame with all features (excluding Materials column)
    
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
    
    prediction = model.predict(input_data)[0]
    return prediction

# Example usage with the first row
print("\n=== EXAMPLE PREDICTION ===")
example_data = X.iloc[0:1].copy()
actual_value = y.iloc[0]
predicted_value = predict_tensile_strength(example_data)
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
print("✅ XGBoost model successfully trained (Generalization Model - Better Test Performance)!")
print("✅ Materials column excluded from features as requested")
print("✅ All other features included (categorical + numerical + featurized)")
print("✅ One-hot encoding applied to categorical features")
print("✅ XGBoost algorithm with gentle regularization")
print("✅ Feature importance analysis completed")
print("✅ Comprehensive visualizations saved as 'comprehensive_polymer_model_results.png'")
print("✅ Model saved as 'comprehensive_polymer_model.pkl'")
print(f"✅ All files saved to: {args.output}")
print("✅ Prediction function created for new data") 