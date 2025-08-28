import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import argparse
import os
import joblib
warnings.filterwarnings('ignore')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train XGBoost model for compostability prediction (max_L and t0)')
parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output', type=str, required=True, help='Output directory path')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input)

# Define target columns for compostability properties
target_col1 = 'property1'  # max_L (disintegration level, 0-100, >90 = home-compostable)
target_col2 = 'property2'  # t0 (time to 50% disintegration in days)

# Separate features and targets
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
X = df.drop(columns=[target_col1, target_col2, 'Materials'] + smiles_cols)
y1 = df[target_col1]  # MD property
y2 = df[target_col2]  # TD property

# Apply log transformation to target values (handle zero values for max_L)
# For max_L: add small constant to avoid log(0), for t0: direct log
log_y1 = np.log(y1 + 1e-6)  # max_L: disintegration level
log_y2 = np.log(y2)          # t0: time to disintegration

# Identify categorical and numerical features
categorical_features = []
numerical_features = []

for col in X.columns:
    if X[col].dtype == 'object' or X[col].dtype == 'string':
        categorical_features.append(col)
    else:
        numerical_features.append(col)

# Handle missing values
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

# Split the data using train_test_split while ensuring last 4 blends are in training
last_4_indices = list(range(len(df) - 4, len(df)))

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

# Oversample the last 4 blends by 10x
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

# Use oversampled data for training
X_train = X_train_oversampled
log_y1_train = log_y1_train_oversampled
log_y2_train = log_y2_train_oversampled

# Create two separate models for MD and TD
model1 = Pipeline([
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

model2 = Pipeline([
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

# Train both models
model1.fit(X_train, log_y1_train)
model2.fit(X_train, log_y2_train)

# Make predictions (log scale)
y1_pred_train = model1.predict(X_train)
y1_pred_test = model1.predict(X_test)
y2_pred_train = model2.predict(X_train)
y2_pred_test = model2.predict(X_test)

# Inverse transform predictions and targets to original scale
orig_y1_train = np.exp(log_y1_train) - 1e-6  # max_L: subtract the small constant
orig_y1_test = np.exp(log_y1_test) - 1e-6
orig_pred1_train = np.exp(y1_pred_train) - 1e-6
orig_pred1_test = np.exp(y1_pred_test) - 1e-6

orig_y2_train = np.exp(log_y2_train)  # t0: direct exponential
orig_y2_test = np.exp(log_y2_test)
orig_pred2_train = np.exp(y2_pred_train)
orig_pred2_test = np.exp(y2_pred_test)

# Get feature names
feature_names = []
if hasattr(model1.named_steps['preprocessor'], 'get_feature_names_out'):
    feature_names = model1.named_steps['preprocessor'].get_feature_names_out()
else:
    feature_names = [f"feature_{i}" for i in range(len(model1.named_steps['regressor'].feature_importances_))]

importances1 = model1.named_steps['regressor'].feature_importances_
importances2 = model2.named_steps['regressor'].feature_importances_

# Last 4 blends performance for max_L
last_4_X = X.iloc[last_4_indices]
last_4_log_y1 = log_y1.iloc[last_4_indices]
last_4_pred_log1 = model1.predict(last_4_X)
last_4_mae_log1 = mean_absolute_error(last_4_log_y1, last_4_pred_log1)
last_4_r2_log1 = r2_score(last_4_log_y1, last_4_pred_log1)

# Original scale
last_4_actual1 = np.exp(last_4_log_y1) - 1e-6
last_4_pred1 = np.exp(last_4_pred_log1) - 1e-6
last_4_mae1 = mean_absolute_error(last_4_actual1, last_4_pred1)
last_4_r21 = r2_score(last_4_actual1, last_4_pred1)

# Last 4 blends performance for t0
last_4_log_y2 = log_y2.iloc[last_4_indices]
last_4_pred_log2 = model2.predict(last_4_X)
last_4_mae_log2 = mean_absolute_error(last_4_log_y2, last_4_pred_log2)
last_4_r2_log2 = r2_score(last_4_log_y2, last_4_pred_log2)

# Original scale
last_4_actual2 = np.exp(last_4_log_y2)
last_4_pred2 = np.exp(last_4_pred_log2)
last_4_mae2 = mean_absolute_error(last_4_actual2, last_4_pred2)
last_4_r22 = r2_score(last_4_actual2, last_4_pred2)

# Create output directory
os.makedirs(args.output, exist_ok=True)

# Create last 4 blends performance plot - max_L
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for max_L
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y1, last_4_pred_log1, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y1.min(), last_4_log_y1.max()], [last_4_log_y1.min(), last_4_log_y1.max()], 'k--', lw=2)
plt.xlabel('Actual Log(max_L)')
plt.ylabel('Predicted Log(max_L)')
plt.title('max_L (Disintegration Level) Predictions')

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

# Plot 2: Original scale predictions for max_L
plt.subplot(1, 2, 2)
plt.scatter(last_4_actual1, last_4_pred1, color='blue', s=100, alpha=0.7)
plt.plot([last_4_actual1.min(), last_4_actual1.max()], [last_4_actual1.min(), last_4_actual1.max()], 'r--', lw=2)
plt.xlabel('Actual max_L (Original Scale)')
plt.ylabel('Predicted max_L (Original Scale)')
plt.title('max_L (Disintegration Level) Predictions')

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
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_max_L.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create last 4 blends performance plot - t0
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for t0
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y2, last_4_pred_log2, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y2.min(), last_4_log_y2.max()], [last_4_log_y2.min(), last_4_log_y2.max()], 'k--', lw=2)
plt.xlabel('Actual Log(t0)')
plt.ylabel('Predicted Log(t0)')
plt.title('t0 (Time to Disintegration) Predictions')

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

# Plot 2: Original scale predictions for t0
plt.subplot(1, 2, 2)
plt.scatter(last_4_actual2, last_4_pred2, color='blue', s=100, alpha=0.7)
plt.plot([last_4_actual2.min(), last_4_actual2.max()], [last_4_actual2.min(), last_4_actual2.max()], 'r--', lw=2)
plt.xlabel('Actual t0 (Original Scale)')
plt.ylabel('Predicted t0 (Original Scale)')
plt.title('t0 (Time to Disintegration) Predictions')

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
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_t0.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save the models
model1_path = os.path.join(args.output, 'comprehensive_polymer_model_max_L.pkl')
model2_path = os.path.join(args.output, 'comprehensive_polymer_model_t0.pkl')
joblib.dump(model1, model1_path)
joblib.dump(model2, model2_path)

# Prediction function for both properties
def predict_compostability(input_data):
    """
    Predict both max_L and t0 for polymer blend data
    Parameters:
    - input_data: pandas DataFrame with all features (excluding SMILES columns)
    Returns:
    - tuple: (max_L_prediction, t0_prediction) where max_L is 0-100 (>90 = home-compostable), t0 is in days
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
    return np.exp(prediction1_log) - 1e-6, np.exp(prediction2_log) 