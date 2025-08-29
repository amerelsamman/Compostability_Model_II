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
parser = argparse.ArgumentParser(description='Train XGBoost model for elongation at break prediction with thickness')
parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output', type=str, required=True, help='Output directory path')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input)

# Define target columns for both EAB1 and EAB2
target_col1 = 'property1'  # EAB1 (First Elongation at Break)
target_col2 = 'property2'  # EAB2 (Second Elongation at Break)

# Separate features and targets
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
X = df.drop(columns=[target_col1, target_col2, 'Materials'] + smiles_cols)
y1 = df[target_col1]  # EAB1 property
y2 = df[target_col2]  # EAB2 property

# Apply log transformation to target values
log_y1 = np.log(y1)
log_y2 = np.log(y2)

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

# Split the data using train_test_split while ensuring last 4 blends are in testing
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

# Combine: last 4 always in testing, rest split by train_test_split
train_indices = temp_train_indices
test_indices = last_4_indices + temp_test_indices

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
log_y1_train = log_y1.iloc[train_indices]
log_y1_test = log_y1.iloc[test_indices]
log_y2_train = log_y2.iloc[train_indices]
log_y2_test = log_y2.iloc[test_indices]

# Oversample the last 16-4 blends (last 12 blends) 20x
last_16_indices = list(range(len(df) - 16, len(df)))
last_12_indices = list(range(len(df) - 16, len(df) - 4))  # Last 12 blends (16-4)

# Create oversampling weights - set all to 1.0, then 20x for last 12
weights = pd.Series(1.0, index=log_y1_train.index)
for idx in last_12_indices:
    if idx in log_y1_train.index:
        weights.iloc[log_y1_train.index.get_loc(idx)] = weights.iloc[log_y1_train.index.get_loc(idx)] * 20

# Create the full pipeline with XGBoost for EAB1
model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=120,
        max_depth=8,
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

# Create the full pipeline with XGBoost for EAB2
model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=120,
        max_depth=8,
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

# Train the models with sample weights
print("Training EAB1 model...")
model1.fit(X_train, log_y1_train, regressor__sample_weight=weights.values)

print("Training EAB2 model...")
model2.fit(X_train, log_y2_train, regressor__sample_weight=weights.values)

# Make predictions on test set
log_y1_pred = model1.predict(X_test)
log_y2_pred = model2.predict(X_test)

# Convert back to original scale
y1_pred = np.exp(log_y1_pred)
y2_pred = np.exp(log_y2_pred)
y1_test_orig = np.exp(log_y1_test)
y2_test_orig = np.exp(log_y2_test)

# Calculate metrics for test set
mae1 = mean_absolute_error(y1_test_orig, y1_pred)
mae2 = mean_absolute_error(y2_test_orig, y2_pred)
r2_1 = r2_score(y1_test_orig, y1_pred)
r2_2 = r2_score(y2_test_orig, y2_pred)

print(f"\nTest Set Performance:")
print(f"EAB1 - MAE: {mae1:.2f}, R²: {r2_1:.3f}")
print(f"EAB2 - MAE: {mae2:.2f}, R²: {r2_2:.3f}")

# Make predictions on last 4 blends
last_4_X_test = X.iloc[last_4_indices]
last_4_log_y1_test = log_y1.iloc[last_4_indices]
last_4_log_y2_test = log_y2.iloc[last_4_indices]

last_4_log_y1_pred = model1.predict(last_4_X_test)
last_4_log_y2_pred = model2.predict(last_4_X_test)

# Convert to original scale
last_4_y1_pred = np.exp(last_4_log_y1_pred)
last_4_y2_pred = np.exp(last_4_log_y2_pred)
last_4_y1_actual = np.exp(last_4_log_y1_test)
last_4_y2_actual = np.exp(last_4_log_y2_test)

# Calculate metrics for last 4 blends
last_4_mae1 = mean_absolute_error(last_4_y1_actual, last_4_y1_pred)
last_4_mae2 = mean_absolute_error(last_4_y2_actual, last_4_y2_pred)
last_4_r21 = r2_score(last_4_y1_actual, last_4_y1_pred)
last_4_r22 = r2_score(last_4_y2_actual, last_4_y2_pred)

# Calculate log-scale metrics for last 4 blends
last_4_mae_log1 = mean_absolute_error(last_4_log_y1_test, last_4_log_y1_pred)
last_4_mae_log2 = mean_absolute_error(last_4_log_y2_test, last_4_log_y2_pred)
last_4_r2_log1 = r2_score(last_4_log_y1_test, last_4_log_y1_pred)
last_4_r2_log2 = r2_score(last_4_log_y2_test, last_4_log_y2_pred)

print(f"\nLast 4 Blends Performance:")
print(f"EAB1 - MAE: {last_4_mae1:.2f}, R²: {last_4_r21:.3f}")
print(f"EAB2 - MAE: {last_4_mae2:.2f}, R²: {last_4_r22:.3f}")

# Create output directory
os.makedirs(args.output, exist_ok=True)

# Create last 4 blends performance plot - EAB1
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for EAB1
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y1_test, last_4_log_y1_pred, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y1_test.min(), last_4_log_y1_test.max()], [last_4_log_y1_test.min(), last_4_log_y1_test.max()], 'k--', lw=2)
plt.xlabel('Actual Log(EAB1)')
plt.ylabel('Predicted Log(EAB1)')
plt.title('EAB1 Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae_log1:.2f}\n'
    f'R²: {last_4_r2_log1:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_log_y1_test, last_4_log_y1_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions for EAB1
plt.subplot(1, 2, 2)
plt.scatter(last_4_y1_actual, last_4_y1_pred, color='blue', s=100, alpha=0.7)
plt.plot([last_4_y1_actual.min(), last_4_y1_actual.max()], [last_4_y1_actual.min(), last_4_y1_actual.max()], 'r--', lw=2)
plt.xlabel('Actual EAB1 (Original Scale)')
plt.ylabel('Predicted EAB1 (Original Scale)')
plt.title('EAB1 Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae1:.2f}\n'
    f'R²: {last_4_r21:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_y1_actual, last_4_y1_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_EAB1.png'), dpi=300, bbox_inches='tight')
plt.close()

# Create last 4 blends performance plot - EAB2
plt.figure(figsize=(15, 6))

# Plot 1: Log-transformed predictions for EAB2
plt.subplot(1, 2, 1)
plt.scatter(last_4_log_y2_test, last_4_log_y2_pred, color='red', s=100, alpha=0.7)
plt.plot([last_4_log_y2_test.min(), last_4_log_y2_test.max()], [last_4_log_y2_test.min(), last_4_log_y2_test.max()], 'k--', lw=2)
plt.xlabel('Actual Log(EAB2)')
plt.ylabel('Predicted Log(EAB2)')
plt.title('EAB2 Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae_log2:.2f}\n'
    f'R²: {last_4_r2_log2:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_log_y2_test, last_4_log_y2_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Plot 2: Original scale predictions for EAB2
plt.subplot(1, 2, 2)
plt.scatter(last_4_y2_actual, last_4_y2_pred, color='blue', s=100, alpha=0.7)
plt.plot([last_4_y2_actual.min(), last_4_y2_actual.max()], [last_4_y2_actual.min(), last_4_y2_actual.max()], 'r--', lw=2)
plt.xlabel('Actual EAB2 (Original Scale)')
plt.ylabel('Predicted EAB2 (Original Scale)')
plt.title('EAB2 Predictions')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {last_4_mae2:.2f}\n'
    f'R²: {last_4_r22:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels
for i, (actual, pred) in enumerate(zip(last_4_y2_actual, last_4_y2_pred)):
    plt.annotate(f'{actual:.1f}', (actual, pred), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_4_blends_performance_EAB2.png'), dpi=300, bbox_inches='tight')
plt.close()

# Save the models
model1_path = os.path.join(args.output, 'comprehensive_polymer_model_EAB1.pkl')
model2_path = os.path.join(args.output, 'comprehensive_polymer_model_EAB2.pkl')
joblib.dump(model1, model1_path)
joblib.dump(model2, model2_path)

# Prediction function for both properties
def predict_elongation_at_break(input_data):
    """
    Predict both EAB1 and EAB2 for polymer blend data
    Parameters:
    - input_data: pandas DataFrame with all features (excluding SMILES columns)
    Returns:
    - tuple: (eab1_prediction, eab2_prediction) in %
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