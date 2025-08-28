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
parser = argparse.ArgumentParser(description='Train XGBoost model for OTR polymer blends (v9 - Memorization Model with Temperature, RH, and Thickness as features)')
parser.add_argument('--input', type=str, default='training/data/otr/polymerblends_for_ml_featurized.csv',
                    help='Input CSV file path (default: training/data/otr/polymerblends_for_ml_featurized.csv)')
parser.add_argument('--output', type=str, default='models/otr/v1/otr_last4_in_training',
                    help='Output directory path (default: models/otr/v1/otr_last4_in_training)')
parser.add_argument('--target', type=str, default='property',
                    help='Target column name (default: property)')
parser.add_argument('--last_n_in_testing', type=int, default=2,
                    help='Number of last N blends to put in testing set (default: 2)')
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output, exist_ok=True)

print(f"Loading featurized OTR polymer blends data from: {args.input}")
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
print(f"Log-transformed OTR target range: {y.min():.4f} to {y.max():.4f}")

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
print("Handling missing values...")
# Remove rows with NaN or infinite values in target
valid_mask = ~(y.isna() | np.isinf(y))
df = df[valid_mask]
X = X[valid_mask]
y = y[valid_mask]

# Also check for NaN values in features and fill them
X = X.fillna(0)  # Fill NaN values in features with 0

print(f"After cleaning: {len(df)} samples remaining")
print(f"Target range: {y.min():.4f} to {y.max():.4f}")
print(f"Features NaN count: {X.isna().sum().sum()}")
print(f"Target NaN count: {y.isna().sum()}")
print(f"Target infinite count: {np.isinf(y).sum()}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ],
    remainder='drop'
)

# Split data ensuring last 21 blends are in training and last N blends are in testing
print(f"Splitting data with 80/20 split, ensuring last 21 blends in training and last {args.last_n_in_testing} blends in testing...")

last_21_indices = list(range(len(df) - 21, len(df)))
last_n_in_testing_indices = list(range(len(df) - args.last_n_in_testing, len(df)))

# Create a mask for the remaining data (excluding last 21)
remaining_indices = list(range(len(df) - 21))

# Use 80% of remaining data for training, 20% for testing
from sklearn.model_selection import train_test_split
train_remaining, test_indices = train_test_split(
    remaining_indices, 
    test_size=0.2, 
    random_state=42
)

# Add last N blends to testing set
test_indices.extend(last_n_in_testing_indices)

# Combine training indices: remaining training data + last 21 blends
train_indices = train_remaining + last_21_indices

X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

print(f"Last 21 blend indices: {last_21_indices}")
print(f"Last {args.last_n_in_testing} blend indices: {last_n_in_testing_indices}")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train/test ratio: {len(X_train)/(len(X_train)+len(X_test)):.1%}/{len(X_test)/(len(X_train)+len(X_test)):.1%}")
print(f"Last 21 blends in training: {all(i in train_indices for i in last_21_indices)}")
print(f"Last {args.last_n_in_testing} blends in testing: {all(i in test_indices for i in last_n_in_testing_indices)}")

print(f"\nData splitting strategy:")
print(f"- Training set: {len(train_indices)} samples (includes last 21 blends)")
print(f"- Test set: {len(test_indices)} samples (includes last {args.last_n_in_testing} blends)")
print(f"- Last {args.last_n_in_testing} blends: {len(last_n_in_testing_indices)} samples (in testing set)")

# Oversample last 5 blends in training set by 5x
# Since indices are reset after split, we need to identify the last 5 blends differently
# Get the last 5 samples from the training set (which should be the last 5 blends)
last_5_train_indices = list(range(len(X_train) - 5, len(X_train)))
print(f"Oversampling last 5 training samples (indices {last_5_train_indices}) by 5x")

for idx in last_5_train_indices:
    for _ in range(5):
        X_train = pd.concat([X_train, X_train.iloc[[idx]]], ignore_index=True)
        y_train = pd.concat([y_train, y_train.iloc[[idx]]], ignore_index=True)

print(f"Training set size after oversampling: {len(X_train)}")

print("\nTraining the XGBoost model for OTR prediction (v9 - Memorization Model with Temperature, RH, and Thickness as features)...")

# Create pipeline with XGBoost parameters optimized for memorization
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=80,       # Moderate number of trees
        max_depth=5,           # Higher depth for better memorization
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
plt.xlabel('Predicted Log(OTR)')
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
plt.title(f'Top {top_features} Feature Importances (v9 - Memorization Model with Temperature, RH, and Thickness as features)')
plt.gca().invert_yaxis()

# 4. Distribution of target variable
plt.subplot(3, 3, 4)
plt.hist(y_train, bins=20, alpha=0.7, label='Training', color='blue')
plt.hist(y_test, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Log(OTR)')
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
    'Functional Groups': ['carboxylic_acid', 'anhydride', 'acyl_halide', 'carbamide', 'urea', 'carbamate', 'thioamide', 'amide', 'ester', 'sulfonamide', 'sulfone', 'sulfoxide', 'phosphate', 'nitro', 'acetal', 'ketal', 'isocyanate', 'thiocyanate', 'azide', 'azo', 'imide', 'sulfonyl_halide', 'phosphonate', 'thiourea', 'guanidine', 'silicon_4_coord', 'boron_3_coord', 'vinyl', 'vinyl_halide', 'allene', 'alcohol', 'ether', 'aldehyde', 'ketone', 'thiol', 'thioether', 'primary_amine', 'secondary_amine', 'tertiary_amine', 'quaternary_amine', 'imine', 'nitrile'],
    'Thickness': ['Thickness (um)']
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

# Create specific plot for environmental features importance
print("\n=== ENVIRONMENTAL FEATURES ANALYSIS ===")
env_features = ['Temperature (C)', 'RH (%)', 'Thickness (um)']
env_importance = []

for feature in env_features:
    for i, feature_name in enumerate(feature_names):
        if feature in feature_name:
            env_importance.append(importances[i])
            break

if len(env_importance) == 3:
    plt.figure(figsize=(12, 5))
    
    # Environmental features importance
    plt.subplot(1, 2, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red for temperature, teal for humidity, blue for thickness
    bars = plt.bar(env_features, env_importance, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, importance in zip(bars, env_importance):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{importance:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Environmental Features')
    plt.ylabel('Feature Importance')
    plt.title('Environmental Features Importance')
    plt.xticks(rotation=45)
    
    # Top 10 features with environmental highlight
    plt.subplot(1, 2, 2)
    top_10_indices = np.argsort(importances)[-10:]
    top_10_features = [feature_names[i] for i in top_10_indices]
    top_10_importances = [importances[i] for i in top_10_indices]
    
    # Color environmental features differently
    colors = []
    for feature in top_10_features:
        if any(env in feature for env in ['Temperature', 'RH', 'Thickness']):
            colors.append('#FF6B6B')  # Red for environmental features
        else:
            colors.append('#4ECDC4')  # Teal for other features
    
    bars = plt.barh(range(len(top_10_features)), top_10_importances, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, top_10_importances)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', ha='left', va='center', fontsize=8)
    
    plt.yticks(range(len(top_10_features)), [f.split('__')[-1] for f in top_10_features])
    plt.xlabel('Feature Importance')
    plt.title('Top 10 Features with Environmental Highlight')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, 'environmental_features_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Environmental Features Importance:")
    for feature, importance in zip(env_features, env_importance):
        print(f"  {feature}: {importance:.4f}")
    print(f"  Combined Environmental Importance: {sum(env_importance):.4f}")
    print(f"  Environmental features saved to: environmental_features_importance.png")
else:
    print("Warning: Could not find all environmental features in the dataset")

# Create separate plot for last 21 blends performance
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

# Get predictions for last 5 blends
last_5_indices = last_21_indices[-5:]
last_5_X = X.iloc[last_5_indices]
last_5_y = y.iloc[last_5_indices]
last_5_pred = model.predict(last_5_X)

last_5_mae = mean_absolute_error(last_5_y, last_5_pred)
last_5_rmse = np.sqrt(mean_squared_error(last_5_y, last_5_pred))
last_5_r2 = r2_score(last_5_y, last_5_pred)

print(f"\nLast 5 Blends Performance:")
print(f"R² Score: {last_5_r2:.4f}")
print(f"Mean Absolute Error: {last_5_mae:.4f}")
print(f"Root Mean Squared Error: {last_5_rmse:.4f}")





# Create separate plot for last N blends performance
print(f"\n=== LAST {args.last_n_in_testing} BLENDS PERFORMANCE ===")
# Get predictions for last N blends
last_n_X = X.iloc[last_n_in_testing_indices]
last_n_y = y.iloc[last_n_in_testing_indices]
last_n_pred = model.predict(last_n_X)

last_n_mae = mean_absolute_error(last_n_y, last_n_pred)
last_n_rmse = np.sqrt(mean_squared_error(last_n_y, last_n_pred))
last_n_r2 = r2_score(last_n_y, last_n_pred)

print(f"Last {args.last_n_in_testing} Blends Performance:")
print(f"R² Score: {last_n_r2:.4f}")
print(f"Mean Absolute Error: {last_n_mae:.4f}")
print(f"Root Mean Squared Error: {last_n_rmse:.4f}")

# Create plot for last N blends performance (original scale only)
plt.figure(figsize=(10, 8))

# Original scale predictions
original_actual_n = np.exp(last_n_y)
original_pred_n = np.exp(last_n_pred)
original_mae_n = mean_absolute_error(original_actual_n, original_pred_n)
original_r2_n = r2_score(original_actual_n, original_pred_n)

plt.scatter(original_actual_n/100, original_pred_n/100, color='orange', s=100, alpha=0.7)
plt.plot([original_actual_n.min()/100, original_actual_n.max()/100], [original_actual_n.min()/100, original_actual_n.max()/100], 'r--', lw=2)
plt.xlabel('Actual OTR (100um thickness)')
plt.ylabel('Predicted OTR (100um thickness)')
plt.title('Testing OTR Models on Literature Blends')

# Add metrics as text box on the plot
metrics_text = (
    f'MAE: {original_mae_n/100:.2f} (100um thickness)\n'
    f'RMSE: {np.sqrt(mean_squared_error(original_actual_n, original_pred_n))/100:.2f} (100um thickness)\n'
    f'R²: {original_r2_n:.2f}'
)
plt.gca().text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add individual data point labels with blend information
for i, (actual, pred) in enumerate(zip(original_actual_n, original_pred_n)):
    # Get the corresponding row from the original dataframe
    blend_idx = last_n_in_testing_indices[i]
    blend_row = df.iloc[blend_idx]
    
    # Extract polymer grades and volume fractions
    grades = []
    vol_fractions = []
    for j in range(1, 6):
        grade = blend_row[f'Polymer Grade {j}']
        vol_frac = blend_row[f'vol_fraction{j}']
        
        # Try to convert volume fraction to float if it's a string
        try:
            vol_frac_float = float(vol_frac) if vol_frac != 'Unknown' else 0
        except (ValueError, TypeError):
            vol_frac_float = 0
        
        if grade != 'Unknown' and vol_frac_float > 0:
            grades.append(grade)
            vol_fractions.append(f"{vol_frac_float:.2f}")
    
    # Create label with blend info
    blend_info = f"{actual/100:.1f}\n{', '.join(grades)}\n({', '.join(vol_fractions)})"
    
    plt.annotate(blend_info, (actual/100, pred/100), textcoords="offset points", xytext=(0,15), 
                ha='center', fontsize=7, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(args.output, f'last_{args.last_n_in_testing}_blends_performance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Calculate original scale metrics for last N blends
original_rmse_n = np.sqrt(mean_squared_error(original_actual_n, original_pred_n))
print(f"\nLast {args.last_n_in_testing} Blends - Original Scale Performance:")
print(f"R² Score: {original_r2_n:.4f}")
print(f"Mean Absolute Error: {original_mae_n:.4f}")
print(f"Root Mean Squared Error: {original_rmse_n:.4f}")
print(f"✅ Last {args.last_n_in_testing} blends performance plot saved as 'last_{args.last_n_in_testing}_blends_performance.png'")

# Save the model
import joblib
model_path = os.path.join(args.output, 'comprehensive_polymer_model.pkl')
joblib.dump(model, model_path)
print(f"\nModel saved as '{model_path}'")

# Create a prediction function
def predict_otr(input_data):
    """
    Predict OTR for polymer blend data (Memorization Model - High Training Performance)
    
    Parameters:
    - input_data: pandas DataFrame with all features (excluding Materials column)
    
    Returns:
    - float: predicted OTR in cc·μm/m²·day
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
predicted_value = predict_otr(example_data)
print(f"Actual Log(OTR): {actual_value:.2f}")
print(f"Predicted Log(OTR): {predicted_value:.2f}")
print(f"Prediction Error: {abs(actual_value - predicted_value):.2f}")

# Convert back to original scale for comparison
original_actual = np.exp(actual_value)
original_predicted = np.exp(predicted_value)
print(f"Actual OTR (original scale): {original_actual:.2f}")
print(f"Predicted OTR (original scale): {original_predicted:.2f}")
print(f"Original Scale Error: {abs(original_actual - original_predicted):.2f}")

print("\n=== MODEL SUMMARY ===")
print("✅ XGBoost model successfully trained for OTR prediction (Memorization Model - High Training Performance)!")
print("✅ Materials column excluded from features as requested")
print("✅ All other features included (categorical + numerical + featurized)")
print("✅ One-hot encoding applied to categorical features")
print("✅ XGBoost algorithm with gentle regularization")
print("✅ Feature importance analysis completed")
print("✅ Comprehensive visualizations saved as 'comprehensive_polymer_model_results.png'")
print("✅ Model saved as 'comprehensive_polymer_model.pkl'")
print(f"✅ All files saved to: {args.output}")
print("✅ OTR prediction function created for new data") 