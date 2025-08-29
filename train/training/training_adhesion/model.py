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
parser = argparse.ArgumentParser(description='Train XGBoost model for adhesion prediction (Dual Properties: Sealing Temperature + Adhesion Strength)')
parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
parser.add_argument('--output', type=str, required=True, help='Output directory path')
args = parser.parse_args()

# Load data
df = pd.read_csv(args.input)

# Define target columns for both properties
target_col1 = 'property1'  # Sealing Temperature
target_col2 = 'property2'  # Adhesion Strength

# Separate features and targets
smiles_cols = [f'SMILES{i}' for i in range(1, 6)]
X = df.drop(columns=[target_col1, target_col2, 'Materials'] + smiles_cols)
y1 = df[target_col1]  # Sealing Temperature
y2 = df[target_col2]  # Adhesion Strength

# Apply log transformation to target values
log_y1 = np.log(y1 + 1e-10)  # Sealing Temperature
log_y2 = np.log(y2 + 1e-10)  # Adhesion Strength

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

# Split the data using train_test_split while ensuring last 5 blends are in testing
last_5_indices = list(range(len(df) - 5, len(df)))

# Remove last 5 from the main pool for train_test_split
remaining_indices = [i for i in range(len(df)) if i not in last_5_indices]
X_remaining = X.iloc[remaining_indices]
log_y1_remaining = log_y1.iloc[remaining_indices]
log_y2_remaining = log_y2.iloc[remaining_indices]

# Use train_test_split on the remaining data
X_temp_train, X_temp_test, y1_temp_train, y1_temp_test, y2_temp_train, y2_temp_test, temp_train_indices, temp_test_indices = train_test_split(
    X_remaining, log_y1_remaining, log_y2_remaining, remaining_indices, 
    test_size=0.2, random_state=42, shuffle=True
)

# Combine: last 5 always in testing, rest split by train_test_split
train_indices = temp_train_indices
test_indices = last_5_indices + temp_test_indices

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]
log_y1_train = log_y1.iloc[train_indices]
log_y1_test = log_y1.iloc[test_indices]
log_y2_train = log_y2.iloc[train_indices]
log_y2_test = log_y2.iloc[test_indices]

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Last 5 blends in testing: {len(last_5_indices)}")

# Create and train model for Sealing Temperature (property1)
print("\n=== TRAINING MODEL FOR SEALING TEMPERATURE (property1) ===")
model1 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ))
])

model1.fit(X_train, log_y1_train)
y1_pred_train = model1.predict(X_train)
y1_pred_test = model1.predict(X_test)

# Create and train model for Adhesion Strength (property2)
print("\n=== TRAINING MODEL FOR ADHESION STRENGTH (property2) ===")
model2 = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    ))
])

model2.fit(X_train, log_y2_train)
y2_pred_train = model2.predict(X_train)
y2_pred_test = model2.predict(X_test)

# Evaluate both models
print("\n=== MODEL PERFORMANCE ===")

# Sealing Temperature Model Performance
print("SEALING TEMPERATURE MODEL:")
print(f"Training R²: {r2_score(log_y1_train, y1_pred_train):.4f}")
print(f"Training MAE: {mean_absolute_error(log_y1_train, y1_pred_train):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(log_y1_train, y1_pred_train)):.4f}")
print(f"Test R²: {r2_score(log_y1_test, y1_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(log_y1_test, y1_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(log_y1_test, y1_pred_test)):.4f}")

print("\nADHESION STRENGTH MODEL:")
print(f"Training R²: {r2_score(log_y2_train, y2_pred_train):.4f}")
print(f"Training MAE: {mean_absolute_error(log_y2_train, y2_pred_train):.4f}")
print(f"Training RMSE: {np.sqrt(mean_squared_error(log_y2_train, y2_pred_train)):.4f}")
print(f"Test R²: {r2_score(log_y2_test, y2_pred_test):.4f}")
print(f"Test MAE: {mean_absolute_error(log_y2_test, y2_pred_test):.4f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(log_y2_test, y2_pred_test)):.4f}")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 15))
plt.suptitle('Adhesion Dual Property Model Results', fontsize=16)

# 1. Sealing Temperature: Actual vs Predicted
plt.subplot(3, 4, 1)
plt.scatter(log_y1_train, y1_pred_train, alpha=0.6, label='Training', color='blue')
plt.scatter(log_y1_test, y1_pred_test, alpha=0.6, label='Test', color='red')
plt.plot([log_y1_test.min(), log_y1_test.max()], [log_y1_test.min(), log_y1_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log(Sealing Temperature)')
plt.ylabel('Predicted Log(Sealing Temperature)')
plt.title('Sealing Temperature Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Adhesion Strength: Actual vs Predicted
plt.subplot(3, 4, 2)
plt.scatter(log_y2_train, y2_pred_train, alpha=0.6, label='Training', color='blue')
plt.scatter(log_y2_test, y2_pred_test, alpha=0.6, label='Test', color='red')
plt.plot([log_y2_test.min(), log_y2_test.max()], [log_y2_test.min(), log_y2_test.max()], 'r--', lw=2)
plt.xlabel('Actual Log(Adhesion Strength)')
plt.ylabel('Predicted Log(Adhesion Strength)')
plt.title('Adhesion Strength Prediction')
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Sealing Temperature Residuals
plt.subplot(3, 4, 3)
residuals1_train = log_y1_train - y1_pred_train
residuals1_test = log_y1_test - y1_pred_test
plt.scatter(y1_pred_train, residuals1_train, alpha=0.6, label='Training', color='blue')
plt.scatter(y1_pred_test, residuals1_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Log(Sealing Temperature)')
plt.ylabel('Residuals')
plt.title('Sealing Temperature Residuals')
plt.legend()
plt.grid(True, alpha=0.3)

# 4. Adhesion Strength Residuals
plt.subplot(3, 4, 4)
residuals2_train = log_y2_train - y2_pred_train
residuals2_test = log_y2_test - y2_pred_test
plt.scatter(y2_pred_train, residuals2_train, alpha=0.6, label='Training', color='blue')
plt.scatter(y2_pred_test, residuals2_test, alpha=0.6, label='Test', color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Log(Adhesion Strength)')
plt.ylabel('Residuals')
plt.title('Adhesion Strength Residuals')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. Feature Importance for Sealing Temperature
plt.subplot(3, 4, 5)
feature_names1 = model1.named_steps['preprocessor'].get_feature_names_out()
importances1 = model1.named_steps['regressor'].feature_importances_
indices1 = np.argsort(importances1)[::-1]
top_features = 15
plt.barh(range(top_features), importances1[indices1[-top_features:]])
plt.yticks(range(top_features), [feature_names1[i].split('__')[-1] for i in indices1[-top_features:]])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - Sealing Temperature')
plt.gca().invert_yaxis()

# 6. Feature Importance for Adhesion Strength
plt.subplot(3, 4, 6)
feature_names2 = model2.named_steps['preprocessor'].get_feature_names_out()
importances2 = model2.named_steps['regressor'].feature_importances_
indices2 = np.argsort(importances2)[::-1]
plt.barh(range(top_features), importances2[indices2[-top_features:]])
plt.yticks(range(top_features), [feature_names2[i].split('__')[-1] for i in indices2[-top_features:]])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - Adhesion Strength')
plt.gca().invert_yaxis()

# 7. Performance Comparison
plt.subplot(3, 4, 7)
properties = ['Sealing Temp', 'Adhesion Strength']
r2_scores = [r2_score(log_y1_test, y1_pred_test), r2_score(log_y2_test, y2_pred_test)]
colors = ['#FF6B6B', '#4ECDC4']
bars = plt.bar(properties, r2_scores, color=colors, alpha=0.7)
for bar, score in zip(bars, r2_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
plt.ylabel('R² Score (Test Set)')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)

# 8. Last 5 Blends Performance
plt.subplot(3, 4, 8)
last_5_X = X.iloc[last_5_indices]
last_5_y1 = log_y1.iloc[last_5_indices]
last_5_y2 = log_y2.iloc[last_5_indices]
last_5_pred1 = model1.predict(last_5_X)
last_5_pred2 = model2.predict(last_5_X)

# Debug: Print the actual values to see what we're working with
print(f"\n=== DEBUG: Last 5 Blends Data ===")
print(f"Last 5 indices: {last_5_indices}")
print(f"Last 5 actual sealing temps (log): {last_5_y1.values}")
print(f"Last 5 predicted sealing temps (log): {last_5_pred1}")
print(f"Last 5 actual adhesion (log): {last_5_y2.values}")
print(f"Last 5 predicted adhesion (log): {last_5_pred2}")

# Convert to original scale for visualization
orig_y1 = np.exp(last_5_y1)
orig_pred1 = np.exp(last_5_pred1)
orig_y2 = np.exp(last_5_y2)
orig_pred2 = np.exp(last_5_pred2)

print(f"Last 5 actual sealing temps (original): {orig_y1.values}")
print(f"Last 5 predicted sealing temps (original): {orig_pred1}")
print(f"Last 5 actual adhesion (original): {orig_y2.values}")
print(f"Last 5 predicted adhesion (original): {orig_pred2}")

# Ensure we have exactly 5 points
assert len(orig_y1) == 5, f"Expected 5 points, got {len(orig_y1)}"
assert len(orig_pred1) == 5, f"Expected 5 predictions, got {len(orig_pred1)}"

# Plot each point individually to ensure visibility
for i in range(len(orig_y1)):
    plt.scatter(orig_y1.iloc[i], orig_pred1[i], color='red', s=100, alpha=0.7, 
                label='Sealing Temp' if i == 0 else "")
    plt.scatter(orig_y2.iloc[i], orig_pred2[i], color='blue', s=100, alpha=0.7, 
                label='Adhesion Strength' if i == 0 else "")
    # Add blend labels
    plt.annotate(f'B{i+1}', (orig_y1.iloc[i], orig_pred1[i]), 
                xytext=(3, 3), textcoords='offset points', fontsize=8, fontweight='bold')

plt.plot([min(orig_y1.min(), orig_y2.min()), max(orig_y1.max(), orig_y2.max())], 
         [min(orig_y1.min(), orig_y2.min()), max(orig_y1.max(), orig_y2.max())], 'k--', lw=2)
plt.xlabel('Actual Values (Original Scale)')
plt.ylabel('Predicted Values (Original Scale)')
plt.title('Last 5 Blends Performance')
plt.legend()
plt.grid(True, alpha=0.3)

# 9. Data Distribution
plt.subplot(3, 4, 9)
plt.hist(log_y1_train, bins=20, alpha=0.7, label='Training', color='blue')
plt.hist(log_y1_test, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Log(Sealing Temperature)')
plt.ylabel('Frequency')
plt.title('Sealing Temperature Distribution')
plt.legend()

# 10. Data Distribution
plt.subplot(3, 4, 10)
plt.hist(log_y2_train, bins=20, alpha=0.7, label='Training', color='blue')
plt.hist(log_y2_test, bins=20, alpha=0.7, label='Test', color='red')
plt.xlabel('Log(Adhesion Strength)')
plt.ylabel('Frequency')
plt.title('Adhesion Strength Distribution')
plt.legend()

# 11. Model Summary
plt.subplot(3, 4, 11)
plt.text(0.1, 0.9, f'Sealing Temperature Model:', fontsize=12, fontweight='bold')
plt.text(0.1, 0.8, f'R²: {r2_score(log_y1_test, y1_pred_test):.3f}', fontsize=10)
plt.text(0.1, 0.7, f'MAE: {mean_absolute_error(log_y1_test, y1_pred_test):.3f}', fontsize=10)
plt.text(0.1, 0.6, f'RMSE: {np.sqrt(mean_squared_error(log_y1_test, y1_pred_test)):.3f}', fontsize=10)
plt.text(0.1, 0.5, f'Training: {len(X_train)}', fontsize=10)
plt.text(0.1, 0.4, f'Test: {len(X_test)}', fontsize=10)
plt.text(0.1, 0.3, f'Features: {X_train.shape[1]}', fontsize=10)
plt.text(0.1, 0.2, f'Last 5 in Test: {len(last_5_indices)}', fontsize=10)
plt.title('Model Summary')
plt.axis('off')

# 12. Model Summary
plt.subplot(3, 4, 12)
plt.text(0.1, 0.9, f'Adhesion Strength Model:', fontsize=12, fontweight='bold')
plt.text(0.1, 0.8, f'R²: {r2_score(log_y2_test, y2_pred_test):.3f}', fontsize=10)
plt.text(0.1, 0.7, f'MAE: {mean_absolute_error(log_y2_test, y2_pred_test):.3f}', fontsize=10)
plt.text(0.1, 0.6, f'RMSE: {np.sqrt(mean_squared_error(log_y2_test, y2_pred_test)):.3f}', fontsize=10)
plt.text(0.1, 0.5, f'Training: {len(X_train)}', fontsize=10)
plt.text(0.1, 0.4, f'Test: {len(X_test)}', fontsize=10)
plt.text(0.1, 0.3, f'Features: {X_train.shape[1]}', fontsize=10)
plt.text(0.1, 0.2, f'Last 5 in Test: {len(last_5_indices)}', fontsize=10)
plt.title('Model Summary')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'comprehensive_polymer_model_results.png'), dpi=300, bbox_inches='tight')
plt.show()

# Create separate plot for last 5 blends performance
plt.figure(figsize=(15, 6))

# Sealing Temperature
plt.subplot(1, 2, 1)
print(f"\n=== DEBUG: Separate Plot - Sealing Temperature ===")
print(f"orig_y1 length: {len(orig_y1)}, values: {orig_y1.values}")
print(f"orig_pred1 length: {len(orig_pred1)}, values: {orig_pred1}")

# Plot each point individually to ensure visibility
for i in range(len(orig_y1)):
    plt.scatter(orig_y1.iloc[i], orig_pred1[i], color='red', s=150, alpha=0.8, 
                label=f'Blend {i+1}' if i == 0 else "")
    # Add data point labels
    plt.annotate(f'B{i+1}', (orig_y1.iloc[i], orig_pred1[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

plt.plot([orig_y1.min(), orig_y1.max()], [orig_y1.min(), orig_y1.max()], 'r--', lw=2)
plt.xlabel('Actual Sealing Temperature (°C)')
plt.ylabel('Predicted Sealing Temperature (°C)')
plt.title('Last 5 Blends - Sealing Temperature')

# Add metrics
mae1 = mean_absolute_error(orig_y1, orig_pred1)
rmse1 = np.sqrt(mean_squared_error(orig_y1, orig_pred1))
r21 = r2_score(orig_y1, orig_pred1)
plt.text(0.05, 0.95, f'MAE: {mae1:.2f}\nRMSE: {rmse1:.2f}\nR²: {r21:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Adhesion Strength
plt.subplot(1, 2, 2)
print(f"\n=== DEBUG: Separate Plot - Adhesion Strength ===")
print(f"orig_y2 length: {len(orig_y2)}, values: {orig_y2.values}")
print(f"orig_pred2 length: {len(orig_pred2)}, values: {orig_pred2}")

# Plot each point individually to ensure visibility
for i in range(len(orig_y2)):
    plt.scatter(orig_y2.iloc[i], orig_pred2[i], color='blue', s=150, alpha=0.8,
                label=f'Blend {i+1}' if i == 0 else "")
    # Add data point labels
    plt.annotate(f'B{i+1}', (orig_y2.iloc[i], orig_pred2[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

plt.plot([orig_y2.min(), orig_y2.max()], [orig_y2.min(), orig_y2.max()], 'b--', lw=2)
plt.xlabel('Actual Adhesion Strength (N/15mm)')
plt.ylabel('Predicted Adhesion Strength (N/15mm)')
plt.title('Last 5 Blends - Adhesion Strength')

# Add metrics
mae2 = mean_absolute_error(orig_y2, orig_pred2)
rmse2 = np.sqrt(mean_squared_error(orig_y2, orig_pred2))
r22 = r2_score(orig_y2, orig_pred2)
plt.text(0.05, 0.95, f'MAE: {mae2:.2f}\nRMSE: {rmse2:.2f}\nR²: {r22:.3f}', 
         transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(args.output, 'last_5_blends_performance.png'), dpi=300, bbox_inches='tight')
plt.show()

# Save both models
model1_path = os.path.join(args.output, 'sealing_temperature_model.pkl')
model2_path = os.path.join(args.output, 'adhesion_strength_model.pkl')
joblib.dump(model1, model1_path)
joblib.dump(model2, model2_path)

print(f"\n=== MODELS SAVED ===")
print(f"Sealing Temperature Model: {model1_path}")
print(f"Adhesion Strength Model: {model2_path}")

# Save feature importance data
feature_importance_df = pd.DataFrame({
    'Feature': feature_names1,
    'Sealing_Temperature_Importance': importances1,
    'Adhesion_Strength_Importance': importances2
})
feature_importance_df = feature_importance_df.sort_values('Sealing_Temperature_Importance', ascending=False)
feature_importance_df.to_csv(os.path.join(args.output, 'feature_importance.csv'), index=False)

print(f"\n=== FEATURE IMPORTANCE SAVED ===")
print(f"Feature importance CSV: {os.path.join(args.output, 'feature_importance.csv')}")

print(f"\n=== TRAINING COMPLETE ===")
print("✅ Dual property adhesion models successfully trained!")
print("✅ Sealing Temperature (property1) and Adhesion Strength (property2) models created")
print("✅ Last 5 blends automatically placed in testing set")
print("✅ Comprehensive visualizations and feature importance analysis completed")
print(f"✅ All files saved to: {args.output}") 