#!/usr/bin/env python3
"""
Differentiable Label Optimization (DLO) for Polymer Blend Properties
Implements a meta-learning approach where some labels are fixed (hard) and others are trainable (soft).
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

class PolymerDataset(Dataset):
    """Custom dataset for polymer blend data with hard/soft label support."""
    
    def __init__(self, features: np.ndarray, hard_labels: np.ndarray, 
                 soft_label_indices: List[int], device: str = 'cpu'):
        self.features = torch.FloatTensor(features).to(device)
        self.hard_labels = torch.FloatTensor(hard_labels).to(device)
        self.soft_label_indices = soft_label_indices
        self.device = device
        
        # Initialize soft labels as trainable parameters
        self.soft_labels = nn.Parameter(
            torch.FloatTensor(hard_labels[soft_label_indices]).to(device),
            requires_grad=True
        )
        
    def __len__(self):
        return len(self.features)
    
    def get_labels(self) -> torch.Tensor:
        """Get combined hard and soft labels."""
        labels = self.hard_labels.clone()
        labels[self.soft_label_indices] = self.soft_labels
        return labels
    
    def __getitem__(self, idx):
        return self.features[idx], self.get_labels()[idx]

class PolymerPredictor(nn.Module):
    """Neural network for predicting polymer properties."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        super(PolymerPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))  # 2 outputs: property1, property2
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class DifferentiableLabelOptimizer:
    """Main class for Differentiable Label Optimization."""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.scaler = StandardScaler()
        self.model = None
        self.dataset = None
        self.optimizer = None
        self.label_optimizer = None
        
    def prepare_data(self, data_file: str = 'data/training.csv', 
                    use_custom_labels: bool = True) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data and identify which labels should be soft (trainable).
        
        Args:
            data_file: Path to the input CSV file
            use_custom_labels: Whether to use custom label assignments from CSV
        """
        print("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(data_file)
        df = df.fillna(0)
        
        # Preserve label columns as strings before numeric conversion
        label_cols = ['property1_label', 'property2_label']
        label_data = {}
        for col in label_cols:
            if col in df.columns:
                label_data[col] = df[col].copy()
        
        # Handle categorical variables
        categorical_cols = ['enzyme kinetics', 'TUV Industrial or BPI', 'TUV Home']
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col.replace(' ', '_'))
                df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
        
        # Convert to numeric, excluding label columns
        exclude_from_numeric = label_cols + ['Materials', 'Polymer Grade 1', 'Polymer Grade 2', 
                                           'Polymer Grade 3', 'Polymer Grade 4', 'Polymer Grade 5',
                                           'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5']
        
        numeric_cols = [col for col in df.columns if col not in exclude_from_numeric]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Restore label columns
        for col in label_cols:
            if col in label_data:
                df[col] = label_data[col]
        
        # Extract features (exclude metadata and target columns)
        exclude_cols = ['Materials', 'Polymer Grade 1', 'Polymer Grade 2', 'Polymer Grade 3', 
                       'Polymer Grade 4', 'Polymer Grade 5', 'SMILES1', 'SMILES2', 'SMILES3', 
                       'SMILES4', 'SMILES5', 'vol_fraction1', 'vol_fraction2', 'vol_fraction3', 
                       'vol_fraction4', 'vol_fraction5', 'property1', 'property2', 
                       'property1_label', 'property2_label']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols].values
        labels = df[['property1', 'property2']].values
        
        # Scale features
        features = self.scaler.fit_transform(features)
        
        # Determine which labels should be soft based on custom assignments
        if use_custom_labels and 'property1_label' in df.columns and 'property2_label' in df.columns:
            # Use custom label assignments from CSV
            soft_indices = []
            
            for i in range(len(df)):
                # Check if either property1 or property2 is marked as "soft"
                if df.iloc[i]['property1_label'] == 'soft' or df.iloc[i]['property2_label'] == 'soft':
                    soft_indices.append(i)
            
            print(f"Using custom label assignments from CSV")
            print(f"Property1 labels: {df['property1_label'].value_counts().to_dict()}")
            print(f"Property2 labels: {df['property2_label'].value_counts().to_dict()}")
            
        else:
            # Fallback to automatic selection (original method)
            n_samples = len(labels)
            n_soft = int(n_samples * 0.3)  # 30% soft labels
            soft_indices = np.random.choice(n_samples, n_soft, replace=False).tolist()
            print(f"Using automatic soft label selection (30% of samples)")
        
        print(f"Data shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Soft labels: {len(soft_indices)} out of {len(labels)} samples")
        print(f"Soft label ratio: {len(soft_indices)/len(labels):.2f}")
        
        return features, labels, soft_indices
    
    def create_model(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
        """Create the neural network model."""
        self.model = PolymerPredictor(input_dim, hidden_dims).to(self.device)
        return self.model
    
    def create_dataset(self, features: np.ndarray, labels: np.ndarray, 
                      soft_indices: List[int]) -> PolymerDataset:
        """Create the dataset with hard/soft label support."""
        self.dataset = PolymerDataset(features, labels, soft_indices, self.device)
        return self.dataset
    
    def setup_optimizers(self, model_lr: float = 1e-3, label_lr: float = 1e-2):
        """Setup optimizers for both model parameters and soft labels."""
        if self.model is None or self.dataset is None:
            raise ValueError("Model and dataset must be created first")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_lr)
        self.label_optimizer = optim.Adam([self.dataset.soft_labels], lr=label_lr)
        
        print(f"Model optimizer: Adam(lr={model_lr})")
        print(f"Label optimizer: Adam(lr={label_lr})")
    
    def train_epoch(self, dataloader: DataLoader, 
                   hard_label_weight: float = 1.0,
                   soft_label_weight: float = 0.5) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_features, batch_labels in dataloader:
            self.optimizer.zero_grad()
            self.label_optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_features)
            
            # Calculate loss
            batch_loss = nn.MSELoss()(predictions, batch_labels)
            
            # Backward pass
            batch_loss.backward()
            
            # Update both model parameters and soft labels
            self.optimizer.step()
            self.label_optimizer.step()
            
            total_loss += batch_loss.item()
            n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'hard_loss': total_loss / n_batches,  # Simplified for now
            'soft_loss': total_loss / n_batches   # Simplified for now
        }
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).to(self.device)
            predictions = self.model(features_tensor).cpu().numpy()
            
            # Calculate metrics for each property
            metrics = {}
            for i, prop_name in enumerate(['property1', 'property2']):
                r2 = r2_score(labels[:, i], predictions[:, i])
                mae = mean_absolute_error(labels[:, i], predictions[:, i])
                rmse = np.sqrt(mean_squared_error(labels[:, i], predictions[:, i]))
                
                metrics[f'{prop_name}_r2'] = r2
                metrics[f'{prop_name}_mae'] = mae
                metrics[f'{prop_name}_rmse'] = rmse
            
            # Overall metrics
            metrics['overall_r2'] = np.mean([metrics['property1_r2'], metrics['property2_r2']])
            metrics['overall_mae'] = np.mean([metrics['property1_mae'], metrics['property2_mae']])
            
        return metrics, predictions
    
    def train(self, features: np.ndarray, labels: np.ndarray, 
              soft_indices: List[int], epochs: int = 100, 
              batch_size: int = 16, patience: int = 20,
              hard_label_weight: float = 1.0,
              soft_label_weight: float = 0.5) -> Dict[str, List[float]]:
        """Train the model with differentiable label optimization."""
        
        # Create model and dataset
        self.create_model(features.shape[1])
        self.create_dataset(features, labels, soft_indices)
        self.setup_optimizers()
        
        # Create dataloader
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Training history
        history = {
            'total_loss': [], 'hard_loss': [], 'soft_loss': [],
            'val_r2': [], 'val_mae': []
        }
        
        best_val_r2 = -np.inf
        patience_counter = 0
        
        print(f"\nTraining for {epochs} epochs...")
        print(f"Hard label weight: {hard_label_weight}")
        print(f"Soft label weight: {soft_label_weight}")
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(dataloader, hard_label_weight, soft_label_weight)
            
            # Evaluate
            val_metrics, _ = self.evaluate(features, labels)
            
            # Store history
            history['total_loss'].append(train_metrics['total_loss'])
            history['hard_loss'].append(train_metrics['hard_loss'])
            history['soft_loss'].append(train_metrics['soft_loss'])
            history['val_r2'].append(val_metrics['overall_r2'])
            history['val_mae'].append(val_metrics['overall_mae'])
            
            # Early stopping
            if val_metrics['overall_r2'] > best_val_r2:
                best_val_r2 = val_metrics['overall_r2']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_metrics['total_loss']:.4f} - "
                      f"Val R²: {val_metrics['overall_r2']:.4f}")
        
        return history
    
    def get_soft_label_changes(self, original_labels: np.ndarray) -> Dict[str, np.ndarray]:
        """Get the changes in soft labels from original to optimized values."""
        if self.dataset is None:
            raise ValueError("Dataset not created")
        
        original_soft = original_labels[self.dataset.soft_label_indices]
        optimized_soft = self.dataset.soft_labels.detach().cpu().numpy()
        
        changes = optimized_soft - original_soft
        
        return {
            'original': original_soft,
            'optimized': optimized_soft,
            'changes': changes,
            'indices': self.dataset.soft_label_indices
        }
    
    def plot_results(self, features: np.ndarray, labels: np.ndarray, 
                    history: Dict[str, List[float]], 
                    label_changes: Dict[str, np.ndarray]):
        """Create comprehensive visualization plots."""
        
        # Get final predictions
        final_metrics, predictions = self.evaluate(features, labels)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Training history
        axes[0, 0].plot(history['total_loss'], label='Total Loss')
        axes[0, 0].plot(history['hard_loss'], label='Hard Label Loss')
        axes[0, 0].plot(history['soft_loss'], label='Soft Label Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss History')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Validation metrics
        axes[0, 1].plot(history['val_r2'], label='R²')
        axes[0, 1].plot(history['val_mae'], label='MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Metric Value')
        axes[0, 1].set_title('Validation Metrics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Property1 predictions vs actual
        axes[0, 2].scatter(labels[:, 0], predictions[:, 0], alpha=0.6)
        axes[0, 2].plot([labels[:, 0].min(), labels[:, 0].max()], 
                       [labels[:, 0].min(), labels[:, 0].max()], 'r--', lw=2)
        axes[0, 2].set_xlabel('Actual Property1')
        axes[0, 2].set_ylabel('Predicted Property1')
        axes[0, 2].set_title(f'Property1 Predictions (R² = {final_metrics["property1_r2"]:.3f})')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Property2 predictions vs actual
        axes[1, 0].scatter(labels[:, 1], predictions[:, 1], alpha=0.6)
        axes[1, 0].plot([labels[:, 1].min(), labels[:, 1].max()], 
                       [labels[:, 1].min(), labels[:, 1].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Property2')
        axes[1, 0].set_ylabel('Predicted Property2')
        axes[1, 0].set_title(f'Property2 Predictions (R² = {final_metrics["property2_r2"]:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Soft label changes
        if len(label_changes['changes']) > 0:
            axes[1, 1].scatter(label_changes['original'], label_changes['optimized'], alpha=0.6)
            axes[1, 1].plot([label_changes['original'].min(), label_changes['original'].max()], 
                           [label_changes['original'].min(), label_changes['original'].max()], 'r--', lw=2)
            axes[1, 1].set_xlabel('Original Soft Labels')
            axes[1, 1].set_ylabel('Optimized Soft Labels')
            axes[1, 1].set_title('Soft Label Optimization')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Label change distribution
        if len(label_changes['changes']) > 0:
            axes[1, 2].hist(label_changes['changes'].flatten(), bins=20, alpha=0.7)
            axes[1, 2].set_xlabel('Label Change')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].set_title('Distribution of Soft Label Changes')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('differentiable_label_optimization_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, features: np.ndarray, labels: np.ndarray, 
                    label_changes: Dict[str, np.ndarray], 
                    filename: str = 'dlo_predictions.csv'):
        """Save predictions and label changes to CSV."""
        
        # Get final predictions
        _, predictions = self.evaluate(features, labels)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'sample_index': range(len(features)),
            'actual_property1': labels[:, 0],
            'predicted_property1': predictions[:, 0],
            'actual_property2': labels[:, 1],
            'predicted_property2': predictions[:, 1],
            'is_soft_label': [i in label_changes['indices'] for i in range(len(features))]
        })
        
        # Add soft label information
        if len(label_changes['changes']) > 0:
            soft_df = pd.DataFrame({
                'sample_index': label_changes['indices'],
                'original_property1': label_changes['original'][:, 0],
                'optimized_property1': label_changes['optimized'][:, 0],
                'property1_change': label_changes['changes'][:, 0],
                'original_property2': label_changes['original'][:, 1],
                'optimized_property2': label_changes['optimized'][:, 1],
                'property2_change': label_changes['changes'][:, 1]
            })
            
            results_df = results_df.merge(soft_df, on='sample_index', how='left')
        
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return results_df
    
    def save_model(self, save_dir: str = 'models/v1/', model_name: str = 'dlo_model'):
        """
        Save the trained model, scaler, and metadata.
        
        Args:
            save_dir: Directory to save the model
            model_name: Name for the model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(save_dir, f'{model_name}.pth')
        torch.save(self.model.state_dict(), model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save optimized soft labels if available
        optimized_labels = None
        if self.dataset is not None:
            optimized_labels = self.dataset.soft_labels.detach().cpu().numpy()
        
        # Save model architecture and metadata
        metadata = {
            'input_dim': self.model.network[0].in_features,
            'hidden_dims': [self.model.network[0].out_features] + 
                          [self.model.network[i].out_features for i in range(3, len(self.model.network)-1, 3)],
            'output_dim': 2,
            'device': self.device,
            'model_class': 'PolymerPredictor',
            'optimized_labels': optimized_labels.tolist() if optimized_labels is not None else None
        }
        
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {save_dir}")
        print(f"  - Model weights: {model_path}")
        print(f"  - Scaler: {scaler_path}")
        print(f"  - Metadata: {metadata_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }
    
    def load_model(self, save_dir: str = 'models/v1/', model_name: str = 'dlo_model'):
        """
        Load a previously saved model, scaler, and metadata.
        
        Args:
            save_dir: Directory containing the saved model
            model_name: Name of the model files
        """
        # Load metadata
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Store metadata for later use
        self._metadata = metadata
        
        # Create model with same architecture
        self.model = PolymerPredictor(
            input_dim=metadata['input_dim'],
            hidden_dims=metadata['hidden_dims']
        ).to(self.device)
        
        # Load model weights
        model_path = os.path.join(save_dir, f'{model_name}.pth')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # Load scaler
        scaler_path = os.path.join(save_dir, f'{model_name}_scaler.pkl')
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"Model loaded from {save_dir}")
        print(f"  - Input dimension: {metadata['input_dim']}")
        print(f"  - Hidden dimensions: {metadata['hidden_dims']}")
        print(f"  - Output dimension: {metadata['output_dim']}")
        
        return metadata
    
    def predict(self, features: np.ndarray, use_scaler: bool = True, features_already_scaled: bool = False) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            features: Input features (n_samples, n_features)
            use_scaler: Whether to apply the fitted scaler to features
            features_already_scaled: Whether the features are already scaled
            
        Returns:
            Predictions for property1 and property2
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        self.model.eval()
        
        # Scale features if requested and not already scaled
        if use_scaler and self.scaler is not None and not features_already_scaled:
            features = self.scaler.transform(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(features_tensor).cpu().numpy()
        
        return predictions
    
    def predict_with_optimized_labels(self, features: np.ndarray, use_scaler: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model and return the optimized labels that the model was trained on.
        
        Args:
            features: Input features (n_samples, n_features)
            use_scaler: Whether to apply the fitted scaler to features
            
        Returns:
            Tuple of (predictions, optimized_labels)
        """
        if self.model is None:
            raise ValueError("No model loaded. Load or train a model first.")
        
        # Get predictions
        predictions = self.predict(features, use_scaler)
        
        # Get optimized labels from metadata if available
        optimized_labels = None
        if hasattr(self, '_metadata') and self._metadata and 'optimized_labels' in self._metadata:
            optimized_labels = np.array(self._metadata['optimized_labels'])
        
        return predictions, optimized_labels
    
    def calculate_k0_from_sigmoid_params(self, max_L: float, t0: float, y0: float = 0.0, t_max: float = 200.0) -> float:
        """
        Calculate k0 (rate constant) from sigmoid parameters.
        
        Args:
            max_L: Maximum disintegration level (predicted_property1)
            t0: Time at 50% disintegration (predicted_property2)
            y0: Initial disintegration level (default 0)
            t_max: Time at which max_L should be reached (default 200 days)
            
        Returns:
            k0: Rate constant for the sigmoid curve
        """
        # SIGMOID FUNCTION: y = max_L / (1 + e^(-k0 * (t - t0)))
        
        # We need to solve for k0 that satisfies both boundary conditions:
        # 1. At t=0: y ≈ 0 (very small value)
        # 2. At t=t_max: y ≈ max_L (very close to max_L)
        
        # For practical purposes, let's say:
        # At t=0: y = 0.001 * max_L (0.1% of max_L)
        # At t=t_max: y = 0.999 * max_L (99.9% of max_L)
        
        # From condition 1: 0.001 * max_L = max_L / (1 + e^(k0 * t0))
        # 0.001 = 1 / (1 + e^(k0 * t0))
        # 1 + e^(k0 * t0) = 1/0.001 = 1000
        # e^(k0 * t0) = 999
        # k0 * t0 = ln(999)
        # k0 = ln(999) / t0
        
        # From condition 2: 0.999 * max_L = max_L / (1 + e^(-k0 * (t_max - t0)))
        # 0.999 = 1 / (1 + e^(-k0 * (t_max - t0)))
        # 1 + e^(-k0 * (t_max - t0)) = 1/0.999
        # e^(-k0 * (t_max - t0)) = 1/0.999 - 1
        # -k0 * (t_max - t0) = ln(1/0.999 - 1)
        # k0 = -ln(1/0.999 - 1) / (t_max - t0)
        
        # We need k0 to satisfy BOTH conditions, so we take the maximum:
        # k0 = max(ln(999)/t0, -ln(1/0.999 - 1)/(t_max - t0))
        
        if t0 <= 0 or t_max <= t0:
            return 0.1  # Default value if parameters are invalid
        
        try:
            # Calculate k0 from both boundary conditions
            k0_from_start = np.log(999) / t0
            k0_from_end = -np.log(1/0.999 - 1) / (t_max - t0)
            
            # Take the maximum to satisfy both conditions
            k0 = max(k0_from_start, k0_from_end)
            
            # Ensure k0 is positive and reasonable
            return max(0.01, min(5.0, k0))
        except (ValueError, ZeroDivisionError):
            return 0.1  # Default value if calculation fails
    
    def generate_sigmoid_curves(self, max_L_values: np.ndarray, t0_values: np.ndarray, 
                               k0_values: np.ndarray, days: int = 200, 
                               save_csv: bool = True, save_plot: bool = True,
                               curve_type: str = 'disintegration') -> pd.DataFrame:
        """
        Generate sigmoid curves for all samples and save results.
        
        Args:
            max_L_values: Array of max_L values for each sample
            t0_values: Array of t0 values for each sample
            k0_values: Array of k0 values for each sample
            days: Number of days to simulate (default 200)
            save_csv: Whether to save CSV with daily values
            save_plot: Whether to save PNG plot
            curve_type: Type of curve ('disintegration' or 'biodegradation')
            
        Returns:
            DataFrame with daily values for all samples
        """
        # Generate time points (1 day intervals)
        time_points = np.arange(0, days + 1, 1)
        
        # Calculate sigmoid curves for all samples
        sigmoid_data = []
        
        for sample_idx, (max_L, t0, k0) in enumerate(zip(max_L_values, t0_values, k0_values)):
            # PROPER SIGMOID FUNCTION: y = max_L / (1 + e^(-k0 * (t - t0)))
            # This gives true S-shape with inflection point at t0
            # k0 is calculated to naturally satisfy boundary conditions
            
            values = max_L / (1 + np.exp(-k0 * (time_points - t0)))
            
            for day, y in zip(time_points, values):
                sigmoid_data.append({
                    'sample_index': sample_idx,
                    'day': day,
                    curve_type: y,
                    'max_L': max_L,
                    't0': t0,
                    'k0': k0
                })
        
        sigmoid_df = pd.DataFrame(sigmoid_data)
        
        # Save CSV
        if save_csv:
            csv_filename = f'sigmoid_{curve_type}_curves.csv'
            sigmoid_df.to_csv(csv_filename, index=False)
            print(f"{curve_type.capitalize()} curves saved to: {csv_filename}")
        
        # Create and save plot
        if save_plot:
            plt.figure(figsize=(12, 8))
            
            # Plot curves for each sample
            for sample_idx in range(len(max_L_values)):
                sample_data = sigmoid_df[sigmoid_df['sample_index'] == sample_idx]
                plt.plot(sample_data['day'], sample_data[curve_type], 
                        alpha=0.7, linewidth=1, label=f'Sample {sample_idx}' if sample_idx < 10 else None)
            
            plt.xlabel('Days')
            plt.ylabel(f'{curve_type.capitalize()} Level')
            plt.title(f'Polymer {curve_type.capitalize()} Sigmoid Curves')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            plot_filename = f'sigmoid_{curve_type}_curves.png'
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{curve_type.capitalize()} curves plot saved to: {plot_filename}")
        
        return sigmoid_df

def main():
    """Main function to run Differentiable Label Optimization."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train DLO model with specified data and save directory')
    parser.add_argument('--data_file', type=str, default='data/training.csv', 
                       help='Path to training data file (default: data/training.csv)')
    parser.add_argument('--save_dir', type=str, default='models/v1/', 
                       help='Directory to save model (default: models/v1/)')
    parser.add_argument('--model_name', type=str, default='dlo_model', 
                       help='Model name (default: dlo_model)')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Data file: {args.data_file}")
    print(f"Save directory: {args.save_dir}")
    
    # Initialize optimizer
    dlo = DifferentiableLabelOptimizer(device=device)
    
    # Prepare data
    features, labels, soft_indices = dlo.prepare_data(
        data_file=args.data_file,
        use_custom_labels=True  # Use custom label assignments from CSV
    )
    
    # Train model
    history = dlo.train(
        features=features,
        labels=labels,
        soft_indices=soft_indices,
        epochs=args.epochs,
        batch_size=16,
        patience=20,
        hard_label_weight=1.0,
        soft_label_weight=0.5
    )
    
    # Get label changes
    label_changes = dlo.get_soft_label_changes(labels)
    
    # Evaluate final performance
    final_metrics, predictions = dlo.evaluate(features, labels)
    
    print("\n" + "="*60)
    print("DIFFERENTIABLE LABEL OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Final Performance:")
    print(f"Property1 - R²: {final_metrics['property1_r2']:.4f}, MAE: {final_metrics['property1_mae']:.4f}")
    print(f"Property2 - R²: {final_metrics['property2_r2']:.4f}, MAE: {final_metrics['property2_mae']:.4f}")
    print(f"Overall - R²: {final_metrics['overall_r2']:.4f}, MAE: {final_metrics['overall_mae']:.4f}")
    
    print(f"\nSoft Label Statistics:")
    print(f"Number of soft labels: {len(soft_indices)}")
    if len(label_changes['changes']) > 0:
        print(f"Average label change: {np.mean(np.abs(label_changes['changes'])):.4f}")
        print(f"Max label change: {np.max(np.abs(label_changes['changes'])):.4f}")
    
    # Create plots
    dlo.plot_results(features, labels, history, label_changes)
    
    # Save results
    results_df = dlo.save_results(features, labels, label_changes)
    
    # Save the trained model
    print("\n" + "="*60)
    print("SAVING TRAINED MODEL")
    print("="*60)
    model_paths = dlo.save_model(save_dir=args.save_dir, model_name=args.model_name)
    
    # Generate sigmoid curves from optimized labels
    print("\n" + "="*60)
    print("GENERATING SIGMOID CURVES")
    print("="*60)
    
    # Get optimized labels
    optimized_labels = dlo.dataset.soft_labels.detach().cpu().numpy()
    max_L_values = optimized_labels[:, 0]  # property1 = max_L
    t0_values = optimized_labels[:, 1]     # property2 = t0
    
    # Calculate k0 for disintegration curves (200 days)
    k0_values_disintegration = np.array([dlo.calculate_k0_from_sigmoid_params(max_L, t0, t_max=200.0) 
                                        for max_L, t0 in zip(max_L_values, t0_values)])
    
    print(f"Disintegration k0 values - Min: {k0_values_disintegration.min():.4f}, Max: {k0_values_disintegration.max():.4f}, Mean: {k0_values_disintegration.mean():.4f}")
    
    # Generate and save disintegration curves
    disintegration_df = dlo.generate_sigmoid_curves(max_L_values, t0_values, k0_values_disintegration, 
                                                   days=200, curve_type='disintegration')
    
    # Calculate biodegradation parameters (t0 doubled, 400 days)
    t0_values_biodegradation = t0_values * 2.0  # Double the t0 values
    k0_values_biodegradation = np.array([dlo.calculate_k0_from_sigmoid_params(max_L, t0_bio, t_max=400.0) 
                                        for max_L, t0_bio in zip(max_L_values, t0_values_biodegradation)])
    
    print(f"Biodegradation k0 values - Min: {k0_values_biodegradation.min():.4f}, Max: {k0_values_biodegradation.max():.4f}, Mean: {k0_values_biodegradation.mean():.4f}")
    
    # Generate and save biodegradation curves
    biodegradation_df = dlo.generate_sigmoid_curves(max_L_values, t0_values_biodegradation, k0_values_biodegradation, 
                                                   days=400, curve_type='biodegradation')
    
    return dlo, results_df, final_metrics

if __name__ == "__main__":
    dlo, results, metrics = main() 