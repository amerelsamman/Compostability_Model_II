#!/usr/bin/env python3
"""
DifferentiableLabelOptimizer module for Differentiable Label Optimization.
Contains the main optimizer class that coordinates training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import warnings
import os
import pickle
import json
warnings.filterwarnings('ignore')

from .dataset import PolymerDataset
from .model import PolymerPredictor
from .utils import calculate_k0_from_sigmoid_params, generate_sigmoid_curves


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
                    use_custom_labels: bool = True, 
                    oversample_high_maxl: bool = True,
                    oversample_low_maxl: bool = True,
                    high_oversample_factor: int = 5,
                    low_oversample_factor: int = 5,
                    high_maxl_threshold: float = 90.0,
                    low_maxl_threshold: float = 5.0) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Prepare data and identify which labels should be soft (trainable).
        
        Args:
            data_file: Path to the input CSV file
            use_custom_labels: Whether to use custom label assignments from CSV
            oversample_high_maxl: Whether to oversample samples with high max_L values
            oversample_factor: How many times to duplicate high max_L samples
            maxl_threshold: Threshold above which to consider max_L as "high"
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
        
        # Exclude categorical variables and physical properties entirely
        exclude_cols = ['enzyme kinetics', 'Polymer Grade 1', 'Polymer Grade 2', 
                       'Polymer Grade 3', 'Polymer Grade 4', 'Polymer Grade 5',
                       'wa', 'wvtr', 'ts', 'Tg', 'eab', 'Xc']
        
        for col in exclude_cols:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Convert to numeric, excluding label columns and SMILES
        exclude_from_numeric = label_cols + ['Materials', 'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5']
        
        numeric_cols = [col for col in df.columns if col not in exclude_from_numeric]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Restore label columns
        for col in label_cols:
            if col in label_data:
                df[col] = label_data[col]
        
        # Note: Categorical features are excluded, so no one-hot encoding needed
        
        # Extract features (exclude metadata and target columns, include volume fractions and molecular features only)
        exclude_cols = ['Materials', 'SMILES1', 'SMILES2', 'SMILES3', 'SMILES4', 'SMILES5',
                       'property1', 'property2', 'property1_label', 'property2_label',
                       'Thickness_certification']
        
        # Include volume fractions in features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        features = df[feature_cols].values
        labels = df[['property1', 'property2']].values
        
        print(f"Feature columns included: {feature_cols}")
        print(f"Number of features: {len(feature_cols)}")
        
        # Oversample high and low max_L samples if requested
        if oversample_high_maxl or oversample_low_maxl:
            print(f"Oversampling samples with max_L > {high_maxl_threshold} and max_L < {low_maxl_threshold}")
            
            # Find high and low max_L samples
            high_maxl_mask = labels[:, 0] > high_maxl_threshold  # property1 is max_L
            low_maxl_mask = labels[:, 0] < low_maxl_threshold   # property1 is max_L
            
            high_maxl_indices = np.where(high_maxl_mask)[0]
            low_maxl_indices = np.where(low_maxl_mask)[0]
            
            print(f"Found {len(high_maxl_indices)} samples with max_L > {high_maxl_threshold}")
            print(f"Found {len(low_maxl_indices)} samples with max_L < {low_maxl_threshold}")
            
            if len(high_maxl_indices) > 0 or len(low_maxl_indices) > 0:
                # Create oversampled data
                oversampled_features = []
                oversampled_labels = []
                
                # Add original data
                oversampled_features.append(features)
                oversampled_labels.append(labels)
                
                # Add oversampled high max_L data
                if oversample_high_maxl and len(high_maxl_indices) > 0:
                    for _ in range(high_oversample_factor - 1):  # -1 because we already have original
                        oversampled_features.append(features[high_maxl_indices])
                        oversampled_labels.append(labels[high_maxl_indices])
                
                # Add oversampled low max_L data
                if oversample_low_maxl and len(low_maxl_indices) > 0:
                    for _ in range(low_oversample_factor - 1):  # -1 because we already have original
                        oversampled_features.append(features[low_maxl_indices])
                        oversampled_labels.append(labels[low_maxl_indices])
                
                # Concatenate all data
                features = np.vstack(oversampled_features)
                labels = np.vstack(oversampled_labels)
                
                print(f"After oversampling: {features.shape[0]} samples (original: {df.shape[0]})")
                if len(high_maxl_indices) > 0:
                    print(f"High max_L samples now represent {len(high_maxl_indices) * high_oversample_factor / features.shape[0]:.1%} of the dataset")
                if len(low_maxl_indices) > 0:
                    print(f"Low max_L samples now represent {len(low_maxl_indices) * low_oversample_factor / features.shape[0]:.1%} of the dataset")
        
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
                   soft_label_weight: float = 0.5,
                   use_weighted_loss: bool = True,
                   high_maxl_weight: float = 3.0,
                   low_maxl_weight: float = 3.0,
                   high_maxl_threshold: float = 90.0,
                   low_maxl_threshold: float = 5.0) -> Dict[str, float]:
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
            if use_weighted_loss:
                batch_loss = self.custom_weighted_loss(predictions, batch_labels, 
                                                     high_maxl_weight, low_maxl_weight,
                                                     high_maxl_threshold, low_maxl_threshold)
            else:
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
              soft_label_weight: float = 0.5,
              use_weighted_loss: bool = True,
              high_maxl_weight: float = 3.0,
              low_maxl_weight: float = 3.0,
              high_maxl_threshold: float = 90.0,
              low_maxl_threshold: float = 5.0) -> Dict[str, List[float]]:
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
            train_metrics = self.train_epoch(dataloader, hard_label_weight, soft_label_weight,
                                           use_weighted_loss, high_maxl_weight, low_maxl_weight,
                                           high_maxl_threshold, low_maxl_threshold)
            
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
                    label_changes: Dict[str, np.ndarray],
                    save_dir: str = '.'):
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
        plt.savefig(os.path.join(save_dir, 'differentiable_label_optimization_results.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self, features: np.ndarray, labels: np.ndarray, 
                    label_changes: Dict[str, np.ndarray], 
                    filename: str = 'dlo_predictions.csv',
                    save_dir: str = '.'):
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
        
        filepath = os.path.join(save_dir, filename)
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
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
    
    def calculate_k0_from_sigmoid_params(self, max_L: float, t0: float, y0: float = 0.0, t_max: float = 200.0, 
                                       majority_polymer_high_disintegration: bool = None) -> float:
        """Wrapper for the utility function."""
        return calculate_k0_from_sigmoid_params(max_L, t0, y0, t_max, majority_polymer_high_disintegration)
    
    def generate_sigmoid_curves(self, max_L_values: np.ndarray, t0_values: np.ndarray, 
                               k0_values: np.ndarray, days: int = 200, 
                               save_csv: bool = True, save_plot: bool = True,
                               curve_type: str = 'disintegration', save_dir: str = '.') -> pd.DataFrame:
        """Wrapper for the utility function."""
        return generate_sigmoid_curves(max_L_values, t0_values, k0_values, days, save_csv, save_plot, curve_type, save_dir) 

    def custom_weighted_loss(self, predictions: torch.Tensor, targets: torch.Tensor, 
                           high_maxl_weight: float = 3.0, low_maxl_weight: float = 3.0,
                           high_maxl_threshold: float = 90.0, low_maxl_threshold: float = 5.0) -> torch.Tensor:
        """
        Custom loss function that gives higher weight to both high and low max_L samples.
        
        Args:
            predictions: Model predictions (batch_size, 2)
            targets: Target values (batch_size, 2)
            high_maxl_weight: Weight multiplier for high max_L samples
            low_maxl_weight: Weight multiplier for low max_L samples
            high_maxl_threshold: Threshold above which to consider max_L as "high"
            low_maxl_threshold: Threshold below which to consider max_L as "low"
            
        Returns:
            Weighted MSE loss
        """
        # Calculate standard MSE loss
        mse_loss = nn.MSELoss(reduction='none')(predictions, targets)
        
        # Create weights tensor (batch_size, 2)
        weights = torch.ones_like(mse_loss)
        
        # Give higher weight to samples with high max_L (property1)
        high_maxl_mask = targets[:, 0] > high_maxl_threshold
        weights[high_maxl_mask, :] = high_maxl_weight
        
        # Give higher weight to samples with low max_L (property1)
        low_maxl_mask = targets[:, 0] < low_maxl_threshold
        weights[low_maxl_mask, :] = low_maxl_weight
        
        # Apply weights and take mean
        weighted_loss = (mse_loss * weights).mean()
        
        return weighted_loss 