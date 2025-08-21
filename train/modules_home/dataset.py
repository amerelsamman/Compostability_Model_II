#!/usr/bin/env python3
"""
PolymerDataset module for Differentiable Label Optimization.
Contains the custom dataset class with hard/soft label support.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from typing import List


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