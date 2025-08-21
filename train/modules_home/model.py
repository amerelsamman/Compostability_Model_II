#!/usr/bin/env python3
"""
PolymerPredictor module for Differentiable Label Optimization.
Contains the neural network model for predicting polymer properties.
"""

import torch
import torch.nn as nn
from typing import List


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