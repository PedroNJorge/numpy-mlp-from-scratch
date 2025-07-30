import numpy as np
from .base import Layer


class LinearLayer(Layer):
    """Dense layer implementing XW + b."""

    def __init__(self, input_dim: int, output_dim: int):
        self.weights = np.random.randn(input_dim, output_dim)
        self.bias = np.zeros(output_dim, 1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return (X.T @ self.weights).T + self.bias
