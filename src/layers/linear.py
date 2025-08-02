import numpy as np
from .base import Layer


class LinearLayer(Layer):
    """Dense layer implementing XW + b."""

    def __init__(self, input_dim: int, output_dim: int):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.x = X
        return X @ self.W.T + self.b

    def backward(self, dL_dz: np.ndarray) -> np.ndarray:
        """Receives dL/dz = delta and returns dL/da^(L-1)."""
        self.dW = dL_dz.T @ self.x
        self.db = dL_dz
        return dL_dz @ self.w
