import numpy as np
from .base import Layer


class LinearLayer(Layer):
    """Dense layer implementing XW + b."""

    def __init__(self, input_dim: int, output_dim: int):
        """Initialize weights with small random values."""
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None
        self.x = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass: Z = X @ W.T + b."""
        self.x = X
        return X @ self.W.T + self.b

    def backward(self, dL_dz: np.ndarray) -> np.ndarray:
        """
        Backward pass:
        - Computes and stores partial derivatives dL/dW and dL/db.
        - Returns dL/da^(k-1).
        """
        self.dW = dL_dz.T @ self.x
        self.db = dL_dz
        return dL_dz @ self.W

    def update(self, lr: float):
        """Updates weights (W, b) using learning rate (lr)."""
        self.W -= lr * self.dW
        self.b -= lr * self.db
