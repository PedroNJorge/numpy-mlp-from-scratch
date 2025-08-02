import numpy as np


class MSE:
    def __init__(self):
        self.cache = {}  # Stores (a_L, y) for backward pass

    def forward(self, a_L: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss: L = 1/(2n) Σ(a_L - y)²."""
        self.cache["a_L"], self.cache["y"] = a_L, y
        return np.mean((a_L - y) ** 2) / 2

    def backward(self) -> np.ndarray:
        """Compute dL/da_L = (1/n) * (a_L - y)."""
        a_L, y = self.cache["a_L"], self.cache["y"]
        return (a_L - y) / len(a_L)
