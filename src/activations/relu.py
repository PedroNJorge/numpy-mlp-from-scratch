import numpy as np
from .base import Activation


class ReLU(Activation):
    """ReLU activation layer."""

    def __init__(self):
        self.z = None

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.z = Z
        return np.maximum(0, Z)

    def backward(self, dL_da: np.ndarray) -> np.ndarray:
        """Receives dL/da^(L) and returns delta.
        The derivative of ReLU(z) is 1 if z > 0
                                     0 if z <= 0
        """
        return dL_da * (self.z > 0)
