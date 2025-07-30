import numpy as np
from .base import Activation


class ReLU(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
