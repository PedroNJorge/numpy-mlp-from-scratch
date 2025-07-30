import numpy as np
from .base import Activation


class Sigmoid(Activation):
    def forward(self, Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))
