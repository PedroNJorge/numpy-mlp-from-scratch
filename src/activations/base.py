import numpy as np
from abc import ABC, abstractmethod


class Activation(ABC):
    """Abstract base class for all activation layers."""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
