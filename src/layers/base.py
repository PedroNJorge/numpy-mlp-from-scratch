import numpy as np
from abc import ABC, abstractmethod


class Layer(ABC):
    """Abstract base class for all layers."""

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
