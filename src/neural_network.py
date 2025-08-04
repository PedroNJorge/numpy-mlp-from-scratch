import numpy as np
import tqdm


class NeuralNetwork:
    def __init__(self, layers: list[object], loss):
        self.layers = layers
        self.loss = loss

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Sequential forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, delta: np.ndarray):
        """Sequential backward pass (updates gradients)."""
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def update(self, lr: float):
        """Updates weights after backward()."""
        for layer in self.layers:
            layer.update(lr)

    def train(self, X: np.ndarray, y: np.ndarray, lr: float, epochs: int):
        """Full training loop with loss integration."""
        pbar = tqdm(range(1, epochs + 1), desc="Training")
        for epoch in pbar:
            # Forward pass
            y_hat = self.forward(X)

            # Compute loss and initial gradient
            L = self.loss.forward(y_hat, y)  # Stores (a_L, y) in cache
            delta = self.loss.backward()     # dL/da_L

            # Backward pass
            self.backward(delta)

            # Stochastic Gradient Descent
            self.update(lr)

            pbar.set_postfix({"Loss": f"{L:.4f}"})
