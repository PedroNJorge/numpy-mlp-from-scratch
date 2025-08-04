import numpy as np
from typing import Dict


class MLP:
    def __init__(self, hidden_layers_node_count: Dict):
        self.num_hidden_layers = len(hidden_layers_node_count)
