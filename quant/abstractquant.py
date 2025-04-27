import torch.nn as nn
from abc import ABC, abstractmethod

class Quantizer(ABC, nn.Module):
    """Quantizer class template interface."""

    def __init__(self, shape=1):
        pass

    @abstractmethod
    def configure(self):
        """Sets up the quantizer with general parameters."""
        pass 

    @abstractmethod
    def find_params(self, x, weight=False):
        """Given a dataset `x`, find the best suited quantization parameters."""
        pass

    @abstractmethod
    def quantize(self, x):
        """Quantize the matrix `x`."""
        pass

    @abstractmethod
    def enabled(self):
        # NOTE: I don't know what this is.
        pass

    @abstractmethod
    def ready(self):
        """Returns true if the quantizer is ready to process parameters."""
        pass