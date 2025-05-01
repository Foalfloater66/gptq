from .abstractquant import QuantizerInterface
import torch
import numpy as np

class LloydMaxQuantizer(QuantizerInterface):
    """Lloyd-Max quantizer implementation.
    
    This implements the iterative Lloyd-Max algorithm that optimizes quantization
    levels to minimize MSE for a given data distribution.
    """

    def __init__(self, shape=1):
        super(LloydMaxQuantizer, self).__init__()
        # Register buffers to store quantization parameters
        # self.register_buffer("quant_levels", torch.zeros(shape))
        self.register_buffer("quantization_lvls", torch.zeros(shape))
        self.register_buffer("decision_boundaries", torch.zeros(shape))
        self.max_iterations = 0  # Max iterations for Lloyd-Max algorithm
        self.tolerance = 1e-6     # Convergence tolerance
        
    def configure(self, bits, **kwargs):
        """Configure the quantizer with bit-width and other parameters."""
        self.num_levels = 2**bits
    
    def find_params(self, x, **kwargs):
        """Find optimal quantization levels using Lloyd-Max algorithm."""
        dev = x.device

        # Find initial decision boundaries:
        quantiles = torch.linspace(0, 1, steps=self.num_levels, device=dev)
        self.quantization_lvls = torch.quantile(x.flatten(1), quantiles, dim=1).T

        # temporary number of iterations:
        # max_iterations = 4

        if self.max_iterations < 1:
            self.decision_boundaries = (self.quantization_lvls[:, 1:] + self.quantization_lvls[:, :-1]) / 2

        # set number of iterations
        for _ in range(self.max_iterations):
            self.decision_boundaries = (self.quantization_lvls[:, 1:] + self.quantization_lvls[:, :-1]) / 2

            tmp_decision_boundaries = torch.cat([self.quantization_lvls[:, 0].unsqueeze(-1), self.decision_boundaries[:, :], self.quantization_lvls[:, -1].unsqueeze(-1)], dim=1)
            masks = (x.unsqueeze(2) >= tmp_decision_boundaries[:, :-1].unsqueeze(1)) & \
                    (x.unsqueeze(2) < tmp_decision_boundaries[:, 1:].unsqueeze(1))
            # Handle maximum values
            last_lvl_masks = (x == tmp_decision_boundaries[:, -1].unsqueeze(1))
            masks[:, :, -1] |= last_lvl_masks

            # Compute the mean explicitly to prevent excluding null weights.
            masked_sums = (masks * x.unsqueeze(2)).sum(dim=1) 
            counts = masks.sum(dim=1)
            self.quantization_lvls = torch.where(counts > 0, masked_sums / counts, self.quantization_lvls)

            """
            TODO:
            If maximum difference between any of the quantization levels is upper bounded by 1e(-6), stop early.
            """

        # include min and max in final decision boundaries
        self.decision_boundaries = torch.cat([self.quantization_lvls[:, 0].unsqueeze(-1), self.decision_boundaries[:, :], self.quantization_lvls[:, -1].unsqueeze(-1)], dim=1)
        return
    
    def quantize(self, x):
        """Quantize input tensor using learned quantization levels."""


        if self.ready():
            masks = (x.unsqueeze(2) >= self.decision_boundaries[:, :-1].unsqueeze(1)) & \
                    (x.unsqueeze(2) < self.decision_boundaries[:, 1:].unsqueeze(1))
            # Handle maximum values
            last_lvl_masks = (x == self.decision_boundaries[:, -1].unsqueeze(1))
            masks[:, :, -1] |= last_lvl_masks
            # Assuming only one `True` per slice (if not, there's a bug)
            res = (masks.flatten(1) * self.quantization_lvls).sum(dim=-1)
            return res
        return x
    
    def ready(self):
        """Check if quantizer is ready (parameters have been learned)."""
        return torch.all(self.quantization_lvls != 0)