from .abstractquant import QuantizerInterface
import torch

class LloydMaxQuantizer(QuantizerInterface):
    """Lloyd-Max quantizer implementation.
    
    This implements the iterative Lloyd-Max algorithm that optimizes quantization
    levels to minimize MSE for a given data distribution.
    """

    def __init__(self, shape=1):
        super(LloydMaxQuantizer, self).__init__()
        # Register buffers to store quantization parameters
        self.register_buffer("quantization_lvls", torch.zeros(shape))
        self.register_buffer("decision_boundaries", torch.zeros(shape))
        self.tolerance = 1e-6    # Convergence tolerance
        
    def configure(self, bits, **kwargs):
        """Configure the quantizer with bit-width and other parameters."""
        self.num_levels = 2**bits
        self.max_iterations = kwargs.get("max_iterations", 1)

    def _estimate_boundaries(self, x: torch.Tensor):
        return (x[:, 1:] + x[:, :-1]) / 2
    
    def _pad_boundaries(self, x: torch.Tensor):
        """Pad boundaries with minimum and maximimum (-inf and +inf) to count all values."""
        n = x.shape[0]
        return torch.cat([torch.full((n, 1), float('-inf'), device=x.device), x[:, :], torch.full((n, 1), float('inf'), device=x.device)], dim=1)

    def _compute_mse(self, x, decision_boundaries):
        """Calculate the MSE between the quantized values and the new values.
        Assumes `x` to be a 2D matrix."""

        masks = (x.unsqueeze(2) >= decision_boundaries[:, :-1].unsqueeze(1)) & \
                    (x.unsqueeze(2) < decision_boundaries[:, 1:].unsqueeze(1))
        res = (masks * self.quantization_lvls.unsqueeze(1)).sum(dim=-1)
        mse = torch.mean(torch.square(x - res))
        return mse
    
    def find_params(self, x, **kwargs):
        """Find optimal quantization levels using Lloyd-Max algorithm."""
        dev = x.device

        # Find initial decision boundaries:
        x_min = x.min(dim=1)[0]
        x_max = x.max(dim=1)[0]
        # Ensure we don't have degenerate cases
        identical_rows = (x_min == x_max)
        if identical_rows.any():
            x_min += identical_rows * (x_min- 1e-6)
            x_max += identical_rows * (x_max + 1e-6)
        x_max = x_max.unsqueeze(1)
        x_min = x_min.unsqueeze(1)
        self.quantization_lvls = torch.linspace(0, 1, self.num_levels, device=dev).unsqueeze(0) * (x_max - x_min) + x_min # Create uniformly spaced representation points for each row.

        if self.max_iterations < 1:
            self.decision_boundaries = self._estimate_boundaries(self.quantization_lvls)

        for i in range(self.max_iterations):
            self.decision_boundaries = self._estimate_boundaries(self.quantization_lvls)
            
            tmp_decision_boundaries = self._pad_boundaries(self.decision_boundaries)
            
            masks = (x.unsqueeze(2) >= tmp_decision_boundaries[:, :-1].unsqueeze(1)) & \
                    (x.unsqueeze(2) < tmp_decision_boundaries[:, 1:].unsqueeze(1))
            
            # Check that there exists no entry with more or less than 1 True.
            if not torch.all(masks.sum(dim=2) == 1):
                raise ValueError("In the matrix 'masks', there exists a vector in the last dimension with more or less than one element set to True. Aborting")

            # Compute the mean explicitly to prevent excluding null weights.
            masked_sums = (masks * x.unsqueeze(2)).sum(dim=1) 
            counts = masks.sum(dim=1)

            self.quantization_lvls = torch.where(counts > 0, masked_sums / counts, self.quantization_lvls)

        self.decision_boundaries = self._pad_boundaries(self.decision_boundaries)
        return
    
    def quantize(self, x):
        """Quantize input tensor using learned quantization levels. ONLY quantize a list."""
        if self.ready():
            masks = (x >= self.decision_boundaries[:, :-1]) & \
                    (x < self.decision_boundaries[:, 1:])
            res = (masks * self.quantization_lvls).sum(dim=-1)
            return res
        return x
    
    def ready(self):
        """Check if quantizer is ready (parameters have been learned)."""
        return torch.all(self.quantization_lvls != 0)