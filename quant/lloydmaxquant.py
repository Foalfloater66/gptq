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
        self.register_buffer("quant_levels", torch.zeros(shape))
        self.register_buffer("decision_boundaries", torch.zeros(shape))
        self.max_iterations = 20  # Max iterations for Lloyd-Max algorithm
        self.tolerance = 1e-6     # Convergence tolerance
        
    def configure(self, bits, **kwargs):
        """Configure the quantizer with bit-width and other parameters."""
        self.num_levels = 2**bits
        self.perchannel = kwargs.get('perchannel', False)
        
    def _lloyd_max_algorithm(self, data):
        """Core Lloyd-Max algorithm implementation."""
        # Initial quantization levels: uniform between min and max
        data_min, data_max = np.min(data), np.max(data)
        levels = np.linspace(data_min, data_max, self.num_levels)
        
        # Add small offset to prevent division by zero
        if data_min == data_max:
            return np.ones(self.num_levels) * data_min, np.ones(self.num_levels+1) * data_min
        
        # Initialize boundaries between levels
        boundaries = np.zeros(self.num_levels + 1)
        boundaries[0] = data_min - 1  # Ensures all data >= first boundary
        boundaries[-1] = data_max + 1  # Ensures all data < last boundary
        
        # Lloyd-Max iterative optimization
        for iteration in range(self.max_iterations):
            # Set decision boundaries between quantization levels
            for i in range(1, self.num_levels):
                boundaries[i] = (levels[i-1] + levels[i]) / 2
            
            # Store previous levels to check for convergence
            prev_levels = levels.copy()
            
            # Update quantization levels (centroid of data in each region)
            for i in range(self.num_levels):
                # Find data points that fall in this region
                mask = (data >= boundaries[i]) & (data < boundaries[i+1])
                if np.any(mask):
                    levels[i] = np.mean(data[mask])
            
            # Check for convergence
            if np.max(np.abs(levels - prev_levels)) < self.tolerance:
                break
        
        return levels, boundaries
    
    def find_params(self, x, **kwargs):
        """Find optimal quantization levels using Lloyd-Max algorithm."""
        dev = x.device
        # is_weight = kwargs.get('weight', False)
        self.quant_levels = self.quant_levels.to(dev)
        self.decision_boundaries = self.decision_boundaries.to(dev)
        
        # Handle per-channel quantization for weights
        # if is_weight and self.perchannel and len(x.shape) > 1:
        # Resize buffers if needed
        if self.quant_levels.shape[0] != x.shape[0]:
            self.quant_levels = torch.zeros((x.shape[0], self.num_levels), device=dev)
            self.decision_boundaries = torch.zeros((x.shape[0], self.num_levels+1), device=dev)
        
        # Apply Lloyd-Max to each channel
        for idx in range(x.shape[0]):
            channel_data = x[idx].flatten().cpu().numpy()
            levels, boundaries = self._lloyd_max_algorithm(channel_data)
            self.quant_levels[idx] = torch.tensor(levels, device=dev)
            self.decision_boundaries[idx] = torch.tensor(boundaries, device=dev)
        # else:
        #     # Global quantization
        #     data = x.flatten().cpu().numpy()
        #     levels, boundaries = self._lloyd_max_algorithm(data)
        #     self.quant_levels = torch.tensor(levels, device=dev)
        #     self.decision_boundaries = torch.tensor(boundaries, device=dev)
        return
    
    def quantize(self, x):
        """Quantize input tensor using learned quantization levels."""
        if not self.ready():
            return x
            
        if len(self.quant_levels.shape) > 1 and len(x.shape) > 1:
            # Per-channel quantization
            result = torch.zeros_like(x)
            for idx in range(x.shape[0]):
                channel_data = x[idx]
                levels = self.quant_levels[idx]
                boundaries = self.decision_boundaries[idx]
                
                # Create a quantized version for each element
                channel_flat = channel_data.flatten()
                quantized_flat = torch.zeros_like(channel_flat)
                
                # Assign each value to its quantization level
                for i in range(self.num_levels):
                    mask = (channel_flat >= boundaries[i]) & (channel_flat < boundaries[i+1])
                    quantized_flat[mask] = levels[i]
                
                result[idx] = quantized_flat.view_as(channel_data)
            return result
        else:
            # Global quantization
            levels = self.quant_levels
            boundaries = self.decision_boundaries
            
            x_flat = x.flatten()
            quantized = torch.zeros_like(x_flat)
            
            for i in range(self.num_levels):
                mask = (x_flat >= boundaries[i]) & (x_flat < boundaries[i+1])
                quantized[mask] = levels[i]
            
            return quantized.view_as(x)
    
    def ready(self):
        """Check if quantizer is ready (parameters have been learned)."""
        return torch.all(self.quant_levels != 0)