from .abstractquant import QuantizerInterface
import torch


class QuantileQuantizer(QuantizerInterface):
    """Quantile-based quantizer.

    Only supports perchannel weight-only quantization.    
    Notes on usage: Only for perchannel weight-only quantization.
    """

    def __init__(self, shape=1):
        super(QuantileQuantizer, self).__init__()
        self.register_buffer("quantization_lvls", torch.zeros(shape))

    def configure(self, bits, **kwargs):
        self.num_levels = 2**bits

    def find_params(self, x, **kwargs):
        dev = x.device
        self.quantization_lvls = self.quantization_lvls.to(dev)

        quantiles = torch.linspace(0, 1, steps=self.num_levels, device=x.device)
        self.quantization_lvls = torch.quantile(x.flatten(), quantiles) 
        return
    
    def quantize(self, x):
        if self.ready():
            quantization_lvls = self.quantization_lvls
            x_flat = x.view(-1)
            diffs = (x_flat.unsqueeze(1) - quantization_lvls.unsqueeze(0)).abs() # absolute difference between the original weights and the quantiles.
            return quantization_lvls[diffs.argmin(dim=1)].view_as(x)
        return x
    

    def ready(self):
        return torch.all(self.quantization_lvls != 0) 
        