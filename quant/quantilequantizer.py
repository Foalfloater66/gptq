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
        quantiles = torch.linspace(0, 1, steps=self.num_levels, device=dev)
        self.quantization_lvls = torch.quantile(x.flatten(1), quantiles, dim=1)
        self.quantization_lvls = self.quantization_lvls.T
        return
    
    def quantize(self, x):
        if self.ready():
            quantization_lvls = self.quantization_lvls
            x_flat = x.flatten(1).unsqueeze(2) 
            qlvls_shape = quantization_lvls.shape
            quantization_lvls = quantization_lvls.reshape(qlvls_shape[0], 1, qlvls_shape[1])
            diffs = (quantization_lvls - x_flat).abs() 
            res = torch.gather(self.quantization_lvls, 1, diffs.argmin(dim=-1))
            return res
        return x
    

    def ready(self):
        return torch.all(self.quantization_lvls != 0) 
        