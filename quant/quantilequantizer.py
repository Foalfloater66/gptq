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
        # self.quantization_lvls = self.quantization_lvls.to(dev)

        # TODO: I think there is an error here. Only the first dimension should be flattened, not everything.
        quantiles = torch.linspace(0, 1, steps=self.num_levels, device=dev)
        # self.quantization_lvls
        self.quantization_lvls = torch.quantile(x.flatten(1), quantiles, dim=1)
        q_shape = self.quantization_lvls.shape
        self.quantization_lvls = self.quantization_lvls.T
        # reshape(q_shape[1], q_shape[0]) 
        return
    
    def quantize(self, x):
        if self.ready():
            quantization_lvls = self.quantization_lvls
            # print(f"x shape: {x.shape}")
            x_flat = x.flatten(1).unsqueeze(2)  # Flatten only the first dimension and add a dimension for broadcasting
            # print(f"quantization levels raw: {quantization_lvls.shape}")
            # quantization_lvls = quantization_lvls.unsqueeze(1)  # Add a dimension for broadcasting
            qlvls_shape = quantization_lvls.shape
            quantization_lvls = quantization_lvls.reshape(qlvls_shape[0], 1, qlvls_shape[1])
            # print(f"x flat shape: {x_flat.shape}")
            # print(f"quantization levels shape: {quantization_lvls.shape}")
            # print(quantization_lvls.shape, x_flat.shape)
            diffs = (quantization_lvls - x_flat).abs()  # Compute absolute difference row-wise
            # print(f"diffs argmin shaep: {diffs.argmin(dim=-1).shape}")
            # print(x.shape)
            res = torch.gather(self.quantization_lvls, 1, diffs.argmin(dim=-1))
            # print(res)
            # exit(0)
            return res
            return quantization_lvls[diffs.argmin(dim=-1)]
        return x
    

    def ready(self):
        return torch.all(self.quantization_lvls != 0) 
        