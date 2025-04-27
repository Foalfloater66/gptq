from .abstractquant import Quantizer
import torch


# TODO: fix this later.
def quantize(x, scale, zero, maxq):
    if maxq < 0: # only if trits(?) is enabled. Not relevant.
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq) # affine quantization scheme.
    return scale * (q - zero)  # affine quantization scheme: https://huggingface.co/docs/optimum/en/concept_guides/quantization 

class QuantileQuantizer(Quantizer):
    """Quantile-based quantizer.
    
    Notes on usage: Only for perchannel weight-only quantization.
    """

    def __init__(self, shape=1):
        super(QuantileQuantizer, self).__init__()
        # TODO: register the quantiles.
        self.register_buffer('scale', torch.zeros(shape))

    def configure(self, bits):
        # TODO: TBA
        # self.perchannel = True
        self.num_levels = 2**bits
        pass

    def find_params(self, x):
        # NOTE: must fix later.

        dev = x.device
        shape = x.shape

        percentiles = [100 * i / (num_levels - 1)]

        # if self.perchannel and weight:
        x = x.flatten(1)
        
        # TODO: may need to remove the below.
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

#  self.zero = torch.round(-xmin / self.scale)
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)

        shape = [-1] + [1] * (len(shape) - 1)
        # self.scale = self.scale.unsqueeze(1)
        # self.zero = self.zero.unsqueeze(1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return
    
    def quantize(self, x):
        if self.ready():
            # NOTE: why does it need to be done externally? This makes no sense.
            return quantize(x, self.scale, self.zero, self.maxq)
        return x
    
    def enabled(self):
        pass

    def ready(self):
        pass
        