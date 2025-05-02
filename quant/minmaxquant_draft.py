import numpy as np
import torch
import torch.nn as nn
# DO NOT USE. THIS IS TO UNDERSTAND THE MINMAX CODE WITHOUT THE UNUSED OPTIONS.

def quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq) # affine quantization scheme.
    return scale * (q - zero)  # affine quantization scheme: https://huggingface.co/docs/optimum/en/concept_guides/quantization 

# TODO (@Morgane): separate into different classes for different types of quantization.
class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        # register_buffer: tensor which isn't a parameter, but should be kept in the model state.
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        """Sets up the number of bits, the norm, the grid size, the maxshrink, and the `perchannel`, `sym`, `mse`, and `trits` booleans."""
        self.maxq = torch.tensor(2 ** bits - 1)  # maxmium possible value in a bit.
        self.perchannel = perchannel  # if enabled, for each dimension, the values in the tensor are quantized with different quantization parameters (less errors).

    def find_params(self, x, weight=False):
        """Finds the quantization parameters for a vector/matrix.
        In practice, performed on either the entire weight matrix or a subgroup.
        
            x (torch.Tensor): Input tensor.
            weight (bool): Whether `x` is a weight tensor or not.
        """
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape  # shape of the vector.
        # In all of the below examples, the tensor is made 2-dimensional.

            # if "per channel" is enabled, and "weight" is NOT enabled, then the tranpose is retrieved.
        x = x.flatten(1)  # ensures the tensor is 2-dimensional while keeping the batch dimension.

        # STEP 2: Get the minimum and maximum values across each vector in the second dimension.
        tmp = torch.zeros(x.shape[0], device=dev)  # set of 0s to clamp the values 
        xmin = torch.minimum(x.min(1)[0], tmp)  # get the minimum values across the 2nd dimension. (for each batch.)
        xmax = torch.maximum(x.max(1)[0], tmp)  # get the maximum value across the 2nd dimension. (for each batch.)

        # where min and max are both zero, simply set them to -1 and +1.
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

          # scale is the difference divided by the maximum possible size.
        self.scale = (xmax - xmin) / self.maxq
        self.zero = torch.round(-xmin / self.scale)


        shape = [-1] + [1] * (len(shape) - 1) # shape: [-1, <1 for every dimension after the first dimension.]
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return
        
    def quantize(self, x):
        """Given a vector, quantizes it using the self.scale, self.zero, and self.maxq passed as input."""
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        """Returns true if the maximum quantized value has been set."""
        return self.maxq > 0

    def ready(self): 
        """Returns true if all entries in self.scale are non-zero."""
        return torch.all(self.scale != 0)


try:
    import quant_cuda
except:
    print('CUDA extension not installed.')

# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)
