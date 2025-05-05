import numpy as np
import torch
import torch.nn as nn


def quantize(x, scale, zero, maxq):
    if maxq < 0: # only if trits(?) is enabled. Not relevant.
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
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
        self.sym = sym # not needed for us.
        self.mse = mse # not needed for us.
        self.norm = norm # only for MSE = True
        self.grid = grid  # only for MSE = True
        self.maxshrink = maxshrink  # only for MSE = True
        if trits:
            self.maxq = torch.tensor(-1) 

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
        if self.perchannel: # good for CNNs.
            # if "per channel" is enabled, and "weight" is NOT enabled, then the tranpose is retrieved.
            if weight: # whether the current tensor is a weight or not.
                x = x.flatten(1)  # ensures the tensor is 2-dimensional while keeping the batch dimension.
            else:
                # NOTE: we don't worry about the below. This is for different types of quantization
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3]) # switches the first dimension with the second dimension (gets the tranpose with respect to those dimensions only).
                    x = x.flatten(1)  # ensures the tensor is 2-dimensional.
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t() # makes the tensor into a 2-d tensor, and gets the tranpose.
                if len(shape) == 2:
                    x = x.t()  # returns transpose.
        else:
            x = x.flatten().unsqueeze(0) # makes it into a 2-d matrix and treat it as one batch.

        # STEP 2: Get the minimum and maximum values across each vector in the second dimension.
        tmp = torch.zeros(x.shape[0], device=dev)  # set of 0s to clamp the values 
        xmin = torch.minimum(x.min(1)[0], tmp)  # get the minimum values across the 2nd dimension. (for each batch.)
        xmax = torch.maximum(x.max(1)[0], tmp)  # get the maximum value across the 2nd dimension. (for each batch.)

        if self.sym:  # enable symmetric mapping.
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        # where min and max are both zero, simply set them to -1 and +1.
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0: # if the maximum quantization value is negative (NOTE to self: HOW WOULD THAT HAPPEN?)
          self.scale = xmax # the scale is the max value vector.
          self.zero = xmin # "zero" is mapped to the min value vector.
        else:
          # scale is the difference divided by the maximum possible size.
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              # if it's symmetric, zero is set to be this.
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              # if it's asymmetric, zero is set to be this.
              self.zero = torch.round(-xmin / self.scale)


        #  PER ROW QUANTIZATION  #
        # ---------------------- #
        #  Code for OBQ. Not used in the GPTQ paper experiments.
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)  
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
    
        # if not a convolutional layer
        if not self.perchannel:
            if weight: # weight layer
                tmp = shape[0]
            else: # non-weight layer (e.g. activation layer)
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1) # shape: [-1, <1 for every dimension after the first dimension.]
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        
        # if, for example, it's a convolutional layer.
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3: 
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2: 
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

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


import math # Add math import for ceil

try:
    import quant_cuda
    print("CUDA 3-bit kernel found.")
    _quant_cuda_3bit_available = True
except ImportError:
    print('CUDA 3-bit kernel not found.')
    _quant_cuda_3bit_available = False

# Import the compiled 4-bit CUDA kernel
try:
    import quant_cuda_4bit
    print("CUDA 4-bit kernel found")
    _quant_cuda_4bit_available = True
except ImportError:
    print('CUDA 4-bit kernel not found.')
    _quant_cuda_4bit_available = False


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


# Define the Quant4Linear layer
class Quant4Linear(nn.Module):
    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        # Only enable faster if available and input features are suitable
        self.faster = faster and _quant_cuda_4bit_available and (infeatures % 8 == 0)

        if faster and not _quant_cuda_4bit_available:
             print("Warning: Faster 4-bit kernel requested but CUDA module not found. Disabling.")
        if faster and _quant_cuda_4bit_available and (infeatures % 8 != 0):
             print(f"Warning: Faster 4-bit kernel requires infeatures ({infeatures}) divisible by 8. Disabling.")

        # Quantization parameters (scales and zeros)
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        # Zeros represents the dequantization offset: zero_point * scale
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        # Bias term (optional, copied from original linear layer)
        self.register_buffer('bias', torch.zeros(outfeatures))

        # Packed weights buffer
        # Each int32 stores 8 4-bit weights.
        # Weights need to be transposed before packing: (outfeatures, infeatures) -> (infeatures, outfeatures)
        # Packed shape is (ceil(infeatures / 8), outfeatures)
        packed_infeatures = math.ceil(infeatures / 8.0)
        self.register_buffer(
            'qweight',
            torch.zeros((packed_infeatures, outfeatures), dtype=torch.int32)
        )

    def pack(self, linear, scales, zeros_float):
        """
        Packs a FloatLinear layer into this Quant4Linear layer.

        Args:
            linear (nn.Linear): The original FloatLinear layer.
            scales (torch.Tensor): The quantization scales (shape: [outfeatures, 1]).
            zeros_float (torch.Tensor): The floating-point zero points (shape: [outfeatures, 1]).
                                        These are the actual zero points, not zero_point * scale.
                                        The kernel expects zero_point * scale, which is calculated here.
        """
        if not _quant_cuda_4bit_available:
            raise ImportError("Cannot pack weights, CUDA 4-bit kernel not found.")

        self.scales = scales.clone().to(linear.weight.device)
        # Store zero_point * scale for the kernel
        self.zeros = (zeros_float * self.scales).clone().to(linear.weight.device)
        if linear.bias is not None:
            self.bias = linear.bias.clone().to(linear.weight.device)

        # Quantize weight to 4-bit range [0, 15] using the provided scales and zero points
        # Q(x) = round(x / scale + zero_point)
        weight_fp = linear.weight.data # Shape: (outfeatures, infeatures)
        weight_q = torch.round(weight_fp / self.scales.t() + zeros_float.t()) # Transpose scales/zeros
        weight_q = torch.clamp(weight_q, 0, 15).to(torch.int32) # Clamp to 4-bit unsigned range

        # Transpose for packing: (outfeatures, infeatures) -> (infeatures, outfeatures)
        weight_q = weight_q.t().contiguous() # Shape: (infeatures, outfeatures)

        # Pad infeatures if not divisible by 8
        padded_infeatures = math.ceil(self.infeatures / 8.0) * 8
        if self.infeatures % 8 != 0:
            padding = padded_infeatures - self.infeatures
            weight_q = torch.cat([weight_q, torch.zeros((padding, self.outfeatures), dtype=torch.int32, device=weight_q.device)], dim=0)
            # print(f"Padding weights from {self.infeatures} to {padded_infeatures} features.")

        # Pack 8 values into one int32
        packed_weight = torch.zeros(
            (padded_infeatures // 8, self.outfeatures), dtype=torch.int32, device=weight_q.device
        )

        for i in range(padded_infeatures // 8):
            base_idx = i * 8
            # Pack bits: val0 | val1 << 4 | val2 << 8 | ...
            packed_val = (
                  (weight_q[base_idx + 0, :] & 0xF) << 0
                | (weight_q[base_idx + 1, :] & 0xF) << 4
                | (weight_q[base_idx + 2, :] & 0xF) << 8
                | (weight_q[base_idx + 3, :] & 0xF) << 12
                | (weight_q[base_idx + 4, :] & 0xF) << 16
                | (weight_q[base_idx + 5, :] & 0xF) << 20
                | (weight_q[base_idx + 6, :] & 0xF) << 24
                | (weight_q[base_idx + 7, :] & 0xF) << 28
            )
            packed_weight[i, :] = packed_val.to(torch.int32) # Ensure packed value is int32

        self.qweight = packed_weight.to(linear.weight.device) # Ensure final device


    def forward(self, x):
        if not _quant_cuda_4bit_available:
             raise ImportError("Cannot perform forward pass, CUDA 4-bit kernel not found.")

        out_shape = x.shape[:-1] + (self.outfeatures, )
        x_reshaped = x.reshape(-1, x.shape[-1]) # Flatten input to (batch_size * seq_len, infeatures)

        # Pad input features if necessary (must match padding done during packing)
        padded_infeatures = math.ceil(self.infeatures / 8.0) * 8
        if self.infeatures % 8 != 0:
            padding = padded_infeatures - self.infeatures
            x_padded = torch.cat([x_reshaped, torch.zeros((x_reshaped.shape[0], padding), dtype=x.dtype, device=x.device)], dim=1)
        else:
            x_padded = x_reshaped

        # Allocate output tensor
        if self.faster:
            # Faster kernel requires FP16 input, FP32 output
            if x_padded.dtype != torch.float16:
                # print("Warning: Faster kernel requires FP16 input. Casting input x.")
                x_padded = x_padded.to(torch.float16)
            out = torch.zeros((x_padded.shape[0], self.outfeatures), dtype=torch.float32, device=x.device)
            # Ensure scales/zeros are float32 for faster kernel
            scales_f32 = self.scales.float()
            zeros_f32 = self.zeros.float() # zero_point * scale
            quant_cuda_4bit.vecquant4matmul_faster(x_padded, self.qweight, out, scales_f32, zeros_f32)
        else:
            # Standard kernel uses input type for output (unless input is FP16, then output is FP16)
            # Let's make output FP32 for consistency if input is FP16, otherwise match input type
            out_dtype = torch.float32 if x_padded.dtype == torch.float16 else x_padded.dtype
            out = torch.zeros((x_padded.shape[0], self.outfeatures), dtype=out_dtype, device=x.device)
            # Ensure scales/zeros match input type for standard kernel
            scales = self.scales.to(x_padded.dtype)
            zeros = self.zeros.to(x_padded.dtype) # zero_point * scale
            quant_cuda_4bit.vecquant4matmul(x_padded, self.qweight, out, scales, zeros)

        # Add bias and reshape
        if hasattr(self, 'bias'):
             out = out + self.bias.to(out.dtype) # Ensure bias matches output type
        return out.reshape(out_shape)

# Helper function like make_quant3
def make_quant4(module, names, name='', faster=False):
    """
    Replaces linear layers in a module with Quant4Linear layers.

    Args:
        module (nn.Module): The module containing linear layers to replace.
        names (dict): A dictionary mapping layer names to their pre-computed
                      quantization scales and integer zero points
                      (e.g., {'layer.name': (scales, zeros_int)}).
        name (str): The current recursive name prefix for layers.
        faster (bool): Whether to use the faster CUDA kernel variant.
    """
    if not _quant_cuda_4bit_available:
        print("WARNING: CUDA 4-bit kernel not available, cannot replace layers with Quant4Linear.")
        return

    if isinstance(module, Quant4Linear): # Avoid double conversion
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names and isinstance(tmp, nn.Linear):
            # Replace nn.Linear with Quant4Linear
            scales, zeros_int = names[name1] # Zeros here should be integer zero point
            qlayer = Quant4Linear(tmp.in_features, tmp.out_features, faster=faster)
            qlayer.pack(tmp, scales, zeros_int) # Pack weights and quantization params
            delattr(module, attr) # Remove original layer
            setattr(module, attr, qlayer) # Add new quantized layer
            print(f"Replaced {name1} with Quant4Linear ({'faster' if qlayer.faster else 'standard'})")

    for name1, child in module.named_children():
        make_quant4(child, names, name + '.' + name1 if name != '' else name1, faster=faster)
