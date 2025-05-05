import math
import torch
import torch.nn as nn

# Import the compiled 4-bit CUDA kernel
try:
    import quant_cuda_4bit
    print("CUDA 4-bit kernel found")
    _quant_cuda_4bit_available = True
except ImportError:
    print('CUDA 4-bit kernel not found.')
    _quant_cuda_4bit_available = False


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
