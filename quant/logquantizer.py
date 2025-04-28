import torch
from .abstractquant import QuantizerInterface

class LogQuantizer(QuantizerInterface):
    """
    Logarithmic quantizer based on the formula:
    Q(x) = Sign(x) * 2^(round(log2(|x|)))
    """

    def __init__(self, shape=1): # Shape might not be relevant here
        super(LogQuantizer, self).__init__()
        self.bits = -1 # To track if configured
        self.register_buffer('_is_ready', torch.tensor(False))


    def configure(self, bits, **kwargs):
        """
        Configure the quantizer. For basic log quantization, 
        bits aren't directly used in the formula but stored for potential future use 
        (e.g., limiting exponent range).
        """
        self.bits = bits


    def find_params(self, x, weight=False, **kwargs):
        """
        Determines parameters for quantization. For basic logarithmic quantization,
        parameters like scale/zero are not needed. This method primarily ensures
        the quantizer is marked as ready after being called.
        """
        self._is_ready = torch.tensor(True)
        return


    def quantize(self, x):
        """
        Applies logarithmic quantization to the input tensor x.
        Formula: Q(x) = Sign(x) * 2^(round(log2(|x|)))
        """
        if not self.ready():
             # Or raise an error, or return x if quantization shouldn't happen yet
            print("Warning: LogQuantizer not ready, returning original tensor.")
            return x

        sign = torch.sign(x)
        abs_x = torch.abs(x)
        q = torch.zeros_like(x)

        # Create mask for non-zero elements to avoid log2(0)
        non_zero_mask = abs_x > 1e-12 

        # Calculate log2 of absolute values for non-zero elements
        log2_abs_x = torch.log2(abs_x[non_zero_mask])

        rounded_log2 = torch.round(log2_abs_x)
        pow2_rounded_log2 = torch.pow(2.0, rounded_log2) 

        q[non_zero_mask] = sign[non_zero_mask] * pow2_rounded_log2

        return q.to(x.dtype) # Ensure output dtype matches input


    def ready(self):
        """
        Checks if the quantizer is ready (parameters found).
        """
        # For basic log quant, ready means configure/find_params has been called.
        return self._is_ready.item()
