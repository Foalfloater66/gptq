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
        # Initialize buffers for exponent range
        # Use float dtype for buffers to avoid potential type issues later
        self.register_buffer('min_exp', torch.tensor(0.0)) 
        self.register_buffer('max_exp', torch.tensor(0.0))
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
        Determines parameters for quantization. For logarithmic quantization,
        this calculates the representable exponent range based on input data range
        and configured bit width.
        """
        if self.bits <= 0:
            raise ValueError("LogQuantizer not configured with bits. Call configure() first.")

        dev = x.device
        # Ensure buffers are on the correct device
        self.min_exp = self.min_exp.to(dev)
        self.max_exp = self.max_exp.to(dev)
        self._is_ready = self._is_ready.to(dev)

        # Calculate max absolute value
        x_abs = torch.abs(x)
        # Add epsilon to avoid log2(-inf) if max_abs is 0
        max_abs = torch.max(x_abs) + 1e-12 

        # Use floor for max_exp to get the largest integer exponent <= log2(max_abs)
        max_exp_val = torch.floor(torch.log2(max_abs))

        # Calculate the number of positive levels based on bits
        # E.g., 4 bits -> 2^(4-1) = 8 positive levels (excluding zero)
        # Handle potential overflow for large bit numbers if necessary, though unlikely here
        num_positive_levels = 2**(self.bits - 1)

        # Determine the minimum exponent
        # The range includes max_exp_val down to min_exp_val
        min_exp_val = max_exp_val - num_positive_levels + 1

        # Store the calculated exponents
        self.max_exp.copy_(max_exp_val)
        self.min_exp.copy_(min_exp_val)
        self._is_ready.copy_(torch.tensor(True)) # Mark as ready

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
        # --- Perform calculations only on non-zero elements ---
        if torch.any(non_zero_mask):
            abs_x_nz = abs_x[non_zero_mask]
            sign_nz = sign[non_zero_mask]

            log2_abs_x = torch.log2(abs_x_nz)
            rounded_log2 = torch.round(log2_abs_x)

            # Clamp the exponents to the range determined in find_params
            clamped_log2 = torch.clamp(rounded_log2, self.min_exp, self.max_exp)

            # Calculate the final power-of-2 value using the clamped exponent
            pow2_clamped_log2 = torch.pow(2.0, clamped_log2)
            q[non_zero_mask] = sign_nz * pow2_clamped_log2
            # Store the clamped exponent for non-zero elements (initialize exponent tensor)
            clamped_log2_full = torch.zeros_like(x, dtype=clamped_log2.dtype) 
            clamped_log2_full[non_zero_mask] = clamped_log2
        else:
            # Handle case where all inputs are zero or below threshold
            clamped_log2_full = torch.zeros_like(x) # Or handle appropriately
        # --- End non-zero calculation ---

        # Return both the quantized value and the clamped exponent
        return q.to(x.dtype), clamped_log2_full.to(x.device)


    def ready(self):
        """
        Checks if the quantizer is ready (parameters found).
        """
        # Ready if find_params has been successfully run (indicated by _is_ready flag)
        # and bits have been configured.
        return self.bits > 0 and self._is_ready.item()
