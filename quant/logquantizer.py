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
        self.register_buffer('min_exp', torch.tensor(0.0)) # Store as float for consistency
        self.register_buffer('max_exp', torch.tensor(0.0)) # Store as float for consistency
        self.register_buffer('_exponent_range', torch.tensor(0.0)) # Store range size
        self.register_buffer('_is_ready', torch.tensor(False))


    def configure(self, bits, **kwargs):
        """
        Configure the quantizer. For log quantization with packing,
        bits determine the number of representable exponent levels.
        """
        if bits <= 1:
            raise ValueError("LogQuantizer requires bits > 1.")
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

        exponent_range_size = max_exp_val - min_exp_val + 1

        # --- Check if the range fits within the allocated bits ---
        # Number of levels available for magnitude = 2**(bits - 1)
        available_levels = 2**(self.bits - 1)
        if exponent_range_size > available_levels:
            print(
                f"Warning: LogQuantizer exponent range ({exponent_range_size}) exceeds "
                f"levels available for {self.bits} bits ({available_levels}). "
                f"Clamping range."
            )
            # Adjust min_exp to fit the available levels
            min_exp_val = max_exp_val - available_levels + 1
            exponent_range_size = available_levels # Update range size

        # Store the calculated exponents and range size
        self.max_exp.copy_(max_exp_val)
        self.min_exp.copy_(min_exp_val)
        self._exponent_range.copy_(exponent_range_size)
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

        # Return the quantized value, the integer exponent, and the sign
        return q.to(x.dtype), clamped_log2_full.to(torch.int32), sign.to(torch.int8)


    def pack(self, exponents, signs):
        """
        Packs mapped 4-bit exponents into int8 tensors.
        Assumes exponents are int32 and signs are int8.
        Requires find_params to have been called.
        """
        if not self.ready():
            raise RuntimeError("LogQuantizer must be configured and find_params called before packing.")
        if self.bits != 4:
            raise NotImplementedError("Packing currently only implemented for bits=4.")

        dev = exponents.device
        if signs.device != dev:
            raise ValueError("Exponents and signs must be on the same device.")

        # Map exponents to 0-15 range. Level 0 is reserved for true zero?
        # Let's map min_exp -> 0, max_exp -> N-1 where N = range_size
        # Map clamped exponents to an unsigned integer range [0, N-1]
        # N = self._exponent_range.item() # Number of levels
        min_exp_val = self.min_exp.item()
        mapped_exponents = (exponents - min_exp_val).to(torch.uint8) # Cast to uint8 after offset

        # Clamp mapped exponents just in case (shouldn't be needed if find_params/quantize are correct)
        # max_map_val = self._exponent_range.item() - 1
        # mapped_exponents = torch.clamp(mapped_exponents, 0, max_map_val)

        # Handle true zeros (where sign is 0). We need a specific packed code for zero.
        # Let's use the mapped exponent value 15 (0xF) to represent zero?
        # Or should we handle it via the sign tensor? Let's keep sign separate.
        # The kernel will check the sign first. If sign is 0, exponent doesn't matter.

        # Ensure the mapped exponents fit in 4 bits (0-15)
        if torch.any(mapped_exponents > 15):
             print(f"Warning: Mapped exponents exceed 4-bit range (0-15). Clamping.")
             mapped_exponents = torch.clamp(mapped_exponents, 0, 15)

        # Packing: Combine two 4-bit values into one int8
        # Ensure the input feature dimension is even for simplicity
        if exponents.shape[-1] % 2 != 0:
            raise ValueError("Last dimension of exponents must be even for 4-bit packing.")

        # Reshape to group elements in pairs
        # Example: (N_out, M_in) -> (N_out, M_in // 2, 2)
        reshaped_exponents = mapped_exponents.view(*exponents.shape[:-1], -1, 2)

        # Pack: high_nibble = exp1 << 4, low_nibble = exp2
        packed_exponents = (reshaped_exponents[..., 0] << 4) | reshaped_exponents[..., 1]

        # Return packed exponents (int8) and original signs (int8)
        return packed_exponents.to(torch.int8), signs


    def ready(self):
        """
        Checks if the quantizer is ready (parameters found).
        """
        # Ready if find_params has been successfully run (indicated by _is_ready flag)
        # and bits have been configured.
        return self.bits > 0 and self._is_ready.item()
