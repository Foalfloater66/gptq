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
        Configure the quantizer. For 1+3 bit log quantization, bits must be 4.
        """
        if bits != 4:
            # Or adapt logic for other bitwidths if needed later
            raise NotImplementedError("Bundled 1+3 bit LogQuantizer currently only supports bits=4.")
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

        # For 1+3 bit scheme, we have 3 bits for exponent magnitude levels
        num_exponent_bits = self.bits - 1
        available_levels = 2**num_exponent_bits # 2^3 = 8 levels
        exponent_range_size = max_exp_val - min_exp_val + 1

        # --- Check if the range fits within the allocated exponent bits ---
        if exponent_range_size > available_levels:
            print(
                f"Warning: LogQuantizer exponent range ({exponent_range_size}) exceeds "
                f"levels available for {num_exponent_bits} exponent bits ({available_levels}). "
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

        # --- Bundled 1+3 bit Quantization ---
        # Calculate sign bit (0 for positive, 1 for negative)
        sign_bit = (sign < 0).to(torch.uint8) # 1 if negative, 0 otherwise

        # Map clamped exponents to an unsigned integer range [0, 7] for the 3 exponent bits
        min_exp_val = self.min_exp.item()
        # Add small epsilon before casting to handle potential floating point inaccuracies near boundaries
        exponent_map = (clamped_log2_full - min_exp_val + 1e-6).clamp(min=0).to(torch.uint8)

        # Ensure exponent map is within 0-7 range
        if torch.any(exponent_map > 7):
             print(f"Warning: Mapped exponents exceed 3-bit range (0-7). Clamping.")
             exponent_map = torch.clamp(exponent_map, 0, 7)

        # Combine sign bit (MSB) and exponent map (3 LSBs)
        # packed_nibble = (sign_bit << 3) | exponent_map
        # Handle the special zero code '0000'
        # If original sign was 0, output 0000.
        # Also, if sign is positive (sign_bit=0) and exponent map is 0 (exp_map=0),
        # this combination (0000) is reserved for zero. We need to map the smallest
        # positive magnitude (+2^min_exp) to a different code, e.g., 0001.
        # Let's adjust the exponent map for the smallest positive value:
        # If sign_bit is 0 and exponent_map is 0, make exponent_map 1 (0001)
        # This means the effective range for positive exponents is [min_exp+1, max_exp]
        # Or, more simply, map min_exp to level 1 for positive numbers.
        # Let's remap: map [min_exp, max_exp] to [0, 7]
        # Smallest positive magnitude (exponent=min_exp) maps to exp_map=0. Code is 0000.
        # Use 0000 as the dedicated zero code.
        # Map smallest positive magnitude (exponent=min_exp) to code 0001.
        # Map largest positive magnitude (exponent=max_exp) to code 0111. (7 levels)
        # Map smallest negative magnitude (exponent=min_exp) to code 1000.
        # Map largest negative magnitude (exponent=max_exp) to code 1111. (8 levels)

        # Recalculate exponent map to fit this new scheme [0..7] -> [1..7] for positive
        # Map [min_exp, max_exp] to [0, 7]
        exponent_map = (clamped_log2_full - min_exp_val + 1e-6).clamp(min=0).to(torch.uint8)
        exponent_map = torch.clamp(exponent_map, 0, 7) # Ensure range 0-7

        # Create the 4-bit code tensor
        packed_nibbles = torch.zeros_like(x, dtype=torch.uint8)

        # Apply positive codes (0001 to 0111)
        positive_mask = (sign > 0) & non_zero_mask
        # Map exponent range [0, 6] to codes [1, 7] for positive values
        # Smallest positive (exp_map=0) becomes code 1. Largest (exp_map=7) becomes code 7? No, 7 levels.
        # Let's use 7 levels for positive: map [min_exp, max_exp-1] to [1, 7]
        # Max positive exponent map value is 6 for codes 1-7
        positive_exp_map = torch.clamp(exponent_map[positive_mask], 0, 6) # Clamp to 0-6
        packed_nibbles[positive_mask] = positive_exp_map + 1 # Map to 1-7

        # Apply negative codes (1000 to 1111)
        negative_mask = (sign < 0) & non_zero_mask
        # Map exponent range [0, 7] to codes [8, 15] for negative values
        negative_exp_map = exponent_map[negative_mask] # Use full 0-7 range
        packed_nibbles[negative_mask] = (1 << 3) | negative_exp_map # Set sign bit (1xxx)

        # Zero values remain 0000

        # Return only the tensor containing the 4-bit codes (stored in uint8)
        return packed_nibbles


    def pack(self, packed_nibbles):
        """
        Packs 4-bit codes (stored in uint8 tensors) into int8 tensors.
        Requires find_params to have been called.
        """
        if not self.ready():
            raise RuntimeError("LogQuantizer must be configured and find_params called before packing.")
        if self.bits != 4:
            raise NotImplementedError("Packing currently only implemented for bits=4.")

        dev = packed_nibbles.device

        # Packing: Combine two 4-bit values into one int8
        if packed_nibbles.shape[-1] % 2 != 0:
            raise ValueError("Last dimension must be even for 4-bit packing.")

        # Reshape to group elements in pairs
        reshaped_nibbles = packed_nibbles.view(*packed_nibbles.shape[:-1], -1, 2)

        # Pack: high_nibble = nibble1 << 4, low_nibble = nibble2
        # Ensure nibbles are uint8 before shifting
        packed_bytes = (reshaped_nibbles[..., 0].to(torch.uint8) << 4) | reshaped_nibbles[..., 1].to(torch.uint8)

        # Return packed bytes as int8 (bits are preserved)
        return packed_bytes.to(torch.int8)


    def ready(self):
        """
        Checks if the quantizer is ready (parameters found).
        """
        # Ready if find_params has been successfully run (indicated by _is_ready flag)
        # and bits have been configured.
        return self.bits > 0 and self._is_ready.item()
