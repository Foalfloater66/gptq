import torch
from .abstractquant import QuantizerInterface
import warnings
import itertools

class APoTQuantizer(QuantizerInterface):
    """
    Additive Powers-of-Two (APoT) based quantizer (PTQ variant).
    Represents weights as sums of k=2 signed powers-of-two.
    Finds a clipping threshold alpha based on weight distribution.
    Pre-computes representable levels within [-alpha, alpha].
    """

    def __init__(self, shape=1): # Shape not relevant here
        super(APoTQuantizer, self).__init__()
        self.bits = -1
        self.k = 2 # Fixed number of terms for simplicity
        self.alpha_percentile = 99.9 # Percentile for clipping threshold alpha

        self.register_buffer("alpha", torch.tensor(0.0)) # Clipping threshold
        # Buffer to store the quantization levels (representable APoT values)
        self.register_buffer("levels", torch.zeros(1)) 
        self.register_buffer('_is_ready', torch.tensor(False))

    def configure(self, bits, **kwargs):
        """Configure the quantizer with the number of bits."""
        if bits <= 0:
            raise ValueError("Number of bits must be positive.")
        self.bits = bits
        # Could potentially parse k from kwargs if needed later

    def _generate_apot_levels(self, min_exp, max_exp, alpha, dev):
        """Helper to generate representable APoT levels for k=2."""
        if max_exp < min_exp:
             warnings.warn(f"APoT: max_exp ({max_exp}) < min_exp ({min_exp}). Resulting levels might be limited.")
             # Handle this case: maybe just return powers of 2 in the limited range?
             # For now, proceed, but level generation might be sparse.
             exponents = torch.arange(min_exp, max_exp + 1, device=dev)
        else:
            exponents = torch.arange(min_exp, max_exp + 1, device=dev)

        if exponents.numel() == 0:
            warnings.warn("APoT: No valid exponents found in range. Returning only zero.")
            return torch.tensor([0.0], device=dev)
            
        # Generate powers of 2
        powers_of_2 = 2.0 ** exponents

        # Generate all sums of k=2 signed powers-of-two
        levels = set([0.0])
        possible_terms = torch.cat([powers_of_2, -powers_of_2])
        
        # Generate sums s1*2^p1 + s2*2^p2
        for term1, term2 in itertools.combinations_with_replacement(possible_terms, self.k):
             level = term1 + term2
             if abs(level) <= alpha:
                 levels.add(level.item()) # Add as float to set

        levels_tensor = torch.tensor(list(levels), device=dev).sort()[0]
        
        return levels_tensor


    def find_params(self, x, weight=False, **kwargs):
        """
        Determines clipping threshold alpha and pre-computes APoT levels.
        """
        if self.bits <= 0:
            raise ValueError("APoTQuantizer not configured. Call configure() first.")

        dev = x.device
        self.alpha = self.alpha.to(dev)
        self.levels = self.levels.to(dev)
        self._is_ready = self._is_ready.to(dev)

        x_flat = x.flatten().float()

        # Determine Clipping Threshold alpha
        abs_x = torch.abs(x_flat)
        if abs_x.numel() > 0:
             alpha_val = torch.kthvalue(abs_x, int(abs_x.numel() * self.alpha_percentile / 100.0))[0]
             # Handle case where alpha might be zero
             if alpha_val <= 1e-9:
                 alpha_val = torch.max(abs_x) 
                 if alpha_val <= 1e-9:
                      alpha_val = torch.tensor(1.0)
             self.alpha.copy_(alpha_val)
        else:
             self.alpha.copy_(torch.tensor(1.0))

        # Determine Exponent Range based on alpha and bits
        alpha_eff = self.alpha.item() + 1e-12 
        max_exp = torch.floor(torch.log2(torch.tensor(alpha_eff)))

        num_positive_exponents = 2**(self.bits -1)
        min_exp = max_exp - num_positive_exponents + 1

        # Generate and Store APoT Levels
        apot_levels = self._generate_apot_levels(min_exp.item(), max_exp.item(), self.alpha.item(), dev)
        self.levels = apot_levels

        self._is_ready.copy_(torch.tensor(True))
        return

    def quantize(self, x):
        """
        Clips input tensor x to [-alpha, alpha] and maps each element 
        to the nearest pre-computed APoT level.
        """
        if not self.ready():
            warnings.warn("APoTQuantizer not ready, returning original tensor.")
            return x

        levels = self.levels.to(x.device)
        alpha = self.alpha.to(x.device)
        
        x_clipped = torch.clamp(x, -alpha, alpha)
        
        original_shape = x.shape
        x_flat = x_clipped.flatten()

        diffs = torch.abs(x_flat.unsqueeze(1) - levels.unsqueeze(0)) # Shape: (N, num_levels)
        assignments = torch.argmin(diffs, dim=1) # Shape: (N)

        q_flat = levels[assignments]

        return q_flat.view(original_shape).to(x.dtype)

    def ready(self):
        """Checks if the quantizer is ready (levels computed)."""
        # Check if bits configured and find_params has run
        return self.bits > 0 and self._is_ready.item()
