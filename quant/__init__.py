from .quantilequantizer import QuantileQuantizer
from .lloydmaxquant import LloydMaxQuantizer
from .minmaxquant import Quantizer as MinMaxQuantizer
from .minmaxquant import Quant3Linear
from .logquantizer import LogQuantizer
from .kmeansquantizer import KMeansQuantizer # <-- Import added
from .apotquantizer import APoTQuantizer # <-- Import added
# Keep the standalone affine quantize function if needed elsewhere
from .minmaxquant import quantize as affine_quantize

def get_quantizer(quantizer_name):
    """Returns the constructor for the specified quantizer."""
    if quantizer_name == 'quantile':
        print("Using Quantile Quantizer")
        return QuantileQuantizer
    elif quantizer_name == 'lloydmax':
        return LloydMaxQuantizer
    elif quantizer_name == 'logarithm':
        print("Using Logarithmic Quantizer")
        return LogQuantizer
    elif quantizer_name == 'uniform_minmax':
        print("Using Uniform MinMax Quantizer")
        return MinMaxQuantizer # Default affine quantizer
    elif quantizer_name == 'kmeans':
        print("Using K-Means Quantizer")
        return KMeansQuantizer
    elif quantizer_name == 'apot':
        print("Using APoT Quantizer (k=2, PTQ variant)")
        return APoTQuantizer
    else:
        # Fallback or error for unknown quantizer
        print(f"Warning: Unknown quantizer '{quantizer_name}', falling back to Uniform MinMax.")
        return MinMaxQuantizer
