from .quantilequantizer import QuantileQuantizer
from .minmaxquant import Quantizer, quantize

def get_quantizer(quantizer_name):
    if quantizer_name == 'quantile':
        return QuantileQuantizer
    elif quantizer_name == 'logarithm':
        raise Exception("Not Implemented.")
    return Quantizer# default is uniform min max.