# Non-Uniform Extended GPTQ
This repository contains the code for the COMP0252 Group Coursework Report "On the Compatibility of Non-Uniform Quantization Schemes with OPTQ-Based LLM Compression".
It extends the code for the ICLR 2023 paper [GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323). 
To refer to the original repository, see [GPTQ](https://github.com/IST-DASLab/gptq).

The extension includes the following features:
* Efficient implementations of non-uniform quantization methods: `quant/quantilequantizer.py`, `quant/logquantizer.py`, `quant/lloydmaxquant.py`, `quant/kmeansquantizer.py`, `quant/apotquantizer.py`
* Extension of OPT and BLOOM model family compression to include different quantization methods: `opt.py`, `bloom.py`
* General custom CUDA 11.7 kernels for 4-bit quantization: `quant_cuda_4bit.cpp`, `quant_cuda_kernel_4bit.cu`, `setup_cuda_4bit.py`
* Special logarithmic variant of the custom CUDA 11.7 kernels for 4-bit quantization: `logmatvec_cuda.cpp`, `logmatvec_cuda_kernel.cu`, `setup_logmatvec_cuda.py`


<!-- The current release includes the following features:

* An efficient implementation of the GPTQ algorithm: `gptq.py`
* Compressing all models from the OPT and BLOOM families to 2/3/4 bits, including weight grouping: `opt.py`, `bloom.py`, `zeroShot/`
* Evaluating the perplexity of quantized models on several language generation tasks: `opt.py`, `bloom.py`
* Evaluating the performance of quantized models on several ZeroShot tasks: `zeroShot/`
* A 3-bit quantized matrix full-precision vector product CUDA kernel: `quant_cuda_kernel.cu`, `quant_cuda.cpp`, `setup_cuda.py`
* Benchmarking code for individual matrix-vector products and for language generation with quantized models: `test_kernel.py`, `opt.py` -->

## Dependencies

* `torch`: tested on v1.13.0+cu117
* `transformers`: tested on v4.21.2
* `datasets`: tested on v3.5.1
* (to run 3-bit kernels: setup for compiling PyTorch CUDA extensions, see also https://pytorch.org/tutorials/advanced/cpp_extension.html, tested on CUDA 11.4)

All experiments were run on a RTX Quadro 6000.
## Language Generation

### OPT

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4
# Run GPTQ and compute results with quantizer choice: [uniform_minmax, quantile, lloydmax, logarithm, kmeans, apot]
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m c4 --wbits 4 --groupsize 1024 --quantizer [quantizer]
````

To run other OPT models replace `opt-125m` with one of: `opt-350m`, `opt-1.3b`, `opt-2.7b`, `opt-6.7b`, `opt-13b`.

### BLOOM

```
# Compute full precision (FP16) results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4
# Run RTN baseline and compute results
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --nearest
# Run GPTQ and compute results with quantizer choice: [uniform_minmax, quantile, lloydmax, logarithm, kmeans, apot]
CUDA_VISIBLE_DEVICES=0 python bloom.py bigscience/bloom-560m c4 --wbits 4 --groupsize 1024 --quantizer [quantizer]
````

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.

## ZeroShot

See `zeroShot/` folder.

## 4-bit CUDA Kernel

```
# Install 4 bit kernel
python setup_cuda_4bit.py install

# Test 4 bit kernel on quantizer method 
CUDA_VISIBLE_DEVICES=0 python test_kernel.py

# Benchmark language generation with 4-bit OPT-125m:
# OPT175B denotes the name of the folder with the HuggingFace OPT-175b checkpoint (see above)

# Save compressed model
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-125m wikitext2 --wbits 4 --quantizer uniform_minmax --save opt125m_4bit_minmax.pt --output-file results.json --benchmark 50
```

## Logarithmic kernel
```
# Install 4 bit kernel
python setup_logmatvec_kernel.py install

# Test 4 bit kernel on quantizer method 
CUDA_VISIBLE_DEVICES=0 python test_logmatvec_kernel.py

# Benchmark language generation with 3-bit OPT-175B:
# Save compressed model
CUDA_VISIBLE_DEVICES=0 python opt-log.py facebook/opt-125m wikitext2 --wbits 4 --quantizer logarithm --save opt125m_4bit_logarithmic.pt --benchmark 50
```

Please note that our 3-bit kernels are currently only optimized for OPT-175B running on 1xA100 or 2xA6000 and may thus yield suboptimal performance on smaller models or on other GPUs.

