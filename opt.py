import time
import json # <-- Add import
import os   # <-- Add import

import torch
import torch.nn as nn
import math # Add math import
import json # <-- Add import
import os   # <-- Add import

from gptq import *
from modelutils import *
# Import MinMaxQuantizer specifically, and other needed components
from quant import get_quantizer, MinMaxQuantizer, Quant3Linear, make_quant3
from quant.quant4linear import Quant4Linear, make_quant4 # Import 4-bit layer

# Import CUDA modules
try:
    import quant_cuda
    _quant_cuda_3bit_available = True
except ImportError:
    print("CUDA 3-bit kernel not found.")
    _quant_cuda_3bit_available = False

try:
    import quant_cuda_4bit
    _quant_cuda_4bit_available = True
except ImportError:
    print("CUDA 4-bit kernel not found.")
    _quant_cuda_4bit_available = False


def get_opt(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = model.config.max_position_embeddings
    return model

@torch.no_grad()
def opt_sequential(model, dataloader, quantizer_name, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev) 
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    # Get the constructor for the chosen quantizer
    SpecificQuantizerClass = get_quantizer(quantizer_name)
    quantizers_for_packing = {} # Store affine params (scale, zero) needed for packing
    for i in range(len(layers)):
        log_print(f"\nProcessing layer {i}")
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            # Instantiate the specific quantizer class obtained earlier
            gptq[name].quantizer = SpecificQuantizerClass()
            gptq[name].quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=args.trits
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            log_print(f"Layer {i}, Module {name}")
            log_print('Running GPTQ + Quantization...')
            # GPTQ uses the specific quantizer internally via gptq[name].quantizer
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
            )

            # --- Crucial Step: Get Affine Params for Packing ---
            # The weight in subset[name] is now quantized by the specific method.
            # We need to find the best affine representation (scale/zero) for this quantized weight.
            W_quant = subset[name].weight.data.clone()
            affine_quantizer = MinMaxQuantizer() # Use MinMax to find affine params
            affine_quantizer.configure(args.wbits, perchannel=True, sym=args.sym)
            affine_quantizer.find_params(W_quant, weight=True)

            # Store the affine scale and zero point for packing later
            layer_name_str = f'model.decoder.layers.{i}.{name}'
            quantizers_for_packing[layer_name_str] = (affine_quantizer.scale.cpu(), affine_quantizer.zero.cpu())
            # ----------------------------------------------------

            gptq[name].free()

        # Run forward pass again to update output states (important for sequential application)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    # Return the dictionary containing affine parameters needed for packing
    return quantizers_for_packing

@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                # Use the imported MinMaxQuantizer
                quantizer = MinMaxQuantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    # print(ppl.item()) # Keep commented out or remove entirely

    model.config.use_cache = use_cache

    return ppl.item() # Return the perplexity


def benchmark_kernels(args, dev):
    """Runs low-level benchmarks for the 3-bit and 4-bit CUDA kernels."""
    print('Benchmarking Custom CUDA Kernels ...')

    # --- Benchmark Dimensions (from test_kernel.py) ---
    M_BENCH = 12288 * 4 # Input features (e.g., OPT-175B FFN)
    N_BENCH = 12288     # Output features
    COUNT = 1000        # Number of iterations

    # --- Setup Data ---
    DTYPE_FP16 = torch.half
    DTYPE_FLOAT = torch.float
    vec_fp16 = torch.randn((1, M_BENCH), device=dev, dtype=DTYPE_FP16)
    vec_f32 = vec_fp16.float()
    mul_fp16 = torch.zeros((1, N_BENCH), device=dev, dtype=DTYPE_FP16)
    mul_f32 = torch.zeros((1, N_BENCH), device=dev, dtype=DTYPE_FLOAT)

    # --- FP16 Baseline (Optional but good reference) ---
    print("\n--- FP16 PyTorch Benchmark ---")
    mat_fp16 = torch.randn((M_BENCH, N_BENCH), device=dev, dtype=DTYPE_FP16)
    tick = time.time()
    for _ in range(COUNT):
        torch.matmul(vec_fp16, mat_fp16, out=mul_fp16)
        torch.cuda.synchronize()
    print(f'FP16 MatVec Time: {(time.time() - tick) / COUNT:.6f} seconds')
    del mat_fp16, mul_fp16 # Free memory

    # --- 3-bit Benchmark ---
    if _quant_cuda_3bit_available:
        print("\n--- 3-bit Kernel Benchmark ---")
        packed_height_3bit = M_BENCH // 32 * 3
        mat_int3_packed = torch.randint(-1000000000, 1000000000, (packed_height_3bit, N_BENCH), device=dev, dtype=torch.int)
        scales_3bit = torch.randn(N_BENCH, device=dev, dtype=DTYPE_FLOAT)
        zeros_3bit = torch.randn(N_BENCH, device=dev, dtype=DTYPE_FLOAT) # zero_point * scale

        # Benchmark standard 3-bit kernel (FP32 input)
        mul_f32.zero_()
        tick = time.time()
        for _ in range(COUNT):
            quant_cuda.vecquant3matmul(vec_f32, mat_int3_packed, mul_f32, scales_3bit, zeros_3bit)
            torch.cuda.synchronize()
        print(f'3-bit MatVec Time (FP32 in): {(time.time() - tick) / COUNT:.6f} seconds')

        # Benchmark faster 3-bit kernel (FP16 input -> FP32 output)
        mul_f32.zero_()
        tick = time.time()
        for _ in range(COUNT):
            quant_cuda.vecquant3matmul_faster(vec_fp16, mat_int3_packed, mul_f32, scales_3bit, zeros_3bit)
            torch.cuda.synchronize()
        print(f'3-bit MatVec Time (Faster, FP16 in): {(time.time() - tick) / COUNT:.6f} seconds')
        del mat_int3_packed, scales_3bit, zeros_3bit # Free memory
    else:
        print("\nSkipping 3-bit kernel benchmark (CUDA module not found).")

    # --- 4-bit Benchmark ---
    if _quant_cuda_4bit_available:
        print("\n--- 4-bit Kernel Benchmark ---")
        # Ensure M_BENCH is divisible by 8
        if M_BENCH % 8 != 0:
             print(f"Warning: M_BENCH ({M_BENCH}) not divisible by 8. Adjusting for benchmark.")
             M_BENCH_4BIT = (M_BENCH // 8) * 8
             vec_fp16_4b = vec_fp16[:, :M_BENCH_4BIT]
             vec_f32_4b = vec_f32[:, :M_BENCH_4BIT]
        else:
             M_BENCH_4BIT = M_BENCH
             vec_fp16_4b = vec_fp16
             vec_f32_4b = vec_f32

        packed_height_4bit = M_BENCH_4BIT // 8
        min_int32 = -(2**31)
        max_int32_exclusive = 2**31
        mat_int4_packed = torch.randint(min_int32, max_int32_exclusive, (packed_height_4bit, N_BENCH), device=dev, dtype=torch.int32)
        scales_4bit_f32 = torch.randn(N_BENCH, device=dev, dtype=DTYPE_FLOAT)
        zeros_4bit_f32 = torch.randn(N_BENCH, device=dev, dtype=DTYPE_FLOAT) # zero_point * scale
        scales_4bit_f16 = scales_4bit_f32.half()
        zeros_4bit_f16 = zeros_4bit_f32.half()

        # Benchmark standard 4-bit kernel (FP32 input -> FP32 output)
        mul_f32.zero_()
        tick = time.time()
        for _ in range(COUNT):
            quant_cuda_4bit.vecquant4matmul(vec_f32_4b, mat_int4_packed, mul_f32, scales_4bit_f32, zeros_4bit_f32)
            torch.cuda.synchronize()
        print(f'4-bit MatVec Time (FP32 in -> FP32 out): {(time.time() - tick) / COUNT:.6f} seconds')

        # Benchmark standard 4-bit kernel (FP16 input -> FP32 output)
        mul_f32.zero_()
        tick = time.time()
        for _ in range(COUNT):
            quant_cuda_4bit.vecquant4matmul(vec_fp16_4b, mat_int4_packed, mul_f32, scales_4bit_f16, zeros_4bit_f16)
            torch.cuda.synchronize()
        print(f'4-bit MatVec Time (FP16 in -> FP32 out): {(time.time() - tick) / COUNT:.6f} seconds')

        # Benchmark faster 4-bit kernel (FP16 input -> FP32 output)
        mul_f32.zero_()
        tick = time.time()
        for _ in range(COUNT):
            quant_cuda_4bit.vecquant4matmul_faster(vec_fp16_4b, mat_int4_packed, mul_f32, scales_4bit_f32, zeros_4bit_f32)
            torch.cuda.synchronize()
        print(f'4-bit MatVec Time (Faster, FP16 in -> FP32 out): {(time.time() - tick) / COUNT:.6f} seconds')
        del mat_int4_packed, scales_4bit_f32, zeros_4bit_f32, scales_4bit_f16, zeros_4bit_f16 # Free memory
    else:
        print("\nSkipping 4-bit kernel benchmark (CUDA module not found).")

    print("\nKernel benchmark finished.")


# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers, faster=args.faster_kernel)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

# Added function for 4-bit packing
def opt_pack4(model, quantizers):
    """Packs OPT model weights into 4-bit Quant4Linear layers."""
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant4(model, quantizers, faster=args.faster_kernel) # Use make_quant4
    qlayers = find_layers(model, [Quant4Linear]) # Find Quant4Linear layers
    print('Packing 4-bit ...')
    for name in qlayers:
        print(name)
        # Ensure scale/zero are on CPU before accessing .pack
        # .pack expects integer zero points
        scale, zero = quantizers[name]
        qlayers[name].pack(layers[name], scale.to(qlayers[name].qweight.device), zero.to(qlayers[name].qweight.device))
    print('Done.')
    return model


def load_quant3(model, checkpoint):
    from transformers import OPTConfig, OPTForCausalLM 
    config = OPTConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant3(model, layers, faster=args.faster_kernel)

    print('Loading model ...')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

# Added function for loading 4-bit model
def load_quant4(model_path, checkpoint_path, faster=False):
    """Loads a 4-bit quantized OPT model."""
    from transformers import OPTConfig, OPTForCausalLM
    config = OPTConfig.from_pretrained(model_path)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = OPTForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    # Exclude layers that are usually not quantized
    for name in ['model.decoder.project_out', 'model.decoder.project_in', 'lm_head']:
        if name in layers:
            del layers[name]
    make_quant4(model, layers, faster=faster) # Use make_quant4

    print('Loading 4-bit model ...')
    if checkpoint_path:
         model.load_state_dict(torch.load(checkpoint_path))
    else:
         print("Warning: No checkpoint path provided for load_quant4. Model weights are initialized but not loaded.")
    model.seqlen = model.config.max_position_embeddings
    print('Done.')
    return model


def opt_multigpu(model, gpus):
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(gpus[0])
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(gpus[0])
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(gpus[0])
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(gpus[-1])
    if hasattr(model.model.decoder, 'final_layer_norm') and model.model.decoder.final_layer_norm:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.decoder.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

def benchmark(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.decoder.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Whether to compute perplexity during benchmarking for verification.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to use the new faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--quantizer', type=str, choices=['uniform_minmax', 'logarithm', 'quantile', 'kmeans', 'apot', 'lloydmax'], default='uniform_minmax',
        help="Which parameter quantizer to use.",
    )
    parser.add_argument(
        '--output-file', type=str, default=None,
        help='Path to save evaluation results (JSON Lines format).'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce verbose logging during quantization and evaluation.'
    )
    # Remove benchmark-kernels, use --benchmark for model inference benchmark
    # parser.add_argument(
    #     '--benchmark-kernels', action='store_true',
    #     help='Run low-level CUDA kernel benchmarks instead of model evaluation.'
    # )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.wbits not in [3, 4, 16]:
        raise ValueError("Only 3, 4, or 16 bits supported for --wbits with kernel evaluation.")
    if args.wbits == 3 and not _quant_cuda_3bit_available:
        raise ImportError("3-bit quantization requested, but 3-bit CUDA kernel not found. Build it first.")
    if args.wbits == 4 and not _quant_cuda_4bit_available:
        raise ImportError("4-bit quantization requested, but 4-bit CUDA kernel not found. Build it first.")
    if args.load and args.save:
        raise ValueError("Cannot specify both --load and --save")
    if args.nearest:
        print("Warning: --nearest flag is ignored when wbits < 16 in this script.")

    # Define DEV globally or pass args around if needed
    DEV = torch.device('cuda:0') # Assumes CUDA_VISIBLE_DEVICES handles mapping

    # Add conditional printing based on --quiet flag
    def log_print(*print_args, **kwargs):
        if not args.quiet:
            print(*print_args, **kwargs)

    # --- Replace print calls with log_print ---
    # Note: You'll need to manually update prints within opt_sequential and opt_eval
    # if you want them silenced by --quiet. For brevity, only key examples shown here.
    # Example: Replace print('Starting ...') with log_print('Starting ...') in opt_sequential/opt_eval
    # Example: Replace print(i, name) with log_print(i, name) in opt_sequential
    # Example: Replace print('Quantizing ...') with log_print('Quantizing ...') in opt_sequential
    # Example: Replace print(i) with log_print(i) in opt_eval

    # --- Run Kernel Benchmark if requested ---
    # Removed kernel benchmark section, use --benchmark for model benchmark

    # --- Main Logic ---
    if args.load:
        log_print(f"Loading quantized model from: {args.load}")
        if args.wbits == 3:
            model = load_quant3(args.model, args.load)
        elif args.wbits == 4:
            # Pass faster_kernel flag to load_quant4
            model = load_quant4(args.model, args.load, faster=args.faster_kernel)
        else:
            # Should not happen due to arg validation, but good practice
             raise ValueError(f"Loading model with {args.wbits} bits not supported.")
        model.eval() # Ensure model is in eval mode
    else:
        # Load FP16 model for quantization or baseline eval
        model = get_opt(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    quantizers = None # Initialize quantizers
    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        # Pass log_print or args if needed inside opt_sequential
        # quantizers now holds affine parameters (scale, zero)
        affine_quantizers = opt_sequential(model, dataloader, args.quantizer, DEV)
        log_print(f"Quantization time: {time.time() - tick:.2f}s")

        # Pack the model using the obtained affine quantizers
        log_print("Packing model weights...")
        if args.wbits == 3:
            model = opt_pack3(model, affine_quantizers)
        elif args.wbits == 4:
            model = opt_pack4(model, affine_quantizers)
        log_print("Packing complete.")

    # --- Benchmarking ---
    if args.benchmark:
        log_print(f"Benchmarking inference speed with {args.benchmark} tokens...")
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            # Use testloader for benchmark input for consistency? Or dataloader? Using dataloader.
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            # Ensure model is on the correct device(s) before benchmark
            if len(gpus) > 1:
                 opt_multigpu(model, gpus)
            else:
                 model = model.to(DEV)
            benchmark(model, input_ids, check=args.check)
        log_print("Benchmarking complete.")

    # --- Evaluation Loop ---
    evaluation_results = {}
    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(f"Evaluating on: {dataset}") # Keep this print for clarity
        # Pass log_print or args if needed inside opt_eval
        ppl = opt_eval(model, testloader, DEV)
        print(f"Perplexity: {ppl:.4f}") # Keep this print for immediate feedback
        evaluation_results[dataset] = ppl

    # Save results if output file specified
    if args.output_file:
        output_data = {
            'model': args.model,
            'quantizer': args.quantizer,
            'wbits': args.wbits,
            'groupsize': args.groupsize,
            'sym': args.sym,
            'percdamp': args.percdamp,
            'act_order': args.act_order,
            'static_groups': args.static_groups,
            'trits': args.trits,
            'results': evaluation_results
        }
        # Append results as a new line in JSON Lines format
        mode = 'a' if os.path.exists(args.output_file) else 'w'
        try:
            with open(args.output_file, mode) as f:
                f.write(json.dumps(output_data) + '\n')
            print(f"Results appended to {args.output_file}")
        except IOError as e:
            print(f"Error writing to output file {args.output_file}: {e}")

    # --- Save Model and Report Size ---
    if args.save:
        log_print(f"Saving quantized model to {args.save} ...")
        # Ensure model is on CPU before saving to avoid GPU memory in file
        model.cpu()
        torch.save(model.state_dict(), args.save)
        log_print("Model saved.")
        try:
            file_size_mb = os.path.getsize(args.save) / (1024 * 1024)
            log_print(f"Quantized model file size: {file_size_mb:.2f} MB")
            # Optionally compare to FP16 size (requires loading original model again)
            # model_fp16 = get_opt(args.model)
            # torch.save(model_fp16.state_dict(), "temp_fp16.pt")
            # fp16_size_mb = os.path.getsize("temp_fp16.pt") / (1024 * 1024)
            # log_print(f"Original FP16 model file size: {fp16_size_mb:.2f} MB")
            # os.remove("temp_fp16.pt")
        except Exception as e:
            log_print(f"Could not determine file size: {e}")
