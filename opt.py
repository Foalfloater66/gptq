import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
# Import necessary quantizers and the custom kernel
from quant import get_quantizer, LogQuantizer
import logmatvec_cuda # Import the compiled custom kernel
import copy # For deepcopying model if needed


# --- Custom Layer Definition (Bundled 1+3 bit) ---

class LogMatVecPackedLinear(nn.Module):
    # Needs ACT_BITS defined globally or passed in
    # Let's define it here for now, assuming 8-bit activations
    ACT_BITS = 8

    def __init__(self, in_features, out_features):
        super().__init__()
        if in_features % 2 != 0:
             raise ValueError("Input features must be even for 4-bit packing.")
        self.in_features = in_features
        self.out_features = out_features

        # Buffers to store packed weights and parameters (initialized empty)
        # Packed weights (int8, shape: out_features x in_features/2) - Stores bundled 4-bit codes
        self.register_buffer('packed_weights', torch.empty((out_features, in_features // 2), dtype=torch.int8))
        # Bias (float32, shape: out_features)
        self.register_buffer('bias', torch.empty(out_features, dtype=torch.float32))
        # Quantization parameters (scalar)
        self.register_buffer('min_exp', torch.tensor(0, dtype=torch.int32)) # Still needed for unmapping
        # Activation quantization scale (calibrated)
        self.register_buffer('activation_scale', torch.tensor(1.0, dtype=torch.float32))

    def configure_quantization(
        self,
        linear_layer: nn.Linear,
        quantizer: LogQuantizer,
        activation_scale: torch.Tensor
    ):
        """Quantizes weights from linear_layer, packs them, stores activation scale."""
        if linear_layer.in_features != self.in_features or linear_layer.out_features != self.out_features:
            raise ValueError("Layer dimensions mismatch")

        weight = linear_layer.weight.data # Assume already on correct device
        bias_data = linear_layer.bias.data if linear_layer.bias is not None else torch.zeros(self.out_features, device=weight.device)

        # Ensure quantizer is ready (find_params should ideally be called once globally or per layer type)
        quantizer.find_params(weight, weight=True)
        if not quantizer.ready():
             raise RuntimeError("Quantizer failed to find parameters.")

        # Quantize to get 4-bit nibbles
        packed_nibbles = quantizer.quantize(weight)

        # Pack nibbles into bytes
        packed_bytes = quantizer.pack(packed_nibbles)

        # Store packed data and parameters
        self.packed_weights.copy_(packed_bytes) # Store packed 4-bit codes
        self.bias.copy_(bias_data)
        self.min_exp.copy_(torch.tensor(int(quantizer.min_exp.item()), dtype=torch.int32))
        self.activation_scale.copy_(activation_scale.clamp(min=1e-9)) # Store calibrated scale

        # print(f"  Layer {self.out_features}x{self.in_features} configured.") # Optional print


    def forward(self, x):
        # x shape: (batch_size, sequence_length, in_features) or (..., in_features)
        original_shape = x.shape
        # Reshape input to (batch*seq, in_features) if needed
        if x.ndim > 2:
            x = x.reshape(-1, x.shape[-1])

        # --- Activation Quantization (Calibrated) ---
        q_max_act = 2**(self.ACT_BITS - 1) - 1
        q_min_act = -2**(self.ACT_BITS - 1)
        # Use the pre-calculated scale stored in the buffer
        delta_lsb = self.activation_scale
        # Quantize and clamp activations
        a_quant = torch.round(x / delta_lsb).clamp(q_min_act, q_max_act).to(torch.int32)
        # --- End Activation Quantization ---

        # Prepare output tensor - Match the expected model dtype (likely half)
        output_shape = (*x.shape[:-1], self.out_features)
        output_dtype = x.dtype # Assume output should match input dtype
        output = torch.empty(output_shape, dtype=output_dtype, device=x.device)

        # Kernel outputs float32, so we need a temporary float32 buffer
        output_float32 = torch.empty(output_shape, dtype=torch.float32, device=x.device)

        # Iterate over batch dimension (kernel expects 1D activation vector)
        # TODO: Optimize this loop - ideally use a batch-aware kernel
        for i in range(x.shape[0]):
            a_quant_single = a_quant[i].contiguous()
            # Use the single scalar scale for the whole layer
            delta_lsb_single = delta_lsb.item()
            # Write kernel output to the temporary float32 buffer slice
            output_single_float32 = output_float32[i]

            # Explicitly set device context before kernel launch
            with torch.cuda.device(self.packed_weights.device): # Use packed_weights device
                logmatvec_cuda.forward_packed4bit( # Call updated kernel signature
                    a_quant_single,
                    self.packed_weights, # Pass packed 4-bit codes
                    output_single_float32, # Write to float32 buffer slice
                    delta_lsb_single,
                    self.min_exp.item()
                )

        # Add bias (ensure bias is correct dtype before adding)
        # Cast kernel output back to target dtype before adding bias
        output = output_float32.to(output_dtype)
        output += self.bias.to(output_dtype) # Ensure bias matches output dtype

        # Reshape output to original shape (if needed)
        if len(original_shape) > 2:
            output = output.reshape(*original_shape[:-1], self.out_features)

        return output

# --- End Custom Layer Definition ---


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

# --- LogPack4bit Sequential Logic (Replaces original opt_sequential) ---
@torch.no_grad()
def opt_sequential(model, dataloader, quantizer_name, dev): # quantizer_name is less relevant here, but kept for signature
    print('Starting LogPack4bit quantization...')

    # Ensure global args is accessible or pass it in if needed
    # Assuming args is accessible globally as defined in if __name__ == '__main__'
    global args # Need to declare global to access args defined in main block

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


    quantizer = get_quantizer(quantizer_name)
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)

        subset = find_layers(layer)
        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quantizer()
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
            print(i, name)
            print('Quantizing ...')
            gptq[name].fasterquant(
                percdamp=args.percdamp, 
                groupsize=args.groupsize, 
                actorder=args.act_order, 
                static_groups=args.static_groups,
                log_error_scale_power=args.log_error_scale_power # Pass the new arg
            )
            quantizers['model.decoder.layers.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free()
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    model.config.use_cache = False
    layers = model.model.decoder.layers # Get layers

    # --- Activation Calibration Setup ---
    print("Collecting activation statistics...")
    act_scales = {}
    act_dict = {} # To store intermediate activations

    def stat_input_hook(m, x, y, name):
        # Find layer name corresponding to module m
        if isinstance(x, tuple):
            x = x[0]
        if name not in act_dict:
             act_dict[name] = []
        # Store activations on CPU to save GPU memory during calibration
        act_dict[name].append(x.detach().cpu())

    hooks = []
    hooked_layer_names = set()
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
             if m.in_features % 2 == 0: # Check if layer is compatible
                  if name not in hooked_layer_names:
                      hooks.append(
                           m.register_forward_hook(
                                lambda mod, inp, outp, n=name: stat_input_hook(mod, inp, outp, n)
                           )
                      )
                      hooked_layer_names.add(name)

    print(f"  Running {args.nsamples} calibration samples...")
    model.to(dev) # Move model to GPU for calibration run
    # Ensure embeddings are on the correct device
    if hasattr(model.model.decoder, 'embed_tokens'): model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    if hasattr(model.model.decoder, 'embed_positions'): model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    # Run calibration samples
    for i in range(args.nsamples):
         # Assuming dataloader is a list of batches
         if i >= len(dataloader):
             print(f"  Warning: Requested {args.nsamples} samples, but dataloader only has {len(dataloader)}. Stopping calibration early.")
             break
         batch = dataloader[i][0].to(dev)
         try:
              _ = model(batch)
         except Exception as e:
              print(f"Warning: Error during calibration forward pass on sample {i}: {e}")
              pass # Continue for now
    for h in hooks:
         h.remove()

    model.cpu() # Move model back to CPU after calibration
    torch.cuda.empty_cache()

    # Calculate activation scales
    q_max_act = 2**(LogMatVecPackedLinear.ACT_BITS - 1) - 1
    print("  Calculating activation scales...")
    for name, activations in act_dict.items():
         if not activations:
              print(f"  Warning: No activations collected for {name}. Skipping scale calculation.")
              continue
         try:
            act_tensor = torch.cat(activations, dim=0).float()
         except RuntimeError as e:
            print(f"  Warning: Could not concatenate activations for {name}. Error: {e}. Skipping.")
            continue
         if act_tensor.ndim < 2:
             print(f"  Warning: Activation tensor for {name} has unexpected shape {act_tensor.shape}. Skipping.")
             continue
         # Calculate max absolute value per feature, then take max across features for per-layer scale
         act_max_abs_per_feat = torch.max(torch.abs(act_tensor.view(-1, act_tensor.shape[-1])), dim=0)[0]
         layer_max_abs = torch.max(act_max_abs_per_feat)
         act_scales[name] = (layer_max_abs / q_max_act).clamp(min=1e-9) # Store scale on CPU
         # print(f"  Activation scale for {name}: {act_scales[name].item():.4f}") # Optional print

    del act_dict
    print("Activation statistics collected.")
    # --- End Activation Calibration ---


    # --- Log Quantization and Layer Replacement ---
    print("\nApplying Logarithmic Quantization and Packing...")
    if quantizer_name != 'logarithm' or args.wbits != 4:
         print("Warning: LogPack4bit flow selected but quantizer/wbits mismatch. Ensure --quantizer logarithm --wbits 4")

    log_quantizer = LogQuantizer()
    log_quantizer.configure(bits=args.wbits) # Should be 4

    layers_replaced_count = 0
    # Iterate through layers again for replacement (ensure layers is correct reference)
    layers = model.model.decoder.layers
    for i in range(len(layers)):
        print(f"Processing layer {i}...")
        layer = layers[i] # Keep on CPU
        # Use find_layers to get names relative to the current layer block
        layer_modules = find_layers(layer)

        for name, lin_layer in layer_modules.items():
            if not isinstance(lin_layer, nn.Linear): continue
            if lin_layer.in_features % 2 != 0:
                print(f"  Skipping {name}: Odd in_features ({lin_layer.in_features}).")
                continue

            # Construct the full name to look up the scale
            full_name_found = f"model.decoder.layers.{i}.{name}"

            if full_name_found not in act_scales:
                 print(f"  Warning: Activation scale not found for {full_name_found}. Skipping replacement.")
                 continue

            print(f"  Replacing {name} with LogMatVecPackedLinear...")
            lin_layer.to(dev) # Move original to GPU for quantization
            custom_layer = LogMatVecPackedLinear(lin_layer.in_features, lin_layer.out_features).to(dev)
            custom_layer.configure_quantization(
                lin_layer,
                log_quantizer,
                act_scales[full_name_found].to(dev) # Pass scale to GPU
            )
            lin_layer.cpu() # Move original back
            custom_layer.cpu() # Move custom layer to CPU

            # Replace the layer using the relative name 'name'
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            sub_layer_name = name.rsplit('.', 1)[-1]
            if parent_name:
                 # Need to get the parent module *within the current layer block*
                 parent_module = layer.get_submodule(parent_name)
            else:
                 # This case might happen if find_layers returns top-level modules in the block
                 # Check if 'name' directly exists in the layer block
                 if hasattr(layer, sub_layer_name):
                     parent_module = layer
                 else:
                     print(f"  Error: Could not find parent for {name} in layer {i}. Skipping replacement.")
                     continue
            setattr(parent_module, sub_layer_name, custom_layer)
            layers_replaced_count += 1

        # Keep layer[i] on CPU after processing its submodules
        # layers[i] = layer # Already modified in place

        if i % 5 == 0: torch.cuda.empty_cache()

    print(f"Logarithmic quantization finished. Replaced {layers_replaced_count} layers.")
    model.config.use_cache = use_cache
    return {}
# --- End LogPack4bit Sequential Logic ---


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
                quantizer = Quantizer()
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
    print(ppl.item())

    model.config.use_cache = use_cache

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
        '--quantizer', type=str, choices=['uniform_minmax', 'logarithm', 'quantile'],default='uniform_minmax',
        help="Which parameter quantizer to use.",
    )
    parser.add_argument(
        '--log-error-scale-power', type=float, default=0.0,
        help='Power p for scaling log quant error: err_scaled = err * (|q|+eps)^(-p). Default 0.0 (no scaling).'
    )

    args = parser.parse_args()

    if args.load:
        model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = opt_sequential(model, dataloader, args.quantizer, DEV)
        print(time.time() - tick)

    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            opt_multigpu(model, gpus)
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark(model, input_ids, check=args.check)
    if args.load:
        exit()

    datasets = ['wikitext2', 'ptb', 'c4'] 
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']
    for dataset in datasets: 
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        opt_eval(model, testloader, DEV)

    if args.save:
        opt_pack3(model, quantizers)
        torch.save(model.state_dict(), args.save) 
