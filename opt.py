import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
# Import necessary quantizers and the custom kernel
from quant import get_quantizer, LogQuantizer
import logmatvec_cuda # Import the compiled custom kernel
import copy # For deepcopying model if needed


# --- Custom Layer Definition ---

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
        # Packed exponents (int8, shape: out_features x in_features/2)
        self.register_buffer('packed_exponents', torch.empty((out_features, in_features // 2), dtype=torch.int8))
        # Signs (int8, shape: out_features x in_features)
        self.register_buffer('signs', torch.empty((out_features, in_features), dtype=torch.int8))
        # Bias (float32, shape: out_features)
        self.register_buffer('bias', torch.empty(out_features, dtype=torch.float32))
        # Quantization parameters (scalar)
        self.register_buffer('min_exp', torch.tensor(0, dtype=torch.int32))
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

        # Quantize to get 4-bit nibbles (bundled codes)
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
        # Need to handle potential shape mismatch if scale is per-token/channel (not implemented here)
        delta_lsb = self.activation_scale
        # Quantize and clamp activations
        a_quant = torch.round(x / delta_lsb).clamp(q_min_act, q_max_act).to(torch.int32)
        # --- End Activation Quantization ---

        # Prepare output tensor - Match the expected model dtype (likely half)
        output_shape = (*x.shape[:-1], self.out_features)
        # Determine dtype from input or a layer parameter if possible, default to half
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
            with torch.cuda.device(self.packed_exponents.device):
                logmatvec_cuda.forward_packed4bit(
                    a_quant_single,
                    self.packed_exponents, # Weight tensors are reused
                    self.signs,
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

# --- Original GPTQ Sequential Logic (Renamed) ---
@torch.no_grad()
def opt_sequential_gptq(model, dataloader, quantizer_name, dev, args): # Added args parameter
    print('Starting GPTQ quantization...') # Updated print statement for clarity

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
    
    # Ensure the original GPTQ function returns the quantizers dictionary
    return quantizers
# --- End Original GPTQ Sequential Logic ---


# --- LogPack4bit Sequential Logic (Replaces original opt_sequential) ---
@torch.no_grad()
def opt_sequential(model, dataloader, quantizer_name, dev): # quantizer_name is less relevant here, but kept for signature
    print('Starting LogPack4bit quantization...')

    # Ensure global args is accessible or pass it in if needed
    # Assuming args is accessible globally as defined in if __name__ == '__main__'
    global args

    use_cache = model.config.use_cache
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
             if m.in_features % 2 == 0:
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
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    # Run calibration samples
    for i in range(args.nsamples):
         # Assuming dataloader is a list of batches
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
         act_max_abs = torch.max(torch.abs(act_tensor.view(-1, act_tensor.shape[-1])), dim=0)[0]
         layer_max_abs = torch.max(act_max_abs)
         act_scales[name] = (layer_max_abs / q_max_act).clamp(min=1e-9) # Store scale on CPU

    del act_dict
    print("Activation statistics collected.")
    # --- End Activation Calibration ---


    # --- Log Quantization and Layer Replacement ---
    print("\nApplying Logarithmic Quantization and Packing...")
    if quantizer_name != 'logarithm' or args.wbits != 4:
         print("Warning: LogPack4bit flow selected but quantizer/wbits mismatch. Ensure --quantizer logarithm --wbits 4")

    log_quantizer = LogQuantizer()
    log_quantizer.configure(bits=args.wbits)

    layers_replaced_count = 0
    # Iterate through layers again for replacement (ensure layers is correct reference)
    layers = model.model.decoder.layers
    for i in range(len(layers)):
        print(f"Processing layer {i}...")
        layer = layers[i] # Keep on CPU
        layer_modules = find_layers(layer)

        for name, lin_layer in layer_modules.items():
            if not isinstance(lin_layer, nn.Linear): continue
            if lin_layer.in_features % 2 != 0:
                print(f"  Skipping {name}: Odd in_features ({lin_layer.in_features}).")
                continue

            full_name_found = None
            for model_name, mod in model.named_modules():
                 if mod is lin_layer:
                      full_name_found = model_name
                      break

            if full_name_found is None or full_name_found not in act_scales:
                 print(f"  Warning: Activation scale not found for {name} (lookup name: {full_name_found}). Skipping.")
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

            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            sub_layer_name = name.rsplit('.', 1)[-1]
            if parent_name:
                 parent_module = layer.get_submodule(parent_name)
            else:
                 parent_module = layer
            setattr(parent_module, sub_layer_name, custom_layer)
            layers_replaced_count += 1

        if i % 5 == 0: torch.cuda.empty_cache()

    print(f"Logarithmic quantization finished. Replaced {layers_replaced_count} layers.")
    model.config.use_cache = use_cache
    return {}
# --- End LogPack4bit Sequential Logic ---


# --- Evaluation Function ---
@torch.no_grad()
def opt_eval(model, testenc, dev): # Use the single 'dev' parameter now
    print('Evaluating ...')

    # Force model to the designated evaluation device, overriding any multi-GPU setup
    print(f"  Moving model to evaluation device: {dev}")
    model.to(dev)
    # Remove the gpus attribute if it exists from benchmarking to prevent confusion
    if hasattr(model, 'gpus'):
        delattr(model, 'gpus')
        # Unwrap layers from MoveModule if necessary (model.to(dev) might handle this, but explicit is safer)
        # Use try-except as MoveModule might not be defined if benchmark wasn't run multi-gpu
        try:
            from gptq import MoveModule # Import locally for isinstance check
            layers = model.model.decoder.layers
            for i in range(len(layers)):
                if isinstance(layers[i], MoveModule):
                     layers[i] = layers[i].module
        except (NameError, ImportError): # Handle if MoveModule isn't defined/imported
            pass
        except AttributeError: # Handle cases where model structure might differ
             print("  Warning: Could not access layers to unwrap MoveModule.")
             pass


    testenc = testenc.input_ids # Assuming testloader passed is actually testenc from get_loaders
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    # If evaluating on multi-GPU, the devices are set by opt_multigpu.
    # If evaluating on single GPU, model.to(input_device) handles placement.
    # The explicit moves below are only needed if NOT using model.to(dev) in the single-GPU case,
    # but they conflict with the multi-GPU case.
    # Let's remove them as they are either redundant or incorrect for multi-GPU.

    # model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(input_device) # Handled by model.to or opt_multigpu
    # model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(input_device) # Handled by model.to or opt_multigpu
    # if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
    #     model.model.decoder.project_in = model.model.decoder.project_in.to(input_device) # Handled by model.to or opt_multigpu
    # if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
    #     model.model.decoder.project_out = model.model.decoder.project_out.to(output_device) # Handled by opt_multigpu
    # if model.model.decoder.final_layer_norm is not None:
    #     model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(output_device) # Handled by opt_multigpu
    # model.lm_head = model.lm_head.to(output_device) # Handled by opt_multigpu

    # Ensure necessary components are on the correct devices if multi-GPU (Handled by opt_multigpu)
    # Embeddings should be on input_device, final_ln/lm_head on output_device (Handled by opt_multigpu)
    # This should already be handled by opt_multigpu if it was called.
    # If evaluating a model *not* processed by opt_multigpu, model.to(dev) handles it.

    # No need to prepare the large 'inps' buffer for evaluation

    nlls = []
    loss_fct = nn.CrossEntropyLoss()
    for i in range(nsamples):
        # Move batch to the single evaluation device
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        # Standard forward pass for evaluation (model is now entirely on 'dev')
        outputs = model(batch)
        lm_logits = outputs.logits # Logits will be on 'dev'

        # Shift logits and labels for next token prediction loss
        shift_logits = lm_logits[:, :-1, :].contiguous()
        # Labels are already on 'dev' from batch assignment
        shift_labels = batch[:, 1:]

        # Calculate loss on 'dev'
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen # Use actual sequence length if different
        nlls.append(neg_log_likelihood)
        # Optional: Clear cache between samples if memory is very tight
        # torch.cuda.empty_cache()

    # Stack NLLs (on 'dev') and calculate PPL
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.4f}")

    model.config.use_cache = use_cache # Restore use_cache setting

    # Move model back to CPU
    model.cpu()
    # Explicitly clear GPU memory after moving model to CPU
    torch.cuda.empty_cache()

    return ppl.item()
# --- End Evaluation Function ---


# TODO: perform packing on GPU
def opt_pack3(model, quantizers):
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
        help='#bits to use for quantization; use 16 for evaluating base model. Use 4 for logpack.'
    )
    # Commenting out GPTQ specific args - enable if needed for GPTQ mode
    # parser.add_argument(
    #     '--trits', action='store_true',
    #     help='Whether to use trits for quantization.'
    # )
    # parser.add_argument(
    #     '--groupsize', type=int, default=-1,
    # #     '--groupsize', type=int, default=-1, # This line seems duplicated, commenting out both
    #     help='Groupsize to use for quantization; default uses full row.'
    # )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization (Currently affects GPTQ/RTN, not LogPack activation quant).'
    )
    # parser.add_argument(
    #     '--save', type=str, default='',
    #     help='Save quantized checkpoint under this name.'
    # )
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
        help="Which parameter quantizer to use. 'logarithm' triggers LogPack4bit flow if wbits=4.",
    )
    # Commenting out GPTQ specific args
    # parser.add_argument(
    #     '--log-error-scale-power', type=float, default=0.0,
    #     help='Power p for scaling log quant error: err_scaled = err * (|q|+eps)^(-p). Default 0.0 (no scaling).'
    # )
    # parser.add_argument(
    #     '--act-order', action='store_true',
    #     help='Whether to apply the activation order GPTQ heuristic'
    # )
    # parser.add_argument(
    #     '--static-groups', action='store_true',
    #     help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    # )
    parser.add_argument(
        '--quant_mode', type=str, default='gptq', choices=['gptq', 'rtn', 'logpack4bit'],
        help="Select quantization mode: gptq, rtn (nearest), or logpack4bit"
    )


    args = parser.parse_args()

    # --- Select Quantization Mode ---
    is_logpack_mode = (args.quant_mode == 'logpack4bit')
    is_rtn_mode = (args.quant_mode == 'rtn' or args.nearest) # Allow --nearest for backward compat
    is_gptq_mode = (args.quant_mode == 'gptq' and not is_rtn_mode and not is_logpack_mode)

    if is_logpack_mode and args.wbits != 4:
         print("Warning: --quant_mode logpack4bit requires --wbits 4. Setting wbits=4.")
         args.wbits = 4
    if is_logpack_mode:
         args.quantizer = 'logarithm' # Ensure correct quantizer name for logpack mode

    # --- Load Model ---
    if args.load:
        # TODO: Need a specific loading function for LogPack4bit models if saving is implemented
        if is_logpack_mode:
             raise NotImplementedError("Loading saved LogPack4bit models not yet implemented.")
        else:
             # Assuming load_quant3 is for GPTQ 3bit - might need adjustment for other GPTQ bits
             print("Loading GPTQ quantized model...")
             model = load_quant3(args.model, args.load)
    else:
        model = get_opt(args.model)
        model.eval()

    # --- Get Data ---
    # Ensure dataloader returns indexable data for calibration
    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    # Convert dataloader to a list for easier indexing during calibration
    # This loads all calibration data into memory - adjust if nsamples is very large
    print("Loading calibration data into memory...")
    dataloader = [batch for batch in dataloader]
    print("Calibration data loaded.")


    # --- Apply Quantization ---
    if args.wbits < 16 and not is_rtn_mode: # Apply GPTQ or LogPack
        tick = time.time()
        if is_gptq_mode:
             print("Applying GPTQ quantization...")
             # Pass necessary GPTQ args here
             # Ensure opt_sequential_gptq exists and accepts args
             quantizers = opt_sequential_gptq(model, dataloader, args.quantizer, DEV, args)
        elif is_logpack_mode:
             print("Applying LogPack4bit quantization...")
             # opt_sequential is now the LogPack function
             quantizers = opt_sequential(model, dataloader, args.quantizer, DEV)
        else:
             # This case should not be reached due to arg choices, but included for safety
             print("Error: Invalid quantization mode selected for wbits < 16.")
             exit(1)
        print(f"Quantization time: {time.time() - tick:.2f}s")
    elif is_rtn_mode:
         print("Applying RTN (nearest neighbor) quantization...")
         # RTN logic is typically applied during evaluation in the original script
         pass # No separate quantization step needed before eval/benchmark for RTN
    else:
        print("Running FP16 model (wbits=16).")


    # --- Benchmarking ---
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
        if is_logpack_mode:
            print("Saving LogPack4bit model not implemented.")
            # TODO: Implement saving logic for LogPack4bit model state_dict
            # This would involve saving the packed buffers and parameters from LogMatVecPackedLinear layers
        elif is_gptq_mode:
            # Assuming opt_pack3 is for GPTQ 3bit
            opt_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save)
        else:
            print("Saving only supported for GPTQ mode currently.")
