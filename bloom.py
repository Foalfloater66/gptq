import argparse
import math
import time
import json # <-- Add import
import os   # <-- Add import

import torch
import torch.nn as nn
import transformers

from gptq import *
from modelutils import *
# Remove specific Quant3Linear import if bloom_pack3 is removed or adapted later
from quant.minmaxquant import Quant3Linear, make_quant3
from quant import get_quantizer # <-- Use generic getter
from datautils import * # <-- Add datautils import


def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

# Define DEV globally or pass args around if needed
# DEV = torch.device('cuda:0') # Set in main block

# Add conditional printing based on --quiet flag
def log_print(*print_args, **kwargs):
    # This function will be defined within the main block
    # based on the args.quiet flag.
    # For now, assume it exists.
    # A placeholder implementation:
    # if not args.quiet:
    #    print(*print_args, **kwargs)
    pass # Actual definition will be in __main__

@torch.no_grad()
def bloom_sequential(model, dataloader, quantizer_name, dev, args): # Add args, quantizer_name
    log_print('Starting Quantization ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    log_print('Quantization setup ready.')

    quantizer_instance = get_quantizer(quantizer_name) # Get the quantizer class
    quantizers = {}
    for i in range(len(layers)):
        log_print(f"Quantizing layer {i+1}/{len(layers)}...")
        layer = layers[i].to(dev)
        full = find_layers(layer)

        # Filter layers; keep only Linear for BLOOM's main blocks
        # Adapt this filtering if other layer types need quantization
        subset = {name: full[name] for name in full if isinstance(full[name], nn.Linear)}

        gptq = {}
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            gptq[name].quantizer = quantizer_instance() # Instantiate the quantizer
            # Configure using args - adapt options as needed for different quantizers
            # Assuming common interface; specific quantizers might need more args
            gptq[name].quantizer.configure(
                 args.wbits, perchannel=True, sym=args.sym, mse=False # mse=False is typical for GPTQ
                 # Add other relevant args like trits=args.trits if applicable
            )

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        for h in handles:
            h.remove()

        for name in subset:
            log_print(f"Layer {i} - {name}: Quantizing...")
            # Add actorder, static_groups if supported and desired
            gptq[name].fasterquant(
                percdamp=args.percdamp, groupsize=args.groupsize #, actorder=args.act_order, static_groups=args.static_groups
            )
            quantizers['transformer.h.%d.%s' % (i, name)] = gptq[name].quantizer
            gptq[name].free() # Free buffers after quantization

        # Run layer again with quantized weights to get outputs for next layer
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def bloom_eval(model, testenc, dev, args): # Add args
    log_print('Evaluating perplexity...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    log_print(f"Using {nsamples} samples for evaluation.")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
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
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    for i in range(len(layers)):
        log_print(f"Evaluating layer {i+1}/{len(layers)}...") # Use log_print
        layer = layers[i].to(dev)

        # Removed --nearest block for simplicity, focusing on GPTQ eval
        # If RTN (--nearest) is needed, it should be implemented here,
        # potentially using a standard quantizer like MinMaxQuantizer.

        # Run layer with potentially quantized weights
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
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
    # print(ppl.item()) # Don't print here, return it

    model.config.use_cache = use_cache
    return ppl.item() # Return perplexity


def bloom_pack3(model, quantizers):
    """
    Packs weights for 3-bit quantization.
    Note: This is specific to Quant3Linear and might need adaptation
          if other quantizers or bit-widths require packing.
    """
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


# --- Main Execution Block ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
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
        help='Whether to run the RTN baseline.' # Note: RTN logic removed from bloom_eval for now
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16], # Adjust choices as needed
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    # Add arguments similar to opt.py
    parser.add_argument(
        '--trits', action='store_true',
        help='Whether to use trits for quantization (specific quantizers).'
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
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    # Arguments for specific quantizers / GPTQ options
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups (relevant for act-order).'
    )
    # Quantizer selection
    parser.add_argument(
        '--quantizer', type=str, choices=['uniform_minmax', 'logarithm', 'quantile', 'kmeans', 'apot', 'lloydmax'], default='uniform_minmax',
        help="Which parameter quantizer to use.",
    )
    # Output and verbosity control
    parser.add_argument(
        '--output-file', type=str, default=None,
        help='Path to save evaluation results (JSON Lines format).'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Reduce verbose logging during quantization and evaluation.'
    )


    args = parser.parse_args()

    # Define DEV globally or pass args around if needed
    DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Assumes CUDA_VISIBLE_DEVICES handles mapping

    # --- Setup Logging ---
    # Define log_print based on the --quiet flag
    if args.quiet:
        def log_print(*print_args, **kwargs):
            pass # No output if quiet
    else:
        # Assign global print function to log_print
        import builtins
        log_print = builtins.print

    # --- Load Model ---
    # TODO: Add load_quant functionality if needed, similar to opt.py's load_quant3
    # if args.load:
    #     model = load_quant_bloom(args.model, args.load) # Needs implementation
    # else:
    model = get_bloom(args.model)
    model.eval()
    log_print("Model loaded successfully.")

    # --- Load Calibration Data ---
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )
    log_print(f"Calibration data loaded from: {args.dataset}")

    # --- Quantization ---
    quantizers = None # Initialize quantizers
    if args.wbits < 16: # and not args.nearest: # Removed nearest check
        log_print(f"Starting quantization ({args.quantizer}, {args.wbits}-bit)...")
        tick = time.time()
        # Pass args to sequential function
        quantizers = bloom_sequential(model, dataloader, args.quantizer, DEV, args)
        log_print(f"Quantization finished in {time.time() - tick:.2f}s")
    else:
        log_print("Skipping quantization (wbits >= 16).")


    # --- Evaluation ---
    evaluation_results = {}
    datasets = ['wikitext2', 'ptb', 'c4']
    if args.new_eval:
      datasets = ['wikitext2', 'ptb-new', 'c4-new']

    log_print(f"\nStarting evaluation on datasets: {', '.join(datasets)}")
    for dataset in datasets:
        _, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        log_print(f"--- Evaluating on: {dataset} ---")
        # Pass args to eval function
        ppl = bloom_eval(model, testloader, DEV, args)
        log_print(f"Perplexity ({dataset}): {ppl:.4f}")
        evaluation_results[dataset] = ppl
        torch.cuda.empty_cache() # Clear cache between dataset evaluations

    log_print("\nEvaluation complete.")

    # --- Save Results ---
    if args.output_file:
        output_data = {
            'model': args.model,
            'quantizer': args.quantizer,
            'wbits': args.wbits,
            'groupsize': args.groupsize,
            'sym': args.sym,
            'percdamp': args.percdamp,
            # Add other relevant args like act_order, static_groups, trits if used
            # 'act_order': args.act_order,
            # 'static_groups': args.static_groups,
            # 'trits': args.trits,
            'results': evaluation_results
        }
        # Append results as a new line in JSON Lines format
        mode = 'a' if os.path.exists(args.output_file) else 'w'
        try:
            with open(args.output_file, mode) as f:
                f.write(json.dumps(output_data) + '\n')
            log_print(f"Results appended to {args.output_file}")
        except IOError as e:
            log_print(f"Error writing to output file {args.output_file}: {e}")

    # --- Save Model ---
    # Note: bloom_pack3 is specific to 3-bit. For general saving,
    # just saving the state_dict might be sufficient, but it won't be packed.
    # If packing is needed for other quantizers/bits, implement specific packing logic.
    if args.save:
        if quantizers is not None and args.wbits == 3: # Only call pack3 for 3-bit
             log_print("Packing model for 3-bit quantization...")
             bloom_pack3(model, quantizers)
             torch.save(model.state_dict(), args.save)
             log_print(f"Packed 3-bit model saved to {args.save}")
        elif quantizers is not None:
             # For other bit-widths, save the state dict directly (unpacked)
             torch.save(model.state_dict(), args.save)
             log_print(f"Quantized model state_dict saved to {args.save} (unpackaged)")
        else:
             log_print("Skipping model save (no quantization performed or --save not specified).")
