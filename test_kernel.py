import torch
import torch.nn as nn
import time # Move import up

# Import the 3-bit CUDA module (keep for comparison if desired)
try:
    import quant_cuda
    _quant_cuda_3bit_available = True
except ImportError:
    print("CUDA 3-bit kernel not found.")
    _quant_cuda_3bit_available = False

# Import the NEW 4-bit CUDA module
try:
    import quant_cuda_4bit
    _quant_cuda_4bit_available = True
    print("CUDA 4-bit kernel found.")
except ImportError:
    print("CUDA 4-bit kernel not found.")
    _quant_cuda_4bit_available = False


# Import Quantizer and the NEW Quant4Linear layer and make_quant4 function
# Assuming they are in quant.minmaxquant or a similar structure
# Adjust the import path if you placed them elsewhere (e.g., quant.quant4linear)
try:
    # Assuming Quantizer is needed for the verification part
    from quant.minmaxquant import Quantizer, Quant3Linear # Keep Quant3Linear if comparing
    # Import the new 4-bit layer and helper from its new location
    from quant.quant4linear import Quant4Linear, make_quant4
    _quant_layers_available = True
except ImportError as e:
    print(f"Could not import quantization layers: {e}")
    _quant_layers_available = False

# Import other quantizers for verification loop
try:
    from quant import get_quantizer, QuantileQuantizer, LloydMaxQuantizer, LogQuantizer, KMeansQuantizer, APoTQuantizer
    # MinMaxQuantizer is already imported as Quantizer
    _other_quantizers_available = True
    print("Successfully imported other quantizer types for verification.")
except ImportError as e:
    print(f"Could not import other quantizer types: {e}")
    _other_quantizers_available = False


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking OPT-175B FC2 matvec ...')

DEV = torch.device('cuda:0')

# --- Benchmark Dimensions ---
# Using dimensions similar to the original script for OPT-175B's FFN layer
M_BENCH = 12288 * 4 # Input features (or hidden_size * 4)
N_BENCH = 12288     # Output features (or hidden_size)
COUNT = 1000        # Number of iterations

# --- FP16 Benchmark ---
print("\n--- FP16 Benchmark ---")
DTYPE_FP16 = torch.half
mat_fp16 = torch.randn((M_BENCH, N_BENCH), device=DEV, dtype=DTYPE_FP16)
vec_fp16 = torch.randn((1, M_BENCH), device=DEV, dtype=DTYPE_FP16)
mul_fp16 = torch.zeros((1, N_BENCH), device=DEV, dtype=DTYPE_FP16)

tick = time.time()
for _ in range(COUNT):
    torch.matmul(vec_fp16, mat_fp16, out=mul_fp16)
    torch.cuda.synchronize()
print(f'FP16 MatVec Time: {(time.time() - tick) / COUNT:.6f} seconds')

# --- 3-bit Benchmark (Optional - Keep for comparison) ---
if _quant_cuda_3bit_available:
    print("\n--- 3-bit Benchmark ---")
    # Prepare data for 3-bit kernel
    # Note: The original script used float for scales/zeros, let's stick to that
    DTYPE_FLOAT = torch.float
    vec_f32 = vec_fp16.float() # Use float input vector for 3-bit kernel
    mul_f32 = torch.zeros((1, N_BENCH), device=DEV, dtype=DTYPE_FLOAT)

    # Packed matrix dimensions for 3-bit (3 bits per value, 32 bits per int)
    # Each int stores 32/3 = 10 values + 2 bits padding. Requires 3 ints for 32 values.
    # Packed height = M_BENCH / 32 * 3
    packed_height_3bit = M_BENCH // 32 * 3
    mat_int3_packed = torch.randint(-1000000000, 1000000000, (packed_height_3bit, N_BENCH), device=DEV, dtype=torch.int)
    scales_3bit = torch.randn(N_BENCH, device=DEV, dtype=DTYPE_FLOAT)
    # Original kernel expects zero_point * scale
    zeros_3bit = torch.randn(N_BENCH, device=DEV, dtype=DTYPE_FLOAT)

    # Benchmark standard 3-bit kernel
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda.vecquant3matmul(vec_f32, mat_int3_packed, mul_f32, scales_3bit, zeros_3bit)
        torch.cuda.synchronize()
    print(f'3-bit MatVec Time: {(time.time() - tick) / COUNT:.6f} seconds')

    # Benchmark faster 3-bit kernel (requires FP16 input)
    mul_f32.zero_() # Reset output buffer
    tick = time.time()
    for _ in range(COUNT):
        # Faster kernel expects FP16 input, FP32 output
        quant_cuda.vecquant3matmul_faster(vec_fp16, mat_int3_packed, mul_f32, scales_3bit, zeros_3bit)
        torch.cuda.synchronize()
    print(f'3-bit MatVec Time (faster): {(time.time() - tick) / COUNT:.6f} seconds')
else:
    print("\nSkipping 3-bit benchmark (CUDA module not found).")


# --- 4-bit Benchmark ---
if _quant_cuda_4bit_available:
    print("\n--- 4-bit Benchmark ---")
    # Prepare data for 4-bit kernel
    DTYPE_FLOAT = torch.float
    vec_f32 = vec_fp16.float() # Use float input vector for standard 4-bit kernel
    mul_f32 = torch.zeros((1, N_BENCH), device=DEV, dtype=DTYPE_FLOAT) # Output for standard kernel

    # Packed matrix dimensions for 4-bit (4 bits per value, 32 bits per int)
    # Each int stores 32/4 = 8 values.
    # Packed height = M_BENCH / 8
    # Ensure M_BENCH is divisible by 8 for simplicity in benchmark setup
    if M_BENCH % 8 != 0:
         print(f"Warning: M_BENCH ({M_BENCH}) not divisible by 8. Adjusting for benchmark.")
         M_BENCH_4BIT = (M_BENCH // 8) * 8
         vec_fp16_4b = vec_fp16[:, :M_BENCH_4BIT] # Adjust vector size
         vec_f32_4b = vec_f32[:, :M_BENCH_4BIT]
    else:
         M_BENCH_4BIT = M_BENCH
         vec_fp16_4b = vec_fp16
         vec_f32_4b = vec_f32

    packed_height_4bit = M_BENCH_4BIT // 8
    # Generate random signed int32 values within the valid range
    # Max value for int32 is 2**31 - 1. randint high is exclusive, so use 2**31.
    # Min value is -(2**31).
    min_int32 = -(2**31)
    max_int32_exclusive = 2**31
    mat_int4_packed = torch.randint(min_int32, max_int32_exclusive, (packed_height_4bit, N_BENCH), device=DEV, dtype=torch.int32)
    scales_4bit = torch.randn(N_BENCH, device=DEV, dtype=DTYPE_FLOAT)
    # Kernel expects zero_point * scale
    zeros_4bit = torch.randn(N_BENCH, device=DEV, dtype=DTYPE_FLOAT) # Represents zero_point * scale

    # Benchmark standard 4-bit kernel
    # Can use FP16 or FP32 input/output (matching types, except FP16 in -> FP16 out)
    tick = time.time()
    for _ in range(COUNT):
        # Example using FP32 input/output
        quant_cuda_4bit.vecquant4matmul(vec_f32_4b, mat_int4_packed, mul_f32, scales_4bit, zeros_4bit)
        torch.cuda.synchronize()
    print(f'4-bit MatVec Time (FP32 in -> FP32 out): {(time.time() - tick) / COUNT:.6f} seconds')

    # Example using FP16 input -> FP32 output (standard kernel)
    mul_f32.zero_() # Use the FP32 output buffer
    scales_4bit_h = scales_4bit.half() # Kernel expects half scales/zeros for half input
    zeros_4bit_h = zeros_4bit.half()
    tick = time.time()
    for _ in range(COUNT):
        # Input vec is FP16, scales/zeros are FP16, output mul is FP32
        quant_cuda_4bit.vecquant4matmul(vec_fp16_4b, mat_int4_packed, mul_f32, scales_4bit_h, zeros_4bit_h)
        torch.cuda.synchronize()
    print(f'4-bit MatVec Time (FP16 in -> FP32 out): {(time.time() - tick) / COUNT:.6f} seconds')


    # Benchmark faster 4-bit kernel (requires FP16 input, FP32 output)
    mul_f32.zero_() # Reset output buffer
    tick = time.time()
    for _ in range(COUNT):
        # Faster kernel expects FP16 input, FP32 output, FP32 scales/zeros
        quant_cuda_4bit.vecquant4matmul_faster(vec_fp16_4b, mat_int4_packed, mul_f32, scales_4bit, zeros_4bit)
        torch.cuda.synchronize()
    print(f'4-bit MatVec Time (faster, FP16->FP32): {(time.time() - tick) / COUNT:.6f} seconds')
else:
    print("\nSkipping 4-bit benchmark (CUDA module not found).")


# --- Verification Section ---
# Note: Benchmarking section remains unchanged as kernel speed isn't expected
#       to significantly vary based on how scales/zeros were derived.

if _quant_layers_available and _quant_cuda_4bit_available and _other_quantizers_available:
    print('\nVerifying 4-bit kernel correctness with different quantization methods...')

    # --- Verification Dimensions ---
    # Use smaller dimensions for verification if needed
    M_VERIF = 1024 # Input features - Use smaller dimensions for faster verification loop
    N_VERIF = 1024 # Output features

    # Ensure M_VERIF is divisible by 8 for packing/kernel
    if M_VERIF % 8 != 0:
        M_VERIF = (M_VERIF // 8) * 8
        print(f"Adjusting verification M to {M_VERIF} (must be divisible by 8)")

    # Create a standard float linear layer
    layer_orig = nn.Linear(M_VERIF, N_VERIF, bias=False) # Bias=False simplifies verification slightly
    layer_orig = layer_orig.to(DEV).to(torch.float) # Use float for quantization process
    vec = torch.randn(M_VERIF, device=DEV, dtype=torch.float) # Input vector

    # --- Loop through different quantizers ---
    # Note: MinMaxQuantizer is imported as 'Quantizer'
    quantizer_types = {
        "MinMax": Quantizer,
        "KMeans": KMeansQuantizer,
        "Log": LogQuantizer,
        "APoT": APoTQuantizer,
        "Quantile": QuantileQuantizer,
        "LloydMax": LloydMaxQuantizer
    }

    for name, QuantizerClass in quantizer_types.items():
        print(f"\n--- Verifying with {name} Quantizer ---")
        layer = nn.Linear(M_VERIF, N_VERIF, bias=False).to(DEV)
        layer.weight.data.copy_(layer_orig.weight.data) # Start with fresh weights

        # 1. Instantiate and configure the specific quantizer
        quantizer_specific = QuantizerClass()
        # Configure for 4 bits. Add specific kwargs if needed (e.g., for LloydMax)
        print(f"Configuring {name} quantizer...")
        if name == "LloydMax":
            quantizer_specific.configure(bits=4, max_iterations=10) # Pass max_iterations only for LloydMax
        else:
            # Most other quantizers likely only need 'bits'
            # Add specific checks here if others need different args
            quantizer_specific.configure(bits=4)
        print("Finding specific quantization parameters...")
        quantizer_specific.find_params(layer.weight.data, weight=True)

        if not quantizer_specific.ready():
            print(f"Skipping {name} as quantizer is not ready after find_params.")
            continue

        # 2. Quantize weights using the specific quantizer's logic (for reference)
        #    This step isn't strictly needed for packing but useful for understanding
        with torch.no_grad():
             quantized_weights_specific = quantizer_specific.quantize(layer.weight.data)

        # 3. Find *Affine* parameters (scale, zero) for the `quantized_weights_specific`
        #    We use MinMaxQuantizer (Quantizer) to find the best affine fit
        #    to the weights already quantized by the specific method.
        print("Finding *affine* parameters for the specifically quantized weights...")
        affine_quantizer = Quantizer()
        affine_quantizer.configure(bits=4, perchannel=True, sym=False, mse=False)
        # Find scale/zero for the weights *already quantized* by KMeans/Log/etc.
        affine_quantizer.find_params(quantized_weights_specific.detach(), weight=True)

        if not affine_quantizer.ready():
            print(f"Skipping {name} as affine quantizer is not ready.")
            continue

        affine_scale = affine_quantizer.scale
        affine_zero = affine_quantizer.zero # This is the integer zero point

        # --- Create Simulated *Affine* Quantized Layer (for comparison) ---
        # We simulate using the affine parameters found in step 3, applied to *original* weights
        # This shows what the ideal affine quantization (using these params) would look like.
        from quant.minmaxquant import quantize # Make sure quantize is imported or defined
        # Use original layer weights and the *affine* scale/zero derived from the specific quantizer's output
        weight_q_sim = quantize(
            layer.weight.data, affine_scale, affine_zero, affine_quantizer.maxq
        )
        layer_sim = nn.Linear(M_VERIF, N_VERIF, bias=False).to(DEV)
        layer_sim.weight.data = weight_q_sim.to(layer_sim.weight.dtype)


        # --- Create Quant4Linear Layer (using the derived affine parameters) ---
        print("Packing weights into Quant4Linear using derived affine parameters...")
        qlayer_4bit = Quant4Linear(layer.in_features, layer.out_features, faster=False)
        # Pack the *original* layer's weights using the *affine* scales and *affine* integer zero points
        qlayer_4bit.pack(layer, affine_scale, affine_zero)
        qlayer_4bit = qlayer_4bit.to(DEV)

        # --- Create Quant4Linear Layer (Faster version) ---
        qlayer_4bit_faster = Quant4Linear(layer.in_features, layer.out_features, faster=True)
        qlayer_4bit_faster.pack(layer, affine_scale, affine_zero)
        qlayer_4bit_faster = qlayer_4bit_faster.to(DEV)


        # --- Run Verification (Comparing Quant4Linear output to the *simulated affine* output) ---
        with torch.no_grad():
            # 1. Simulated *Affine* quantization output (Float)
            out_sim = layer_sim(vec.to(layer_sim.weight.dtype))
            print(f'Simulated Affine (Float): {out_sim.abs().mean().item():.5f} (mean abs val)')

            # 2. Kernel output (Standard, Float input -> Float output)
            out_kern = qlayer_4bit(vec)
            print(f'Kernel Std (F32 in -> F32 out): {out_kern.abs().mean().item():.5f} (mean abs val)')
            print(f'--> Diff (Sim Affine vs Kern Std F32): {(out_sim - out_kern).abs().mean().item():.8f}')

            # 3. Kernel output (Standard, Half input -> Float output)
            #    The standard kernel now always outputs float.
            #    We still use the qlayer_4bit instance packed with float params,
            #    but call it with a half input vector.
            out_kern_f16_in = qlayer_4bit(vec.half()) # Input is half, output is float
            print(f'Kernel Std (F16 in -> F32 out): {out_kern_f16_in.abs().mean().item():.5f} (mean abs val)')
            # Compare float kernel output to float simulated output
            print(f'--> Diff (Sim Affine vs Kern Std F16in): {(out_sim - out_kern_f16_in).abs().mean().item():.8f}')


            # 4. Kernel output (Faster, Half input -> Float output)
            if qlayer_4bit_faster.faster: # Check if faster kernel was actually enabled
                 out_kern_faster = qlayer_4bit_faster(vec.half()) # Input must be half
                 print(f'Kernel Fst (F16->F32):    {out_kern_faster.abs().mean().item():.5f} (mean abs val)')
                 # Compare FP32 kernel output to FP32 simulated output
                 print(f'--> Diff (Sim Affine vs Kern Fst): {(out_sim - out_kern_faster).abs().mean().item():.8f}')
            else:
                 print("Skipping faster kernel verification (not enabled or available).")

        # Clean up memory (optional, but good practice in a loop)
        del layer, quantizer_specific, quantized_weights_specific, affine_quantizer
        del layer_sim, qlayer_4bit, qlayer_4bit_faster # Removed qlayer_4bit_fp16
        torch.cuda.empty_cache()

else:
    print("\nSkipping verification (Quantization layers, 4-bit CUDA module, or other quantizers not found).")
