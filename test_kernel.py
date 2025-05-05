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
    # Import the new 4-bit layer and helper
    from quant.minmaxquant import Quant4Linear, make_quant4
    _quant_layers_available = True
except ImportError as e:
    print(f"Could not import quantization layers: {e}")
    _quant_layers_available = False

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
    mat_int4_packed = torch.randint(0, 2**32 - 1, (packed_height_4bit, N_BENCH), device=DEV, dtype=torch.int32) # Use int32
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
    print(f'4-bit MatVec Time (FP32): {(time.time() - tick) / COUNT:.6f} seconds')

    # Example using FP16 input/output
    mul_fp16_4b = torch.zeros((1, N_BENCH), device=DEV, dtype=DTYPE_FP16)
    scales_4bit_h = scales_4bit.half()
    zeros_4bit_h = zeros_4bit.half()
    tick = time.time()
    for _ in range(COUNT):
        quant_cuda_4bit.vecquant4matmul(vec_fp16_4b, mat_int4_packed, mul_fp16_4b, scales_4bit_h, zeros_4bit_h)
        torch.cuda.synchronize()
    print(f'4-bit MatVec Time (FP16): {(time.time() - tick) / COUNT:.6f} seconds')


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
if _quant_layers_available and _quant_cuda_4bit_available:
    print('\nVerifying 4-bit kernel correctness ...')

    # --- Verification Dimensions ---
    # Use smaller dimensions for verification if needed
    M_VERIF = 4096 # Input features
    N_VERIF = 4096 # Output features

    # Ensure M_VERIF is divisible by 8 for packing/kernel
    if M_VERIF % 8 != 0:
        M_VERIF = (M_VERIF // 8) * 8
        print(f"Adjusting verification M to {M_VERIF} (must be divisible by 8)")

    # Create a standard float linear layer
    layer = nn.Linear(M_VERIF, N_VERIF, bias=False) # Bias=False simplifies verification slightly
    layer = layer.to(DEV).to(torch.float) # Use float for quantization process
    vec = torch.randn(M_VERIF, device=DEV, dtype=torch.float) # Input vector

    # --- Quantize to 4-bit ---
    # Use the Quantizer from minmaxquant (or your chosen quantizer)
    quantizer_4bit = Quantizer()
    # Configure for 4 bits, per-channel, asymmetric
    quantizer_4bit.configure(4, perchannel=True, sym=False, mse=False)
    # Find quantization parameters (scale, integer zero point)
    quantizer_4bit.find_params(layer.weight.data, weight=True)

    # --- Create Simulated Quantized Layer (for comparison) ---
    # Apply quantization in float to simulate the operation
    # Q_sim(w) = (clamp(round(w / scale + zero_int), 0, 15) - zero_int) * scale
    # Need the quantize function, assuming it's available from the import
    from quant.minmaxquant import quantize # Make sure quantize is imported or defined
    weight_q_sim = quantize(
        layer.weight.data, quantizer_4bit.scale, quantizer_4bit.zero, quantizer_4bit.maxq
    )
    layer_sim = nn.Linear(M_VERIF, N_VERIF, bias=False)
    layer_sim.weight.data = weight_q_sim
    layer_sim = layer_sim.to(DEV)

    # --- Create Quant4Linear Layer ---
    qlayer_4bit = Quant4Linear(layer.in_features, layer.out_features, faster=False) # Test standard kernel first
    # Pack the original layer's weights using the found scales and integer zero points
    # The pack method will calculate zero_point * scale internally
    qlayer_4bit.pack(layer, quantizer_4bit.scale, quantizer_4bit.zero)
    qlayer_4bit = qlayer_4bit.to(DEV)

    # --- Create Quant4Linear Layer (Faster version) ---
    # Note: Faster kernel needs FP16 input. We'll cast the input vector later.
    qlayer_4bit_faster = Quant4Linear(layer.in_features, layer.out_features, faster=True)
    qlayer_4bit_faster.pack(layer, quantizer_4bit.scale, quantizer_4bit.zero)
    qlayer_4bit_faster = qlayer_4bit_faster.to(DEV)


    # --- Run Verification ---
    with torch.no_grad():
        # 1. Simulated quantization output (Float)
        out_sim = layer_sim(vec)
        print(f'\nSimulated (Float): {out_sim.abs().mean().item():.5f} (mean abs val)')

        # 2. Kernel output (Standard, Float input -> Float output)
        out_kern = qlayer_4bit(vec)
        print(f'Kernel Std (F32):  {out_kern.abs().mean().item():.5f} (mean abs val)')
        print(f'--> Diff (Sim vs Kern Std): {(out_sim - out_kern).abs().mean().item():.8f}')

        # 3. Kernel output (Standard, Half input -> Half output)
        #    Requires casting input and potentially adjusting layer params if needed
        qlayer_4bit_fp16 = Quant4Linear(layer.in_features, layer.out_features, faster=False)
        qlayer_4bit_fp16.pack(layer, quantizer_4bit.scale.half(), quantizer_4bit.zero.half()) # Use half params
        qlayer_4bit_fp16 = qlayer_4bit_fp16.to(DEV)
        out_kern_fp16 = qlayer_4bit_fp16(vec.half())
        print(f'Kernel Std (F16):  {out_kern_fp16.abs().mean().item():.5f} (mean abs val)')
        print(f'--> Diff (Sim vs Kern F16): {(out_sim.half() - out_kern_fp16).abs().mean().item():.8f}')


        # 4. Kernel output (Faster, Half input -> Float output)
        if qlayer_4bit_faster.faster: # Check if faster kernel was actually enabled
             out_kern_faster = qlayer_4bit_faster(vec.half()) # Input must be half
             print(f'Kernel Fst (F16->F32): {out_kern_faster.abs().mean().item():.5f} (mean abs val)')
             print(f'--> Diff (Sim vs Kern Fst): {(out_sim - out_kern_faster).abs().mean().item():.8f}')
        else:
             print("Skipping faster kernel verification (not enabled or available).")

else:
    print("\nSkipping verification (Quantization layers or 4-bit CUDA module not found).")
