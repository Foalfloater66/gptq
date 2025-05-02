import torch
import torch.nn as nn
import time
import logmatvec_cuda # Import the compiled custom kernel
from quant.logquantizer import LogQuantizer # Import the LogQuantizer

# --- Configuration ---
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

print('Benchmarking Logarithmic Quantized MatVec Kernel ...')

DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if DEV == torch.device('cpu'):
    print("Warning: CUDA device not found, running on CPU (Kernel won't work)")

# Use similar dimensions as test_kernel.py for comparison
# Example: OPT-175B FC2 layer size
M_in = 12288  # Input features (vector size)
N_out = 12288 * 4 # Output features (number of rows in weight matrix)
# Note: LogMatVec expects W (N_out x M_in) and a (M_in x 1) -> output (N_out x 1)

# --- Benchmark Standard Matmul ---
print('\n--- Standard Matmul Benchmarks ---')
COUNT = 1000

# FP16 Benchmark
try:
    mat_fp16 = torch.randn((N_out, M_in), device=DEV, dtype=torch.half)
    vec_fp16 = torch.randn((1, M_in), device=DEV, dtype=torch.half) # Shape (1, M_in) for matmul
    # Output shape will be (1, N_out) for torch.matmul(vec, mat.T)
    # Or (N_out, 1) for torch.matmul(mat, vec.T)
    # Let's match the kernel's effective operation: W @ a -> (N_out, 1) output
    vec_fp16_col = vec_fp16.T # Shape (M_in, 1)
    mul_fp16 = torch.zeros((N_out, 1), device=DEV, dtype=torch.half)

    torch.cuda.synchronize()
    tick = time.time()
    for _ in range(COUNT):
        # Equivalent operation: W @ a
        torch.matmul(mat_fp16, vec_fp16_col, out=mul_fp16)
        torch.cuda.synchronize()
    fp16_time = (time.time() - tick) / COUNT
    print(f'FP16 Matmul (W[{N_out},{M_in}] @ a[{M_in},1]): {fp16_time:.6f} s')
    del mat_fp16, vec_fp16, vec_fp16_col, mul_fp16
    torch.cuda.empty_cache()
except Exception as e:
    print(f"FP16 benchmark failed (possibly insufficient VRAM or device capability): {e}")


# FP32 Benchmark
mat_fp32 = torch.randn((N_out, M_in), device=DEV, dtype=torch.float)
vec_fp32 = torch.randn((1, M_in), device=DEV, dtype=torch.float)
vec_fp32_col = vec_fp32.T # Shape (M_in, 1)
mul_fp32 = torch.zeros((N_out, 1), device=DEV, dtype=torch.float)

torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    torch.matmul(mat_fp32, vec_fp32_col, out=mul_fp32)
    torch.cuda.synchronize()
fp32_time = (time.time() - tick) / COUNT
print(f'FP32 Matmul (W[{N_out},{M_in}] @ a[{M_in},1]): {fp32_time:.6f} s')


# --- Setup LogMatVec Kernel Benchmark Data (Randomized Packed Inputs) ---
print('\n--- LogMatVec Kernel Benchmark (Randomized Packed Inputs) ---')
LOG_BITS = 4 # Must be 4 for current packing implementation
ACT_BITS = 8 # Define bitwidth for activation range

# 1. Create Random Packed Weight Components (Bundled 1+3 bit)
print(f"Generating random packed kernel inputs ({LOG_BITS}bW / {ACT_BITS}bA equivalent)...")
# Packed weights: each byte holds two 4-bit codes (0-15)
packed_in_features_rand = M_in // 2
# Generate random bytes (0-255), each representing two packed 4-bit codes
w_packed_4bit_kernel_rand_uint8 = torch.randint(0, 256, (N_out, packed_in_features_rand), device=DEV, dtype=torch.uint8).contiguous()

# 2. Create Random Quantized Activations (Matching Kernel Input Type)
# Range for 8-bit symmetric: [-128, 127]
q_min_act = -2**(ACT_BITS - 1)
q_max_act = 2**(ACT_BITS - 1) - 1
a_quant_kernel_rand = torch.randint(q_min_act, q_max_act + 1, (M_in,), device=DEV, dtype=torch.int32).contiguous()

# 3. Set Dummy Activation Scale and Min Exponent for Benchmark Loop
delta_lsb_kernel_dummy = 1.0
min_exp_dummy = -7 # Example plausible min_exp for 4-bit

print("Random packed kernel inputs generated.")

# 4. Prepare Output Tensor
output_log_rand = torch.zeros(N_out, device=DEV, dtype=torch.float32) # Kernel outputs float32

# --- Benchmark LogMatVec Kernel with Random Packed Inputs ---
print("Benchmarking custom kernel with random packed inputs...")
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    # Ensure output is zeroed or use a fresh tensor if needed
    # output_log_rand.zero_()
    # Pass the uint8 tensor viewed as int8 to match C++ expectation
    logmatvec_cuda.forward_packed4bit( # Call the updated function
        a_quant_kernel_rand,
        w_packed_4bit_kernel_rand_uint8.view(torch.int8), # Pass packed codes (viewed as int8)
        output_log_rand,
        delta_lsb_kernel_dummy,
        min_exp_dummy             # Pass dummy min_exp
    )
    torch.cuda.synchronize()
log_kernel_time_rand = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel (Bundled Packed Random {LOG_BITS}bW/{ACT_BITS}bA Equiv): {log_kernel_time_rand:.6f} s') # Updated label

if log_kernel_time_rand > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time_rand:.2f}x')
if log_kernel_time_rand > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time_rand:.2f}x')


# --- Benchmark Bundled 4-bit Kernel (Float Mul Version) ---
print("\nBenchmarking custom kernel with random packed inputs (Float Mul Version)...")
output_log_rand_fm = torch.zeros(N_out, device=DEV, dtype=torch.float32) # Separate output tensor
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    # output_log_rand_fm.zero_()
    # Call the float mul version for bundled 4-bit
    logmatvec_cuda.forward_bundled4bit_floatmul( # Call the new float mul function
        a_quant_kernel_rand,
        w_packed_4bit_kernel_rand_uint8.view(torch.int8), # Pass packed codes (viewed as int8)
        output_log_rand_fm,       # Pass output tensor
        delta_lsb_kernel_dummy,
        min_exp_dummy
    )
    torch.cuda.synchronize()
log_kernel_time_rand_fm = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel (FloatMul Bundled Random {LOG_BITS}bW/{ACT_BITS}bA Equiv): {log_kernel_time_rand_fm:.6f} s')

if log_kernel_time_rand_fm > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time_rand_fm:.2f}x')
if log_kernel_time_rand_fm > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time_rand_fm:.2f}x')
# Compare the two bundled 4-bit kernels
if log_kernel_time_rand_fm > 0 and log_kernel_time_rand > 0:
     print(f'  Speedup (BitShift vs FloatMul - Bundled 4bit): {log_kernel_time_rand_fm / log_kernel_time_rand:.2f}x')


# --- Correctness Check (Using Actual Quantized and Packed Data) ---
print('\n--- Correctness Check (Simulated vs Kernels on Quantized Bundled Packed Data) ---') # Updated title
# Use the original FP32 matrix and vector for this section
# 1. Initialize LogQuantizer
log_quantizer = LogQuantizer()
log_quantizer.configure(bits=LOG_BITS) # Must be 4

# 2. Quantize Weights (using the FP32 matrix)
print(f"Quantizing weights to {LOG_BITS}-bit Log format for check...")
log_quantizer.find_params(mat_fp32, weight=True)
# Get 4-bit packed nibbles (stored in uint8 tensor)
packed_nibbles_check = log_quantizer.quantize(mat_fp32)
print("Weight quantization done for check.")

# 3. Pack Weights
print("Packing nibbles into bytes...")
try:
    # LogQuantizer.pack now takes the nibbles and packs them
    w_packed_4bit_check = log_quantizer.pack(packed_nibbles_check)
    # Ensure they are on the correct device and contiguous
    w_packed_4bit_kernel_check = w_packed_4bit_check.to(device=DEV).contiguous() # Already int8
    min_exp_check = int(log_quantizer.min_exp.item()) # Get min_exp for kernel
    print("Weight packing done.")
except ValueError as e:
    print(f"Skipping correctness check due to packing error: {e}")
    # Set a flag or exit if check cannot proceed
    can_run_check = False
else:
    can_run_check = True


if can_run_check:
    # 4. Quantize Activations (Linear Symmetric) - using the FP32 vector
    print(f"Quantizing activations to {ACT_BITS}-bit Linear format for check...")
    a_min_check = vec_fp32_col.min()
    a_max_check = vec_fp32_col.max()
    delta_lsb_check = torch.max(torch.abs(a_min_check), torch.abs(a_max_check)) / q_max_act
    if delta_lsb_check == 0: delta_lsb_check += 1e-9 # Avoid division by zero if input is all zero
    a_quant_check = torch.round(vec_fp32_col / delta_lsb_check).clamp(q_min_act, q_max_act).to(torch.int32)
    a_quant_kernel_check = a_quant_check.squeeze().to(device=DEV).contiguous()
    delta_lsb_kernel_check = delta_lsb_check.item()
    print("Activation quantization done for check.")

    # 5. Prepare Output Tensor for Check
    output_log_check = torch.zeros(N_out, device=DEV, dtype=torch.float32)

    # 6. Run Kernel Once with Quantized Bundled Packed Data
    print("Running kernel once with quantized bundled packed data...")
    # Create a dummy bias tensor for the check
    bias_check = torch.randn(N_out, device=DEV, dtype=torch.float32) * 0.01 # Small random bias

    # Call the updated kernel signature
    logmatvec_cuda.forward_packed4bit(
        vec_fp32_col.squeeze().to(DEV), # Pass original FP32 activations
        w_packed_4bit_kernel_check,     # Pass packed codes
        bias_check,                     # Pass bias tensor
        output_log_check,               # Output tensor
        delta_lsb_kernel_check,         # Activation scale
        min_exp_check,                  # Weight min exponent
        ACT_BITS                        # Activation bits
    )
    torch.cuda.synchronize()

    # Also run the Bundled FloatMul kernel once for comparison (if needed, update its signature too)
    # For now, focus on the bitshift kernel. Commenting out FloatMul check update.
    # output_log_check_fm = torch.zeros(N_out, device=DEV, dtype=torch.float32)
    # print("Running bundled float mul kernel once with quantized packed data...")
    # logmatvec_cuda.forward_bundled4bit_floatmul( # Call new float mul function - NEEDS UPDATE
    #     vec_fp32_col.squeeze().to(DEV), # Pass original FP32 activations
    #     w_packed_4bit_kernel_check, # Pass packed codes
    #     bias_check,                 # Pass bias
    #     output_log_check_fm,
    #     delta_lsb_kernel_check,
    #     min_exp_check,
    #     ACT_BITS
    # )
    # torch.cuda.synchronize()


    # 7. Simulate the Kernel Operation with Bundled Packing/Unpacking in PyTorch
    # This simulation still represents the BIT SHIFT logic, which is useful as a reference
    print("Simulating exact kernel operation (bit shift logic) in Python/PyTorch...")
    # --- Vectorized Simulation Logic (Bundled 1+3 bit) ---
    sim_packed_4bit_uint8 = w_packed_4bit_kernel_check.cpu().view(torch.uint8)
    sim_a_quant = a_quant_kernel_check.cpu().double() # Use double for activations in simulation
    min_exp_check_f = float(min_exp_check)
    delta_lsb_kernel_check_f = float(delta_lsb_kernel_check)
    N_out_sim, packed_cols_sim = sim_packed_4bit_uint8.shape

    # Unpack codes
    codes1_uint8 = (sim_packed_4bit_uint8 >> 4) # High nibble
    codes2_uint8 = (sim_packed_4bit_uint8 & 0x0F) # Low nibble

    # Reshape activations for broadcasting
    sim_a_quant_r = sim_a_quant.view(1, -1) # Shape (1, M_in)
    act1 = sim_a_quant_r[:, 0::2] # Shape (1, M_in/2)
    act2 = sim_a_quant_r[:, 1::2] # Shape (1, M_in/2)

    # Initialize terms tensor
    terms = torch.zeros((N_out_sim, packed_cols_sim), dtype=torch.float64)

    # --- Process first nibble (codes1) ---
    mask_nz1 = codes1_uint8 != 0
    codes1_nz = codes1_uint8[mask_nz1]
    sign_bit1 = (codes1_nz >> 3) # 1 for negative, 0 for positive
    exp_map1 = (codes1_nz & 0x07) # 3 LSBs
    sign1 = 1.0 - 2.0 * sign_bit1.double() # Convert 0/1 to +1.0/-1.0
    # Unmap exponent based on sign
    exponent1 = torch.where(
        sign_bit1 == 0, # Positive
        exp_map1.double() - 1.0 + min_exp_check_f, # Map 1..7 -> min_exp..max_exp-1
        exp_map1.double() + min_exp_check_f       # Map 0..7 -> min_exp..max_exp
    )
    pow2_exp1 = torch.pow(2.0, exponent1)
    # Broadcast act1: (1, M_in/2) to match terms[mask_nz1] shape
    act1_nz = act1.expand(N_out_sim, -1)[mask_nz1]
    terms[mask_nz1] += sign1 * act1_nz * pow2_exp1

    # --- Process second nibble (codes2) ---
    mask_nz2 = codes2_uint8 != 0
    codes2_nz = codes2_uint8[mask_nz2]
    sign_bit2 = (codes2_nz >> 3)
    exp_map2 = (codes2_nz & 0x07)
    sign2 = 1.0 - 2.0 * sign_bit2.double()
    exponent2 = torch.where(
        sign_bit2 == 0,
        exp_map2.double() - 1.0 + min_exp_check_f,
        exp_map2.double() + min_exp_check_f
    )
    pow2_exp2 = torch.pow(2.0, exponent2)
    # Broadcast act2
    act2_nz = act2.expand(N_out_sim, -1)[mask_nz2]
    terms[mask_nz2] += sign2 * act2_nz * pow2_exp2

    # Sum terms across the packed dimension
    simulated_output_raw = torch.sum(terms, dim=1) # Shape (N_out)

    # Scale by activation LSB
    simulated_output_check = (simulated_output_raw * delta_lsb_kernel_check_f).float() # Final cast to float32
    # --- End Vectorized Simulation ---

    simulated_output_check = simulated_output_check.to(device=DEV) # Move final result to GPU

    # 8. Compare Kernel Output with Simulation and FP32
    print("\nComparing outputs...")
    print("--- Bundled 4-bit Kernel (with Internal Act Quant & Bias Add) ---")
    # Note: Simulation logic still represents the *original* kernel without internal quant/bias.
    # A direct comparison Kernel vs Sim is less meaningful now for exactness,
    # but we can still compare the new kernel output vs FP32.
    # The simulation output `simulated_output_check` does NOT include bias.
    simulated_output_with_bias = simulated_output_check + bias_check # Add bias to simulation result

    abs_diff_sim = torch.abs(output_log_check - simulated_output_with_bias)
    mean_abs_err_sim = torch.mean(abs_diff_sim).item()
    max_abs_err_sim = torch.max(abs_diff_sim).item()
    rel_err_sim = torch.mean(abs_diff_sim / (torch.abs(simulated_output_with_bias) + 1e-9)).item()
    print(f"Mean Absolute Error (Kernel vs Biased Sim): {mean_abs_err_sim:.6f}")
    print(f"Max Absolute Error (Kernel vs Biased Sim): {max_abs_err_sim:.6f}")
    print(f"Mean Relative Error (Kernel vs Biased Sim): {rel_err_sim:.6f}")

    # Compare kernel output with original FP32 output (mul_fp32 is N_out x 1, does not include bias)
    fp32_output_with_bias = mul_fp32.squeeze() + bias_check # Add bias to FP32 result for comparison
    abs_diff_fp32 = torch.abs(output_log_check - fp32_output_with_bias)
    mean_abs_err_fp32 = torch.mean(abs_diff_fp32).item()
    max_abs_err_fp32 = torch.max(abs_diff_fp32).item()
    rel_err_fp32 = torch.mean(abs_diff_fp32 / (torch.abs(fp32_output_with_bias) + 1e-9)).item()
    print(f"Mean Absolute Error (New Kernel vs Biased FP32): {mean_abs_err_fp32:.6f}")
    print(f"Max Absolute Error (New Kernel vs Biased FP32): {max_abs_err_fp32:.6f}")
    print(f"Mean Relative Error (New Kernel vs Biased FP32): {rel_err_fp32:.6f}")

    # print("\n--- Bundled Float Mul Kernel ---") # Commenting out FloatMul comparison for now
    # # Compare FloatMul kernel to the bit shift simulation
    # abs_diff_sim_fm = torch.abs(output_log_check_fm - simulated_output_check)
    # mean_abs_err_sim_fm = torch.mean(abs_diff_sim_fm).item()
    # max_abs_err_sim_fm = torch.max(abs_diff_sim_fm).item()
    # rel_err_sim_fm = torch.mean(abs_diff_sim_fm / (torch.abs(simulated_output_check) + 1e-9)).item()
    # print(f"Mean Absolute Error (FloatMul Kernel vs BitShift Sim): {mean_abs_err_sim_fm:.6f}")
    # print(f"Max Absolute Error (FloatMul Kernel vs BitShift Sim): {max_abs_err_sim_fm:.6f}")
    # print(f"Mean Relative Error (FloatMul Kernel vs BitShift Sim): {rel_err_sim_fm:.6f}")

    # # Compare FloatMul kernel output with original FP32 output
    # abs_diff_fp32_fm = torch.abs(output_log_check_fm - mul_fp32.squeeze())
    # mean_abs_err_fp32_fm = torch.mean(abs_diff_fp32_fm).item()
    # max_abs_err_fp32_fm = torch.max(abs_diff_fp32_fm).item()
    # rel_err_fp32_fm = torch.mean(abs_diff_fp32_fm / (torch.abs(mul_fp32.squeeze()) + 1e-9)).item()
    # print(f"Mean Absolute Error (FloatMul Kernel vs FP32): {mean_abs_err_fp32_fm:.6f}")
    # print(f"Max Absolute Error (FloatMul Kernel vs FP32): {max_abs_err_fp32_fm:.6f}")
    # print(f"Mean Relative Error (FloatMul Kernel vs FP32): {rel_err_fp32_fm:.6f}")


print("\nBenchmarking finished.")
