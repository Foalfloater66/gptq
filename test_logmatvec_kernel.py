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

# 1. Create Random Packed Weight Components
print(f"Generating random packed kernel inputs ({LOG_BITS}bW / {ACT_BITS}bA equivalent)...")
# Packed exponents: each byte holds two 4-bit values (0-15). Generate as uint8.
packed_in_features_rand = M_in // 2
w_packed_exp_kernel_rand_uint8 = torch.randint(0, 256, (N_out, packed_in_features_rand), device=DEV, dtype=torch.uint8).contiguous()
# Random signs: -1, 0, 1 (still need one sign per original weight)
w_sign_kernel_rand = torch.randint(-1, 2, (N_out, M_in), device=DEV, dtype=torch.int8).contiguous()

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
    logmatvec_cuda.forward_packed4bit( # Call the new packed function
        a_quant_kernel_rand,
        w_packed_exp_kernel_rand_uint8.view(torch.int8), # Pass packed exponents (viewed as int8)
        w_sign_kernel_rand,       # Pass signs
        output_log_rand,
        delta_lsb_kernel_dummy,
        min_exp_dummy             # Pass dummy min_exp
    )
    torch.cuda.synchronize()
log_kernel_time_rand = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel (Packed Random {LOG_BITS}bW/{ACT_BITS}bA Equiv): {log_kernel_time_rand:.6f} s') # Updated label

if log_kernel_time_rand > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time_rand:.2f}x')
if log_kernel_time_rand > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time_rand:.2f}x')


# --- Benchmark LogMatVec Kernel with Random Packed Inputs (Float Mul Version) ---
print("\nBenchmarking custom kernel with random packed inputs (Float Mul Version)...")
output_log_rand_fm = torch.zeros(N_out, device=DEV, dtype=torch.float32) # Separate output tensor
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    # output_log_rand_fm.zero_()
    # Call the float mul version
    logmatvec_cuda.forward_packed4bit_floatmul(
        a_quant_kernel_rand,
        w_packed_exp_kernel_rand_uint8.view(torch.int8), # Pass packed exponents (viewed as int8)
        w_sign_kernel_rand,       # Pass signs
        output_log_rand_fm,       # Pass output tensor
        delta_lsb_kernel_dummy,
        min_exp_dummy
    )
    torch.cuda.synchronize()
log_kernel_time_rand_fm = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel (FloatMul Packed Random {LOG_BITS}bW/{ACT_BITS}bA Equiv): {log_kernel_time_rand_fm:.6f} s')

if log_kernel_time_rand_fm > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time_rand_fm:.2f}x')
if log_kernel_time_rand_fm > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time_rand_fm:.2f}x')
if log_kernel_time_rand_fm > 0 and log_kernel_time_rand > 0:
     print(f'  Speedup (BitShift vs FloatMul): {log_kernel_time_rand_fm / log_kernel_time_rand:.2f}x')


# --- Correctness Check (Using Actual Quantized and Packed Data) ---
print('\n--- Correctness Check (Simulated vs Kernels on Quantized Packed Data) ---') # Updated title
# Use the original FP32 matrix and vector for this section
# 1. Initialize LogQuantizer
log_quantizer = LogQuantizer()
log_quantizer.configure(bits=LOG_BITS) # Must be 4

# 2. Quantize Weights (using the FP32 matrix)
print(f"Quantizing weights to {LOG_BITS}-bit Log format for check...")
log_quantizer.find_params(mat_fp32, weight=True)
# Get quantized value (not needed here), exponents, and signs
_, w_exp_check, w_sign_check = log_quantizer.quantize(mat_fp32)
print("Weight quantization done for check.")

# 3. Pack Weights
print("Packing weights...")
try:
    # LogQuantizer.pack returns int8, which is fine here as it represents packed bits
    w_packed_exp_check, w_sign_check_packed = log_quantizer.pack(w_exp_check, w_sign_check)
    # Ensure they are on the correct device and contiguous
    w_packed_exp_kernel_check = w_packed_exp_check.to(device=DEV, dtype=torch.int8).contiguous() # Ensure int8 type
    w_sign_kernel_check = w_sign_check_packed.to(device=DEV).contiguous()
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

    # 6. Run Kernel Once with Quantized Packed Data
    print("Running kernel once with quantized packed data...")
    logmatvec_cuda.forward_packed4bit( # Call the new packed function
        a_quant_kernel_check,
        w_packed_exp_kernel_check, # Pass packed exponents
        w_sign_kernel_check,       # Pass signs
        output_log_check,
        delta_lsb_kernel_check,
        min_exp_check              # Pass actual min_exp
    )
    torch.cuda.synchronize()

    # Also run the FloatMul kernel once for comparison
    output_log_check_fm = torch.zeros(N_out, device=DEV, dtype=torch.float32)
    print("Running float mul kernel once with quantized packed data...")
    logmatvec_cuda.forward_packed4bit_floatmul(
        a_quant_kernel_check,
        w_packed_exp_kernel_check,
        w_sign_kernel_check,
        output_log_check_fm,
        delta_lsb_kernel_check,
        min_exp_check
    )
    torch.cuda.synchronize()


    # 7. Simulate the Kernel Operation with Packing/Unpacking in PyTorch
    print("Simulating exact kernel operation (bit shift logic) in Python/PyTorch...")
    # --- Vectorized Simulation Logic (Bit Shift) ---
    # Perform simulation on CPU to avoid GPU OOM during check, use float64 for precision
    sim_packed_exp_uint8 = w_packed_exp_kernel_check.cpu().view(torch.uint8)
    sim_signs = w_sign_kernel_check.cpu().float() # Use float for calculations
    sim_a_quant = a_quant_kernel_check.cpu().float() # Use float for calculations
    min_exp_check_f = float(min_exp_check) # Use float
    delta_lsb_kernel_check_f = float(delta_lsb_kernel_check)

    # Unpack exponents using vectorized operations
    mapped_exp1 = (sim_packed_exp_uint8 >> 4).float() # High nibble
    mapped_exp2 = (sim_packed_exp_uint8 & 0x0F).float() # Low nibble

    # Unmap to actual exponents
    exponent1 = mapped_exp1 + min_exp_check_f
    exponent2 = mapped_exp2 + min_exp_check_f

    # Get signs for each position
    # Shape: (N_out, M_in) -> (N_out, M_in/2)
    sign1 = sim_signs[:, 0::2]
    sign2 = sim_signs[:, 1::2]

    # Get activations for each position
    # Shape: (M_in) -> (M_in/2)
    act1 = sim_a_quant[0::2]
    act2 = sim_a_quant[1::2]

    # Calculate powers of 2 (use float64 for intermediate pow for precision)
    pow2_exp1 = torch.pow(2.0, exponent1.double())
    pow2_exp2 = torch.pow(2.0, exponent2.double())

    # Calculate contributions for each packed pair element-wise
    # Need to broadcast activations: act1/act2 (M_in/2) -> (1, M_in/2)
    # pow2_exp1/2 and sign1/2 are (N_out, M_in/2)
    term1 = (act1.double().unsqueeze(0) * pow2_exp1) * sign1.double()
    term2 = (act2.double().unsqueeze(0) * pow2_exp2) * sign2.double()

    # Sum contributions across the packed dimension (dim=1)
    # Add term1 and term2 contributions element-wise first, then sum
    simulated_output_raw = torch.sum(term1 + term2, dim=1) # Shape (N_out)

    # Scale by activation LSB
    simulated_output_check = (simulated_output_raw * delta_lsb_kernel_check_f).float() # Final cast to float32
    # --- End Vectorized Simulation ---

    simulated_output_check = simulated_output_check.to(device=DEV) # Move final result to GPU

    # 8. Compare Kernel Outputs with Simulation and FP32
    print("\nComparing outputs...")
    print("--- Bit Shift Kernel ---")
    abs_diff_sim_bs = torch.abs(output_log_check - simulated_output_check)
    mean_abs_err_sim_bs = torch.mean(abs_diff_sim_bs).item()
    max_abs_err_sim_bs = torch.max(abs_diff_sim_bs).item()
    rel_err_sim_bs = torch.mean(abs_diff_sim_bs / (torch.abs(simulated_output_check) + 1e-9)).item()
    print(f"Mean Absolute Error (BitShift Kernel vs Simulated): {mean_abs_err_sim_bs:.6f}")
    print(f"Max Absolute Error (BitShift Kernel vs Simulated): {max_abs_err_sim_bs:.6f}")
    print(f"Mean Relative Error (BitShift Kernel vs Simulated): {rel_err_sim_bs:.6f}")

    abs_diff_fp32_bs = torch.abs(output_log_check - mul_fp32.squeeze()) # Squeeze FP32 output
    mean_abs_err_fp32_bs = torch.mean(abs_diff_fp32_bs).item()
    max_abs_err_fp32_bs = torch.max(abs_diff_fp32_bs).item()
    rel_err_fp32_bs = torch.mean(abs_diff_fp32_bs / (torch.abs(mul_fp32.squeeze()) + 1e-9)).item()
    print(f"Mean Absolute Error (BitShift Kernel vs FP32): {mean_abs_err_fp32_bs:.6f}")
    print(f"Max Absolute Error (BitShift Kernel vs FP32): {max_abs_err_fp32_bs:.6f}")
    print(f"Mean Relative Error (BitShift Kernel vs FP32): {rel_err_fp32_bs:.6f}")

    print("\n--- Float Mul Kernel ---")
    # Compare FloatMul kernel to the same simulation (which represents the bit shift logic)
    # This comparison isn't strictly "correctness" but shows difference between kernel types
    abs_diff_sim_fm = torch.abs(output_log_check_fm - simulated_output_check)
    mean_abs_err_sim_fm = torch.mean(abs_diff_sim_fm).item()
    max_abs_err_sim_fm = torch.max(abs_diff_sim_fm).item()
    rel_err_sim_fm = torch.mean(abs_diff_sim_fm / (torch.abs(simulated_output_check) + 1e-9)).item()
    print(f"Mean Absolute Error (FloatMul Kernel vs Simulated): {mean_abs_err_sim_fm:.6f}")
    print(f"Max Absolute Error (FloatMul Kernel vs Simulated): {max_abs_err_sim_fm:.6f}")
    print(f"Mean Relative Error (FloatMul Kernel vs Simulated): {rel_err_sim_fm:.6f}")

    # Compare FloatMul kernel output with original FP32 output
    abs_diff_fp32_fm = torch.abs(output_log_check_fm - mul_fp32.squeeze())
    mean_abs_err_fp32_fm = torch.mean(abs_diff_fp32_fm).item()
    max_abs_err_fp32_fm = torch.max(abs_diff_fp32_fm).item()
    rel_err_fp32_fm = torch.mean(abs_diff_fp32_fm / (torch.abs(mul_fp32.squeeze()) + 1e-9)).item()
    print(f"Mean Absolute Error (FloatMul Kernel vs FP32): {mean_abs_err_fp32_fm:.6f}")
    print(f"Max Absolute Error (FloatMul Kernel vs FP32): {max_abs_err_fp32_fm:.6f}")
    print(f"Mean Relative Error (FloatMul Kernel vs FP32): {rel_err_fp32_fm:.6f}")


print("\nBenchmarking finished.")
