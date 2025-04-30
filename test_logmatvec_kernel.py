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


# --- Setup LogMatVec Kernel Benchmark Data (Randomized Inputs) ---
print('\n--- LogMatVec Kernel Benchmark (Randomized Inputs) ---')
LOG_BITS = 4 # Define bitwidth for context, though not used for random generation range directly
ACT_BITS = 8 # Define bitwidth for activation range

# 1. Create Random Weight Components (Matching Kernel Input Types)
print(f"Generating random kernel inputs ({LOG_BITS}bW / {ACT_BITS}bA equivalent)...")
# Plausible exponent range for 4-bit log (e.g., -10 to 2)
w_exp_kernel_rand = torch.randint(-10, 3, (N_out, M_in), device=DEV, dtype=torch.int32).contiguous()
# Random signs: -1, 0, 1
w_sign_kernel_rand = torch.randint(-1, 2, (N_out, M_in), device=DEV, dtype=torch.int8).contiguous()

# 2. Create Random Quantized Activations (Matching Kernel Input Type)
# Range for 8-bit symmetric: [-128, 127]
q_min_act = -2**(ACT_BITS - 1)
q_max_act = 2**(ACT_BITS - 1) - 1
a_quant_kernel_rand = torch.randint(q_min_act, q_max_act + 1, (M_in,), device=DEV, dtype=torch.int32).contiguous()

# 3. Set Dummy Activation Scale for Benchmark Loop
delta_lsb_kernel_dummy = 1.0 # Kernel needs a float scale, use 1.0 for isolated benchmark

print("Random kernel inputs generated.")

# 4. Prepare Output Tensor
output_log_rand = torch.zeros(N_out, device=DEV, dtype=torch.float32) # Kernel outputs float32

# --- Benchmark LogMatVec Kernel with Random Inputs ---
print("Benchmarking custom kernel with random inputs...")
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    # Ensure output is zeroed or use a fresh tensor if needed
    # output_log_rand.zero_()
    logmatvec_cuda.forward(
        a_quant_kernel_rand,
        w_exp_kernel_rand,
        w_sign_kernel_rand,
        output_log_rand, # Pass the pre-allocated tensor
        delta_lsb_kernel_dummy # Use dummy scale
    )
    torch.cuda.synchronize()
log_kernel_time_rand = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel (Random {LOG_BITS}bW/{ACT_BITS}bA Equiv): {log_kernel_time_rand:.6f} s')

if log_kernel_time_rand > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time_rand:.2f}x')
if log_kernel_time_rand > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time_rand:.2f}x')


# --- Correctness Check (Using Actual Quantized Data) ---
print('\n--- Correctness Check (Simulated vs Kernel on Quantized Data) ---')
# Use the original FP32 matrix and vector for this section
# 1. Initialize LogQuantizer
log_quantizer = LogQuantizer()
log_quantizer.configure(bits=LOG_BITS)

# 2. Quantize Weights (using the FP32 matrix)
print(f"Quantizing weights to {LOG_BITS}-bit Log format for check...")
log_quantizer.find_params(mat_fp32, weight=True)
q_w_log_check, w_exp_check = log_quantizer.quantize(mat_fp32) # Use chunking version if needed
w_sign_check = torch.sign(mat_fp32).to(torch.int8)
w_exp_kernel_check = w_exp_check.to(device=DEV, dtype=torch.int32).contiguous()
w_sign_kernel_check = w_sign_check.to(device=DEV).contiguous()
print("Weight quantization done for check.")

# 3. Quantize Activations (Linear Symmetric) - using the FP32 vector
print(f"Quantizing activations to {ACT_BITS}-bit Linear format for check...")
a_min_check = vec_fp32_col.min()
a_max_check = vec_fp32_col.max()
delta_lsb_check = torch.max(torch.abs(a_min_check), torch.abs(a_max_check)) / q_max_act
if delta_lsb_check == 0: delta_lsb_check += 1e-9 # Avoid division by zero if input is all zero
a_quant_check = torch.round(vec_fp32_col / delta_lsb_check).clamp(q_min_act, q_max_act).to(torch.int32)
a_quant_kernel_check = a_quant_check.squeeze().to(device=DEV).contiguous()
delta_lsb_kernel_check = delta_lsb_check.item()
print("Activation quantization done for check.")

# 4. Prepare Output Tensor for Check
output_log_check = torch.zeros(N_out, device=DEV, dtype=torch.float32)

# 5. Run Kernel Once with Quantized Data
print("Running kernel once with quantized data...")
logmatvec_cuda.forward(
    a_quant_kernel_check,
    w_exp_kernel_check,
    w_sign_kernel_check,
    output_log_check,
    delta_lsb_kernel_check
)
torch.cuda.synchronize()

# 6. Simulate the Kernel Operation in PyTorch
print("Simulating exact kernel operation in Python/PyTorch...")
# Ensure tensors are on the correct device and float for calculation
w_exp_f = w_exp_kernel_check.float()
w_sign_f = w_sign_kernel_check.float()
a_quant_f = a_quant_kernel_check.float() # Shape (M_in)

# Calculate powers of 2 for exponents
pow2_exp = torch.pow(2.0, w_exp_f) # Shape (N_out, M_in)

# Element-wise product: sign * 2^exp
w_eff = w_sign_f * pow2_exp # Shape (N_out, M_in)

# Multiply by quantized activations (broadcast a_quant_f)
# Result shape: (N_out, M_in)
terms = w_eff * a_quant_f.unsqueeze(0) # Add batch dim to activations for broadcasting

# Sum across input features dimension (dim=1)
simulated_output_raw = torch.sum(terms, dim=1) # Shape (N_out)

# Scale by activation LSB
simulated_output_check = simulated_output_raw * delta_lsb_kernel_check

# 7. Compare Kernel Output with Simulation and FP32
print("Comparing outputs...")
abs_diff_sim = torch.abs(output_log_check - simulated_output_check)
mean_abs_err_sim = torch.mean(abs_diff_sim).item()
max_abs_err_sim = torch.max(abs_diff_sim).item()
rel_err_sim = torch.mean(abs_diff_sim / (torch.abs(simulated_output_check) + 1e-9)).item() # Avoid div by zero
print(f"Mean Absolute Error (Kernel vs Simulated): {mean_abs_err_sim:.6f}")
print(f"Max Absolute Error (Kernel vs Simulated): {max_abs_err_sim:.6f}")
print(f"Mean Relative Error (Kernel vs Simulated): {rel_err_sim:.6f}")

# Compare kernel output with original FP32 output (mul_fp32 is N_out x 1)
abs_diff_fp32 = torch.abs(output_log_check - mul_fp32.squeeze()) # Squeeze FP32 output
mean_abs_err_fp32 = torch.mean(abs_diff_fp32).item()
max_abs_err_fp32 = torch.max(abs_diff_fp32).item()
rel_err_fp32 = torch.mean(abs_diff_fp32 / (torch.abs(mul_fp32.squeeze()) + 1e-9)).item()
print(f"Mean Absolute Error (Kernel vs FP32): {mean_abs_err_fp32:.6f}")
print(f"Max Absolute Error (Kernel vs FP32): {max_abs_err_fp32:.6f}")
print(f"Mean Relative Error (Kernel vs FP32): {rel_err_fp32:.6f}")


print("\nBenchmarking finished.")
