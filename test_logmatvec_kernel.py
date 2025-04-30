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


# --- Setup Log Quantization Data ---
print('\n--- LogMatVec Kernel Benchmark ---')
LOG_BITS = 4 # Example bitwidth for log quantization
ACT_BITS = 8 # Example bitwidth for activation quantization

# 1. Initialize LogQuantizer
log_quantizer = LogQuantizer()
log_quantizer.configure(bits=LOG_BITS)

# 2. Quantize Weights (using the FP32 matrix)
print(f"Quantizing weights to {LOG_BITS}-bit Log format...")
log_quantizer.find_params(mat_fp32, weight=True)
# quantize returns (quantized_value, exponent)
q_w_log, w_exp = log_quantizer.quantize(mat_fp32)
w_sign = torch.sign(mat_fp32).to(torch.int8) # Get signs as Int8 (+1, -1, 0)

# Prepare weight tensors for the kernel
w_exp_kernel = w_exp.to(device=DEV, dtype=torch.int32).contiguous()
w_sign_kernel = w_sign.to(device=DEV).contiguous() # Already int8

print("Weight quantization done.")

# 3. Quantize Activations (Linear Symmetric) - using the FP32 vector
print(f"Quantizing activations to {ACT_BITS}-bit Linear format...")
# Use the column vector vec_fp32_col (M_in, 1)
a_min = vec_fp32_col.min()
a_max = vec_fp32_col.max()
q_max_act = 2**(ACT_BITS - 1) - 1
q_min_act = -2**(ACT_BITS - 1)
# Symmetric quantization scale based on max absolute value
delta_lsb = torch.max(torch.abs(a_min), torch.abs(a_max)) / q_max_act
# Quantize and clamp
a_quant = torch.round(vec_fp32_col / delta_lsb).clamp(q_min_act, q_max_act).to(torch.int32)

# Prepare activation tensor for the kernel (needs to be 1D)
a_quant_kernel = a_quant.squeeze().to(device=DEV).contiguous() # Shape (M_in)
delta_lsb_kernel = delta_lsb.item() # Pass scale as float

print("Activation quantization done.")

# 4. Prepare Output Tensor
output_log = torch.zeros(N_out, device=DEV, dtype=torch.float32) # Kernel outputs float32

# --- Benchmark LogMatVec Kernel ---
print("Benchmarking custom kernel...")
torch.cuda.synchronize()
tick = time.time()
for _ in range(COUNT):
    # Ensure output is zeroed or use a fresh tensor if needed
    # output_log.zero_()
    logmatvec_cuda.forward(
        a_quant_kernel,
        w_exp_kernel,
        w_sign_kernel,
        output_log, # Pass the pre-allocated tensor
        delta_lsb_kernel
    )
    torch.cuda.synchronize()
log_kernel_time = (time.time() - tick) / COUNT
print(f'LogMatVec Kernel ({LOG_BITS}bW/{ACT_BITS}bA): {log_kernel_time:.6f} s')

if log_kernel_time > 0 and fp32_time > 0:
     print(f'  Speedup vs FP32: {fp32_time / log_kernel_time:.2f}x')
if log_kernel_time > 0 and 'fp16_time' in locals() and fp16_time > 0:
     print(f'  Speedup vs FP16: {fp16_time / log_kernel_time:.2f}x')


# --- Optional: Correctness Check ---
print('\n--- Correctness Check (Simulated vs Kernel) ---')
# Simulate the operation in PyTorch using log-quantized weights and float activations
# Note: This uses the float activations, while the kernel uses quantized activations.
# A perfect match isn't expected, but the magnitude should be comparable.
print("Simulating Log MatVec in Python/PyTorch...")
simulated_output = torch.matmul(q_w_log.to(DEV), vec_fp32_col.to(DEV)) # W_log @ a_float

# Compare simulated output with kernel output
if output_log.numel() == simulated_output.numel():
    # Ensure simulated_output is also (N_out, 1) or (N_out)
    simulated_output = simulated_output.view_as(output_log) # Reshape if needed
    abs_diff = torch.abs(output_log - simulated_output)
    mean_abs_err = torch.mean(abs_diff).item()
    max_abs_err = torch.max(abs_diff).item()
    print(f"Mean Absolute Error (Kernel vs Simulated): {mean_abs_err:.6f}")
    print(f"Max Absolute Error (Kernel vs Simulated): {max_abs_err:.6f}")

    # Compare kernel output with original FP32 output
    abs_diff_fp32 = torch.abs(output_log.squeeze() - mul_fp32.squeeze()) # Ensure shapes match
    mean_abs_err_fp32 = torch.mean(abs_diff_fp32).item()
    max_abs_err_fp32 = torch.max(abs_diff_fp32).item()
    print(f"Mean Absolute Error (Kernel vs FP32): {mean_abs_err_fp32:.6f}")
    print(f"Max Absolute Error (Kernel vs FP32): {max_abs_err_fp32:.6f}")

else:
    print("Output shape mismatch, cannot compare.")
    print(f"Kernel output shape: {output_log.shape}")
    print(f"Simulated output shape: {simulated_output.shape}")


print("\nBenchmarking finished.")
