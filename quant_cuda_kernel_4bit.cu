#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAException.h> // For C10_CUDA_KERNEL_LAUNCH_CHECK

// Forward declarations for explicit float and half kernels
__global__ void VecQuant4MatMulKernel_float(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros, // Note: This is zero_point * scale
    int height, // Packed height (orig_height / 8)
    int width
);

__global__ void VecQuant4MatMulKernel_half(
    const c10::Half* __restrict__ vec, // Use c10::Half
    const    int* __restrict__ mat,
           float* __restrict__ mul, // Output is still float due to float accumulation
    const   half* __restrict__ scales,
    const   half* __restrict__ zeros, // Note: This is zero_point * scale
    int height, // Packed height (orig_height / 8)
    int width
);

__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec, // Input vector (FP16)
    const    int* __restrict__ mat, // Packed weights (INT32)
           float* __restrict__ mul, // Output vector (FP32)
    const  float* __restrict__ scales, // FP32 scales
    const  float* __restrict__ zeros,  // FP32 zero points * scales
    int height, // Packed height (orig_height / 8)
    int width
);

// --- Configuration ---
// These can be tuned depending on the GPU architecture and matrix sizes
const int BLOCKWIDTH_4BIT  = 256; // Threads per block (must be multiple of 8 for 4-bit)
const int BLOCKHEIGHT_4BIT = 8;   // Rows processed per block per iteration (tune for occupancy)
// ---

// Helper function to reinterpret int bits as unsigned int
__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// --- CUDA Kernel Launchers (accept raw pointers) ---

// Launcher for the standard kernel
void vecquant4matmul_launcher(
    const void* vec_ptr,      // Use void* for generic pointer
    const int* mat_ptr,
    float* mul_ptr,           // Output is always float
    const void* scales_ptr,   // Use void*
    const void* zeros_ptr,    // Use void*
    c10::ScalarType vec_type, // Pass input data type
    int height,
    int width,
    cudaStream_t stream
) {
  // Grid dimensions (calculated from height/width passed in)
  dim3 blocks(
    (height + BLOCKHEIGHT_4BIT - 1) / BLOCKHEIGHT_4BIT, // Number of blocks in Y dimension (for rows)
    (width + BLOCKWIDTH_4BIT - 1) / BLOCKWIDTH_4BIT    // Number of blocks in X dimension (for columns)
  );
  // Block dimensions
  dim3 threads(BLOCKWIDTH_4BIT); // Threads per block

  // Explicitly call float or half kernel based on vec_type passed in
  if (vec_type == torch::kFloat32) {
      VecQuant4MatMulKernel_float<<<blocks, threads, 0, stream>>>( // Use stream
          static_cast<const float*>(vec_ptr), // Cast void*
          mat_ptr,
          mul_ptr,
          static_cast<const float*>(scales_ptr), // Cast void*
          static_cast<const float*>(zeros_ptr),  // Cast void*
          height, width
      );
  } else if (vec_type == torch::kFloat16) {
      VecQuant4MatMulKernel_half<<<blocks, threads, 0, stream>>>( // Use stream
          static_cast<const c10::Half*>(vec_ptr), // Cast void* to c10::Half*
          mat_ptr,
          mul_ptr, // Output is still float
          static_cast<const half*>(scales_ptr), // Cast void*
          static_cast<const half*>(zeros_ptr),  // Cast void*
          height, width
      );
  } else {
      // This case should ideally not be reached due to checks in C++ wrapper
      // If needed, add specific error handling or default behavior
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for kernel launch errors
}


// Launcher for the faster kernel (accepts raw pointers)
void vecquant4matmul_faster_launcher( // Renamed from _cuda
    const c10::Half* vec_ptr, // Expects c10::Half*
    const int* mat_ptr,
    float* mul_ptr,
    const float* scales_ptr,  // Expects float*
    const float* zeros_ptr,   // Expects float*
    int height,
    int width,
    cudaStream_t stream
) {
  // Grid dimensions (calculated from height/width passed in)
  dim3 blocks(
    (height + BLOCKHEIGHT_4BIT - 1) / BLOCKHEIGHT_4BIT, // Number of blocks in Y dimension (for rows)
    (width + BLOCKWIDTH_4BIT - 1) / BLOCKWIDTH_4BIT    // Number of blocks in X dimension (for columns)
  );
  // Block dimensions
  dim3 threads(BLOCKWIDTH_4BIT); // Threads per block

  VecQuant4MatMulKernelFaster<<<blocks, threads, 0, stream>>>( // Use stream
    (const half2*) vec_ptr, // Cast c10::Half* to half2*
    mat_ptr,
    mul_ptr,
    scales_ptr,
    zeros_ptr,
    height, // Packed height
    width
  );
   C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for kernel launch errors
}


// --- Kernel Implementations ---

// Explicit kernel for FLOAT input
__global__ void VecQuant4MatMulKernel_float(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros, // zero_point * scale
    int height, // Packed height (orig_height / 8)
    int width
) {
    // Calculate the column index for this thread
    const int col = blockIdx.y * BLOCKWIDTH_4BIT + threadIdx.x;

    // Bounds check for column
    if (col >= width) return;

    // Calculate the starting row index for this block
    const int row_base = blockIdx.x * BLOCKHEIGHT_4BIT;

    // Load scale and zero for the current column
    const float scale = scales[col];
    const float zero = zeros[col]; // This is zero_point * scale

    // Accumulator for the result - ALWAYS use float for accumulation
    float res = 0.0f;

    // Loop over the packed rows assigned to this block
    for (int packed_row_offset = 0; packed_row_offset < BLOCKHEIGHT_4BIT; ++packed_row_offset) {
        const int packed_row = row_base + packed_row_offset;
        // Bounds check for row (important if height is not perfectly divisible by BLOCKHEIGHT_4BIT)
        if (packed_row >= height) continue;

        const int original_row_base = packed_row * 8; // Base row in the original un-packed matrix

        // Load the packed 32-bit integer containing 8 x 4-bit weights
        const unsigned int packed_val = as_unsigned(mat[packed_row * width + col]);

        // Unpack 8 values and multiply-accumulate with corresponding vector elements
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Extract the i-th 4-bit value
            float val = static_cast<float>((packed_val >> (i * 4)) & 0xF);
            // Dequantize: scale * quantized_value - zero_point * scale
            float dequant_val = scale * val - zero;
            // Multiply with corresponding vector element and accumulate
            res += dequant_val * vec[original_row_base + i];
        }
    }

    // Atomically add the partial result to the output vector element
    atomicAdd(&mul[col], res);
}

// Explicit kernel for HALF input
__global__ void VecQuant4MatMulKernel_half(
    const c10::Half* __restrict__ vec, // Use c10::Half
    const    int* __restrict__ mat,
           float* __restrict__ mul, // Output buffer MUST be float
    const c10::Half* __restrict__ scales, // Use c10::Half
    const c10::Half* __restrict__ zeros, // Use c10::Half
    int height, // Packed height (orig_height / 8)
    int width
) {
    // Calculate the column index for this thread
    const int col = blockIdx.y * BLOCKWIDTH_4BIT + threadIdx.x;

    // Bounds check for column
    if (col >= width) return;

    // Calculate the starting row index for this block
    // Each block processes BLOCKHEIGHT_4BIT packed rows (BLOCKHEIGHT_4BIT * 8 original rows)
    const int row_base = blockIdx.x * BLOCKHEIGHT_4BIT;

    // Load scale and zero for the current column
    const c10::Half scale_h = scales[col]; // Use c10::Half
    const c10::Half zero_h = zeros[col];   // Use c10::Half

    // Accumulator for the result - ALWAYS use float for accumulation
    float res = 0.0f;

    // Loop over the packed rows assigned to this block
    for (int packed_row_offset = 0; packed_row_offset < BLOCKHEIGHT_4BIT; ++packed_row_offset) {
        const int packed_row = row_base + packed_row_offset;
        // Bounds check for row (important if height is not perfectly divisible by BLOCKHEIGHT_4BIT)
        if (packed_row >= height) continue;

        const int original_row_base = packed_row * 8; // Base row in the original un-packed matrix

        // Load the packed 32-bit integer containing 8 x 4-bit weights
        const unsigned int packed_val = as_unsigned(mat[packed_row * width + col]);

        // Unpack 8 values and multiply-accumulate with corresponding vector elements
        // This loop should be unrolled by the compiler
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            // Extract the i-th 4-bit value and convert to c10::Half
            // Note: CUDA's half type is implicitly convertible for __int2half_rn if needed,
            // but let's be explicit if direct conversion isn't available.
            // Assuming __int2half_rn returns CUDA's half, which can be assigned/cast to c10::Half
            c10::Half val_h = static_cast<c10::Half>(__int2half_rn((packed_val >> (i * 4)) & 0xF));

            // Dequantize: scale * quantized_value - zero_point * scale
            // Perform calculation in float due to disabled half operators
            float dequant_val_f = static_cast<float>(scale_h) * static_cast<float>(val_h) - static_cast<float>(zero_h);
            // Multiply with corresponding vector element (c10::Half) and accumulate in float
            // Ensure we don't read past the end of the input vector if padding occurred
            // Note: The Python code pads the input vector, so this check might be redundant
            // if padding is guaranteed, but it's safer.
            // int vec_idx = original_row_base + i;
            // if (vec_idx < total_input_features) { // Need total_input_features passed or calculated
                 // Accumulate the float dequantized value multiplied by the float converted vector element
                 res += dequant_val_f * static_cast<float>(vec[original_row_base + i]);
            // }
        }
    }

    // Atomically add the partial result to the output vector element
    // Note: The original kernel accumulated into shared memory first.
    // This simpler version uses atomicAdd directly. For very large matrices,
    // a shared memory reduction within the block might be faster.
    // Perform atomicAdd on the float output buffer.
    atomicAdd(&mul[col], res);
}


__global__ void VecQuant4MatMulKernelFaster(
    const  half2* __restrict__ vec, // Input vector (FP16 pairs)
    const    int* __restrict__ mat, // Packed weights (INT32)
           float* __restrict__ mul, // Output vector (FP32)
    const  float* __restrict__ scales, // FP32 scales
    const  float* __restrict__ zeros,  // FP32 zero points * scales
    int height, // Packed height (orig_height / 8)
    int width
) {
    // Calculate the column index for this thread
    const int col = blockIdx.y * BLOCKWIDTH_4BIT + threadIdx.x;

    // Bounds check for column
    if (col >= width) return;

    // Calculate the starting row index for this block
    const int row_base = blockIdx.x * BLOCKHEIGHT_4BIT;

    // Load scale and zero for the current column as half2
    // Note: We negate zero here because we use FMA: val * scale + (-zero)
    const half2 scale_h2 = __float2half2_rn(scales[col]);
    const half2 zero_h2  = __float2half2_rn(-zeros[col]); // Negated zero_point * scale

    // Accumulator for the result (using float for better precision)
    float res = 0.0f;

    // Loop over the packed rows assigned to this block
    for (int packed_row_offset = 0; packed_row_offset < BLOCKHEIGHT_4BIT; ++packed_row_offset) {
        const int packed_row = row_base + packed_row_offset;
        // Bounds check for row
        if (packed_row >= height) continue;

        // Base row in the original un-packed matrix, divided by 2 for half2 indexing
        const int original_row_base_div2 = (packed_row * 8) / 2;

        // Load the packed 32-bit integer containing 8 x 4-bit weights
        const unsigned int packed_val = as_unsigned(mat[packed_row * width + col]);

        // Process 4 pairs of 4-bit values (4 * half2)
        #pragma unroll
        for (int i = 0; i < 4; ++i) { // i iterates over half2 elements
            // Extract two 4-bit values (8 bits total)
            unsigned int two_vals = (packed_val >> (i * 8)) & 0xFF;

            // Convert the two 4-bit integers to two half-precision floats
            half val_lo = __int2half_rn(two_vals & 0xF);        // Lower 4 bits
            half val_hi = __int2half_rn((two_vals >> 4) & 0xF); // Upper 4 bits
            half2 vals_h2 = __halves2half2(val_lo, val_hi);

            // Load corresponding pair of vector elements
            // Ensure we don't read past the end of the input vector (in half2 units)
            // int vec_idx_h2 = original_row_base_div2 + i;
            // if (vec_idx_h2 < total_input_features / 2) { // Need total_input_features
                 half2 vec_h2 = vec[original_row_base_div2 + i];

                 // Dequantize and multiply using fused multiply-add (FMA)
                 // dequant = vals_h2 * scale_h2 + (-zero_h2)
                 half2 dequant_h2 = __hfma2(vals_h2, scale_h2, zero_h2);

                 // Multiply dequantized weights with vector elements and accumulate
                 // res += dequant_h2.x * vec_h2.x + dequant_h2.y * vec_h2.y;
                 // Manual summation as __hfma2_sum might not be available/standard
                 res += __half2float(dequant_h2.x) * __half2float(vec_h2.x);
                 res += __half2float(dequant_h2.y) * __half2float(vec_h2.y);
            // }
        }
    }

    // Atomically add the partial result (float) to the output vector element (float)
    atomicAdd(&mul[col], res);
}
