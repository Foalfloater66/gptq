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
    const   half* __restrict__ vec,
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

// --- CUDA Function Implementations ---

void vecquant4matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0); // Packed height (orig_height / 8)
  int width = mat.size(1);  // Output features

  // Basic dimension checks - more robust checks could be added
  TORCH_CHECK(vec.dim() >= 1, "vec must have at least 1 dimension");
  TORCH_CHECK(mat.dim() == 2, "mat must be 2-dimensional");
  TORCH_CHECK(mul.dim() >= 1, "mul must have at least 1 dimension");
  TORCH_CHECK(scales.dim() >= 1, "scales must have at least 1 dimension");
  TORCH_CHECK(zeros.dim() >= 1, "zeros must have at least 1 dimension");

  // Check feature dimension consistency
  int vec_features = vec.size(-1); // Last dimension of vec is input features
  int expected_packed_height = (vec_features + 7) / 8; // ceil(vec_features / 8)
  TORCH_CHECK(height == expected_packed_height, "Packed matrix height (mat.size(0)) does not match expected packed height based on vec features");
  TORCH_CHECK(width == mul.size(-1), "Matrix width (mat.size(1)) must match output features (mul.size(-1))");
  TORCH_CHECK(width == scales.size(-1), "Matrix width (mat.size(1)) must match scales features (scales.size(-1))");
  TORCH_CHECK(width == zeros.size(-1), "Matrix width (mat.size(1)) must match zeros features (zeros.size(-1))");


  // Grid dimensions
  dim3 blocks(
    (height + BLOCKHEIGHT_4BIT - 1) / BLOCKHEIGHT_4BIT, // Number of blocks in Y dimension (for rows)
    (width + BLOCKWIDTH_4BIT - 1) / BLOCKWIDTH_4BIT    // Number of blocks in X dimension (for columns)
  );
  // Block dimensions
  dim3 threads(BLOCKWIDTH_4BIT); // Threads per block

  // Dispatch based on input type, but output `mul` must be float.
  TORCH_CHECK(mul.scalar_type() == torch::kFloat32, "mul tensor must be Float32 for vecquant4matmul_cuda");

  // Explicitly call float or half kernel based on input type
  if (vec.scalar_type() == torch::kFloat32) {
      // Ensure scales/zeros are also float
      auto scales_f32 = scales.to(torch::kFloat32);
      auto zeros_f32 = zeros.to(torch::kFloat32);
      VecQuant4MatMulKernel_float<<<blocks, threads>>>(
          vec.data_ptr<float>(),
          mat.data_ptr<int>(),
          mul.data_ptr<float>(),
          scales_f32.data_ptr<float>(),
          zeros_f32.data_ptr<float>(),
          height, width
      );
  } else if (vec.scalar_type() == torch::kFloat16) {
      // Ensure scales/zeros are also half
      auto scales_f16 = scales.to(torch::kFloat16);
      auto zeros_f16 = zeros.to(torch::kFloat16);
      VecQuant4MatMulKernel_half<<<blocks, threads>>>(
          vec.data_ptr<half>(),
          mat.data_ptr<int>(),
          mul.data_ptr<float>(), // Output is still float
          scales_f16.data_ptr<half>(),
          zeros_f16.data_ptr<half>(),
          height, width
      );
  } else {
      TORCH_CHECK(false, "vecquant4matmul_cuda supports only Float32 or Float16 input types");
  }

  C10_CUDA_KERNEL_LAUNCH_CHECK(); // Check for kernel launch errors
}

void vecquant4matmul_faster_cuda(
  torch::Tensor vec, // FP16
  torch::Tensor mat, // INT32
  torch::Tensor mul, // FP32
  torch::Tensor scales, // FP32
  torch::Tensor zeros // FP32
) {
  int height = mat.size(0); // Packed height (orig_height / 8)
  int width = mat.size(1);  // Output features

  TORCH_CHECK(vec.scalar_type() == torch::kFloat16, "vec must be FP16 for faster kernel");
  TORCH_CHECK(mul.scalar_type() == torch::kFloat32, "mul must be FP32 for faster kernel");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be FP32 for faster kernel");
  TORCH_CHECK(zeros.scalar_type() == torch::kFloat32, "zeros must be FP32 for faster kernel");

  // Basic dimension checks
  TORCH_CHECK(vec.dim() >= 1, "vec must have at least 1 dimension");
  TORCH_CHECK(mat.dim() == 2, "mat must be 2-dimensional");
  TORCH_CHECK(mul.dim() >= 1, "mul must have at least 1 dimension");
  TORCH_CHECK(scales.dim() >= 1, "scales must have at least 1 dimension");
  TORCH_CHECK(zeros.dim() >= 1, "zeros must have at least 1 dimension");

  // Check feature dimension consistency
  int vec_features = vec.size(-1); // Last dimension of vec is input features
  TORCH_CHECK(vec_features % 2 == 0, "vec features must be divisible by 2 for half2 access");
  int expected_packed_height = (vec_features + 7) / 8; // ceil(vec_features / 8)
  TORCH_CHECK(height == expected_packed_height, "Packed matrix height (mat.size(0)) does not match expected packed height based on vec features");
  TORCH_CHECK(width == mul.size(-1), "Matrix width (mat.size(1)) must match output features (mul.size(-1))");
  TORCH_CHECK(width == scales.size(-1), "Matrix width (mat.size(1)) must match scales features (scales.size(-1))");
  TORCH_CHECK(width == zeros.size(-1), "Matrix width (mat.size(1)) must match zeros features (zeros.size(-1))");


  // Grid dimensions
  dim3 blocks(
    (height + BLOCKHEIGHT_4BIT - 1) / BLOCKHEIGHT_4BIT, // Number of blocks in Y dimension (for rows)
    (width + BLOCKWIDTH_4BIT - 1) / BLOCKWIDTH_4BIT    // Number of blocks in X dimension (for columns)
  );
  // Block dimensions
  dim3 threads(BLOCKWIDTH_4BIT); // Threads per block

  VecQuant4MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(), // Cast to half2 pointer
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<float>(),
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
    const   half* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul, // Output buffer MUST be float
    const   half* __restrict__ scales,
    const   half* __restrict__ zeros, // zero_point * scale
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
    const half scale = scales[col];
    const half zero = zeros[col]; // This is zero_point * scale

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
            // Extract the i-th 4-bit value and convert to half
            half val = __int2half_rn((packed_val >> (i * 4)) & 0xF);
            // Dequantize: scale * quantized_value - zero_point * scale (in half precision)
            half dequant_val_h = scale * val - zero;
            // Multiply with corresponding vector element (half) and accumulate in float
            // Ensure we don't read past the end of the input vector if padding occurred
            // Note: The Python code pads the input vector, so this check might be redundant
            // if padding is guaranteed, but it's safer.
            // int vec_idx = original_row_base + i;
            // if (vec_idx < total_input_features) { // Need total_input_features passed or calculated
                 res += __half2float(dequant_val_h) * __half2float(vec[original_row_base + i]);
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
