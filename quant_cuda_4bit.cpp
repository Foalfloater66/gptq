#include <torch/all.h>
#include <torch/python.h>
#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

#include <c10/cuda/CUDAStream.h> // Include for getCurrentCUDAStream

// Declare the CUDA kernel launcher functions (modified to accept pointers)
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
);

void vecquant4matmul_faster_launcher( // Renamed to launcher
    const c10::Half* vec_ptr, // Use c10::Half type for consistency with PyTorch
    const int* mat_ptr,
    float* mul_ptr,
    const float* scales_ptr,  // Expects float*
    const float* zeros_ptr,   // Expects float*
    int height,
    int width,
    cudaStream_t stream
);

// Define wrapper functions to handle device context and call CUDA kernels
void vecquant4matmul(
  torch::Tensor vec, // Input: float or half
  torch::Tensor mat, // Input: int32
  torch::Tensor mul, // Input: float32 (Output)
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  TORCH_CHECK(vec.is_cuda(), "vec must be a CUDA tensor");
  TORCH_CHECK(mat.is_cuda(), "mat must be a CUDA tensor");
  TORCH_CHECK(mul.is_cuda(), "mul must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");
  TORCH_CHECK(mat.scalar_type() == torch::kInt32, "mat must be an Int32 tensor");
  TORCH_CHECK(mul.scalar_type() == torch::kFloat32, "mul tensor must be Float32 for vecquant4matmul output");
  TORCH_CHECK(vec.scalar_type() == torch::kFloat32 || vec.scalar_type() == torch::kFloat16,
              "vec tensor must be Float32 or Float16 for vecquant4matmul input");
  TORCH_CHECK(scales.scalar_type() == vec.scalar_type(), "scales dtype must match vec dtype");
  TORCH_CHECK(zeros.scalar_type() == vec.scalar_type(), "zeros dtype must match vec dtype");

  TORCH_CHECK(vec.is_contiguous(), "vec must be contiguous");
  TORCH_CHECK(mat.is_contiguous(), "mat must be contiguous");
  TORCH_CHECK(mul.is_contiguous(), "mul must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(zeros.is_contiguous(), "zeros must be contiguous");

  // Add dimension checks similar to logmatvec
  const int height = mat.size(0); // Packed height
  const int width = mat.size(1);  // Output features
  // Check dimensions against vec and mul
  const int vec_features = vec.size(-1);
  const int expected_packed_height = (vec_features + 7) / 8;
  TORCH_CHECK(height == expected_packed_height, "Packed matrix height mismatch");
  TORCH_CHECK(width == mul.size(-1), "Matrix width vs mul features mismatch");
  TORCH_CHECK(width == scales.size(-1), "Matrix width vs scales features mismatch");
  TORCH_CHECK(width == zeros.size(-1), "Matrix width vs zeros features mismatch");

  // Get current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Call the launcher with raw pointers
  vecquant4matmul_launcher(
      vec.data_ptr(), // Pass raw pointer (void*)
      mat.data_ptr<int>(),
      mul.data_ptr<float>(),
      scales.data_ptr(), // Pass raw pointer (void*)
      zeros.data_ptr(),  // Pass raw pointer (void*)
      vec.scalar_type(), // Pass the data type enum
      height,
      width,
      stream
  );
}

void vecquant4matmul_faster(
  torch::Tensor vec,    // Input: float16
  torch::Tensor mat,    // Input: int32
  torch::Tensor mul,    // Input: float32 (Output)
  torch::Tensor scales, // Input: float32
  torch::Tensor zeros   // Input: float32
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec)); // Keep guard for context
  TORCH_CHECK(vec.is_cuda(), "vec must be a CUDA tensor");
  TORCH_CHECK(mat.is_cuda(), "mat must be a CUDA tensor");
  TORCH_CHECK(mul.is_cuda(), "mul must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");

  // Enforce types for faster kernel
  TORCH_CHECK(vec.scalar_type() == torch::kFloat16, "vec must be Float16 for faster kernel");
  TORCH_CHECK(mat.scalar_type() == torch::kInt32, "mat must be an Int32 tensor");
  TORCH_CHECK(mul.scalar_type() == torch::kFloat32, "mul must be Float32 for faster kernel output");
  TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be Float32 for faster kernel");
  TORCH_CHECK(zeros.scalar_type() == torch::kFloat32, "zeros must be Float32 for faster kernel");

  TORCH_CHECK(vec.is_contiguous(), "vec must be contiguous");
  TORCH_CHECK(mat.is_contiguous(), "mat must be contiguous");
  TORCH_CHECK(mul.is_contiguous(), "mul must be contiguous");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(zeros.is_contiguous(), "zeros must be contiguous");

  // Add dimension checks
  const int height = mat.size(0); // Packed height
  const int width = mat.size(1);  // Output features
  // Check dimensions against vec and mul
  const int vec_features = vec.size(-1);
  const int expected_packed_height = (vec_features + 7) / 8;
  TORCH_CHECK(vec_features % 2 == 0, "vec features must be divisible by 2 for faster kernel");
  TORCH_CHECK(height == expected_packed_height, "Packed matrix height mismatch");
  TORCH_CHECK(width == mul.size(-1), "Matrix width vs mul features mismatch");
  TORCH_CHECK(width == scales.size(-1), "Matrix width vs scales features mismatch");
  TORCH_CHECK(width == zeros.size(-1), "Matrix width vs zeros features mismatch");

  // Get current CUDA stream
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Call the faster launcher with raw pointers
  vecquant4matmul_faster_launcher(
      vec.data_ptr<c10::Half>(), // Pass c10::Half*
      mat.data_ptr<int>(),
      mul.data_ptr<float>(),
      scales.data_ptr<float>(),
      zeros.data_ptr<float>(),
      height,
      width,
      stream
  );
}

// Define the Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul_faster", &vecquant4matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version for FP16 input");
}
