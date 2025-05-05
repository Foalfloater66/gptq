#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

// Declare the functions defined in quant_cuda_kernel_4bit.cu
void vecquant4matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

// Define wrapper functions to handle device context and call CUDA kernels
void vecquant4matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  TORCH_CHECK(vec.is_cuda(), "vec must be a CUDA tensor");
  TORCH_CHECK(mat.is_cuda(), "mat must be a CUDA tensor");
  TORCH_CHECK(mul.is_cuda(), "mul must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");
  TORCH_CHECK(mat.scalar_type() == torch::kInt32, "mat must be an Int32 tensor");
  // Enforce float32 output for the standard kernel, as it now always accumulates in float
  TORCH_CHECK(mul.scalar_type() == torch::kFloat32, "mul tensor must be Float32 for vecquant4matmul");
  // Enforce float or half input for the standard kernel
  TORCH_CHECK(vec.scalar_type() == torch::kFloat32 || vec.scalar_type() == torch::kFloat16,
              "vec tensor must be Float32 or Float16 for vecquant4matmul");
  // Add more shape/dimension checks if needed
  vecquant4matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  TORCH_CHECK(vec.is_cuda(), "vec must be a CUDA tensor");
  TORCH_CHECK(mat.is_cuda(), "mat must be a CUDA tensor");
  TORCH_CHECK(mul.is_cuda(), "mul must be a CUDA tensor");
  TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
  TORCH_CHECK(zeros.is_cuda(), "zeros must be a CUDA tensor");
  TORCH_CHECK(mat.scalar_type() == torch::kInt32, "mat must be an Int32 tensor");
  // Add more shape/dimension checks if needed
  vecquant4matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

// Define the Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4matmul", &vecquant4matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant4matmul_faster", &vecquant4matmul_faster, "Vector 4-bit Quantized Matrix Multiplication (CUDA), faster version for FP16 input");
}
