#include <torch/extension.h>
#include <vector>
#include <cstdint> // For int64_t and int8_t
#include <cuda_runtime.h> // Include CUDA runtime headers for dim3, cudaStream_t
#include <c10/cuda/CUDAStream.h> // Include for getCurrentCUDAStream

// Forward declaration of the CUDA kernel launcher function (defined in .cu file)
// Updated for bundled packed 4-bit codes
void LogMatVecKernelLauncher(
    const int* a_quant,
    const int8_t* w_packed_4bit, // Changed parameter
    float* output,
    const float delta_lsb,
    const int min_exp,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size,
    cudaStream_t stream);

// Forward declaration for the Bundled Float Multiplication kernel launcher
void LogMatVecKernelLauncher_BundledFloatMul(
    const int* a_quant,
    const int8_t* w_packed_4bit,
    float* output,
    const float delta_lsb,
    const int min_exp,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size,
    cudaStream_t stream);


// C++ wrapper function callable from Python - Bundled packed 4-bit (Bit Shift Version)
torch::Tensor logmatvec_forward_packed4bit(
    torch::Tensor a_quant,          // Quantized activations (Int32)
    torch::Tensor w_packed_4bit,    // Packed 4-bit codes (Int8)
    torch::Tensor output,           // Pre-allocated output tensor (Float32)
    double delta_lsb,               // Activation scaling factor
    int min_exp                     // Minimum exponent for unmapping
) {
    // Input validation
    TORCH_CHECK(a_quant.device().is_cuda(), "a_quant must be a CUDA tensor");
    TORCH_CHECK(w_packed_4bit.device().is_cuda(), "w_packed_4bit must be a CUDA tensor"); // Check packed tensor
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(a_quant.is_contiguous(), "a_quant must be contiguous");
    TORCH_CHECK(w_packed_4bit.is_contiguous(), "w_packed_4bit must be contiguous"); // Check packed tensor
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(a_quant.dtype() == torch::kInt32, "a_quant must be Int32");
    TORCH_CHECK(w_packed_4bit.dtype() == torch::kInt8, "w_packed_4bit must be Int8"); // Check packed tensor type
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be Float32");

    TORCH_CHECK(w_packed_4bit.dim() == 2, "w_packed_4bit must be 2D"); // Packed tensor shape
    TORCH_CHECK(a_quant.dim() == 1, "a_quant must be 1D");
    TORCH_CHECK(output.dim() == 1, "output must be 1D");

    const int out_features = w_packed_4bit.size(0); // Get dimensions from packed tensor
    const int packed_in_features = w_packed_4bit.size(1);
    const int in_features = packed_in_features * 2; // Infer original in_features

    // Dimension checks
    TORCH_CHECK(a_quant.size(0) == in_features, "a_quant size mismatch");
    TORCH_CHECK(output.size(0) == out_features, "output size mismatch");

    // Kernel launch configuration
    // Define threads per block (tune this value, e.g., 128, 256, 512)
    const int threads_per_block = 256;
    // Define grid dimensions: One block per output feature
    const dim3 blocks(out_features);
    const dim3 threads(threads_per_block);

    // Calculate shared memory size needed for reduction (one int64_t per thread)
    const size_t shared_mem_size = threads_per_block * sizeof(int64_t);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch Kernel via the launcher function - Updated call
    LogMatVecKernelLauncher(
        a_quant.data_ptr<int>(),
        w_packed_4bit.data_ptr<int8_t>(), // Pass packed codes ptr
        output.data_ptr<float>(),
        static_cast<float>(delta_lsb),
        min_exp,
        in_features,
        out_features,
        blocks,
        threads,
        shared_mem_size,
        stream
    );

    // Check for CUDA errors after kernel launch (optional but recommended for debugging)
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

#if 0 // Removed stray code block causing errors
// Boilerplate for Python binding using Pybind11 - Updated function name
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_packed4bit", // New Python function name
        &logmatvec_forward_packed4bit, // C++ function pointer
        "LogNet Style Matrix-Vector Forward Pass with Packed 4-bit Exponents (CUDA)" // Docstring
    );

    // Check for CUDA errors after kernel launch (optional but recommended for debugging)
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}
#endif


// C++ wrapper function callable from Python - Bundled packed 4-bit (Float Mul Version)
torch::Tensor logmatvec_forward_bundled4bit_floatmul( // New function name
    torch::Tensor a_quant,          // Quantized activations (Int32)
    torch::Tensor w_packed_4bit,    // Packed 4-bit codes (Int8)
    torch::Tensor output,           // Pre-allocated output tensor (Float32)
    double delta_lsb,               // Activation scaling factor
    int min_exp                     // Minimum exponent for unmapping
) {
    // --- Input validation (Identical to bit shift version) ---
    TORCH_CHECK(a_quant.device().is_cuda(), "a_quant must be a CUDA tensor");
    TORCH_CHECK(w_packed_4bit.device().is_cuda(), "w_packed_4bit must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(a_quant.is_contiguous(), "a_quant must be contiguous");
    TORCH_CHECK(w_packed_4bit.is_contiguous(), "w_packed_4bit must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(a_quant.dtype() == torch::kInt32, "a_quant must be Int32");
    TORCH_CHECK(w_packed_4bit.dtype() == torch::kInt8, "w_packed_4bit must be Int8");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be Float32");

    TORCH_CHECK(w_packed_4bit.dim() == 2, "w_packed_4bit must be 2D");
    TORCH_CHECK(a_quant.dim() == 1, "a_quant must be 1D");
    TORCH_CHECK(output.dim() == 1, "output must be 1D");

    const int out_features = w_packed_4bit.size(0);
    const int packed_in_features = w_packed_4bit.size(1);
    const int in_features = packed_in_features * 2;

    TORCH_CHECK(a_quant.size(0) == in_features, "a_quant size mismatch");
    TORCH_CHECK(output.size(0) == out_features, "output size mismatch");
    // --- End Input Validation ---

    // Kernel launch configuration (Identical)
    const int threads_per_block = 256;
    const dim3 blocks(out_features);
    const dim3 threads(threads_per_block);

    // Shared memory size (pass base size, launcher adjusts)
    const size_t shared_mem_size = threads_per_block * sizeof(double); // Base on double

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch the Bundled Float Multiplication Kernel via its launcher
    LogMatVecKernelLauncher_BundledFloatMul( // Call the new launcher
        a_quant.data_ptr<int>(),
        w_packed_4bit.data_ptr<int8_t>(),
        output.data_ptr<float>(),
        static_cast<float>(delta_lsb),
        min_exp,
        in_features,
        out_features,
        blocks,
        threads,
        shared_mem_size, // Pass base size
        stream
    );

    // Check for CUDA errors after kernel launch (optional but recommended for debugging)
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}


// Boilerplate for Python binding using Pybind11 - Add new function
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_packed4bit", // Bit shift version
        &logmatvec_forward_packed4bit,
        "LogNet Style Matrix-Vector Forward Pass with Bundled Packed 4-bit Weights (Bit Shift) (CUDA)" // Updated Docstring
    );
    m.def(
        "forward_bundled4bit_floatmul", // Float mul version
        &logmatvec_forward_bundled4bit_floatmul,
        "LogNet Style Matrix-Vector Forward Pass with Bundled Packed 4-bit Weights (Float Mul) (CUDA)" // Docstring
    );
}
