#include <torch/extension.h>
#include <vector>
#include <cstdint> // For int64_t and int8_t
#include <cuda_runtime.h> // Include CUDA runtime headers for dim3, cudaStream_t
#include <c10/cuda/CUDAStream.h> // Include for getCurrentCUDAStream

#include <cuda_fp16.h> // Include for __half type

// Forward declaration of the CUDA kernel launcher function (defined in .cu file)
// Updated signature: Takes float/half activations, bias, act_scale, act_bits
template<typename T_ACT> // Template for activation type (float or half)
void LogMatVecKernelLauncher(
    const T_ACT* x,               // Activations (float or half)
    const int8_t* w_packed_4bit,  // Packed weights
    const float* bias,            // Bias tensor
    float* output,                // Output tensor (float32)
    const float act_scale,        // Activation scale (delta_lsb)
    const int min_exp,            // Weight min exponent
    const int act_bits,           // Activation bits
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
// Updated signature: Takes float/half activations, bias, act_scale, act_bits
torch::Tensor logmatvec_forward_packed4bit(
    torch::Tensor x,                // Activations (Float32 or Half)
    torch::Tensor w_packed_4bit,    // Packed 4-bit codes (Int8)
    torch::Tensor bias,             // Bias tensor (Float32 or Half)
    torch::Tensor output,           // Pre-allocated output tensor (Float32)
    double act_scale,               // Activation scaling factor
    int min_exp,                    // Minimum exponent for weights
    int act_bits                    // Activation bits
) {
    // Input validation
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(w_packed_4bit.device().is_cuda(), "w_packed_4bit must be a CUDA tensor");
    TORCH_CHECK(bias.device().is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(w_packed_4bit.is_contiguous(), "w_packed_4bit must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    // Check activation dtype (Float32 or Half)
    TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kHalf, "x must be Float32 or Half");
    TORCH_CHECK(w_packed_4bit.dtype() == torch::kInt8, "w_packed_4bit must be Int8");
    // Bias should ideally match activation type, but kernel expects float bias input for now
    TORCH_CHECK(bias.dtype() == torch::kFloat32 || bias.dtype() == torch::kHalf, "bias must be Float32 or Half");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be Float32");

    TORCH_CHECK(w_packed_4bit.dim() == 2, "w_packed_4bit must be 2D");
    TORCH_CHECK(x.dim() == 1, "x must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(output.dim() == 1, "output must be 1D");

    const int out_features = w_packed_4bit.size(0);
    const int packed_in_features = w_packed_4bit.size(1);
    const int in_features = packed_in_features * 2;

    // Dimension checks
    TORCH_CHECK(x.size(0) == in_features, "x size mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias size mismatch");
    TORCH_CHECK(output.size(0) == out_features, "output size mismatch");

    // Kernel launch configuration
    const int threads_per_block = 256;
    // Define grid dimensions: One block per output feature
    const dim3 blocks(out_features);
    const dim3 threads(threads_per_block);

    // Calculate shared memory size needed for reduction (one int64_t per thread)
    const size_t shared_mem_size = threads_per_block * sizeof(int64_t);

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // --- Launch Kernel via the launcher function ---
    // Need to handle different activation types (float/half)
    if (x.dtype() == torch::kFloat32) {
        // Ensure bias is also float for the kernel
        auto bias_float = bias.to(torch::kFloat32);
        LogMatVecKernelLauncher<float>( // Explicitly instantiate template for float
            x.data_ptr<float>(),
            w_packed_4bit.data_ptr<int8_t>(),
            bias_float.data_ptr<float>(), // Pass float bias ptr
            output.data_ptr<float>(),
            static_cast<float>(act_scale),
            min_exp,
            act_bits,
            in_features,
            out_features,
            blocks,
            threads,
            shared_mem_size,
            stream
        );
    } else { // x.dtype() == torch::kHalf
        // Ensure bias is also float for the kernel (kernel expects float bias)
        auto bias_float = bias.to(torch::kFloat32);
        LogMatVecKernelLauncher<__half>( // Explicitly instantiate template for half
            reinterpret_cast<const __half*>(x.data_ptr()), // Cast half tensor ptr
            w_packed_4bit.data_ptr<int8_t>(),
            bias_float.data_ptr<float>(), // Pass float bias ptr
            output.data_ptr<float>(),
            static_cast<float>(act_scale),
            min_exp,
            act_bits,
            in_features,
            out_features,
            blocks,
            threads,
            shared_mem_size,
            stream
        );
    }
    // --- End Kernel Launch ---

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
        "LogNet Style Matrix-Vector Forward Pass with Packed 4-bit Weights (Bit Shift, Internal Act Quant & Bias Add) (CUDA)" // Updated Docstring
    );

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
