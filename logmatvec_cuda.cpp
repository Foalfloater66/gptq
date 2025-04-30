#include <torch/extension.h>
#include <vector>
#include <cstdint> // For int64_t and int8_t

// Forward declaration of the CUDA kernel launcher function (defined in .cu file)
void LogMatVecKernelLauncher(
    const int* a_quant,
    const int* w_exp,
    const signed char* w_sign, // Use signed char explicitly
    float* output,
    const float delta_lsb,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size,
    cudaStream_t stream);


// C++ wrapper function callable from Python
torch::Tensor logmatvec_forward(
    torch::Tensor a_quant,      // Quantized activations (Int32)
    torch::Tensor w_exp,        // Weight exponents (Int32)
    torch::Tensor w_sign,       // Weight signs (Int8/Char)
    torch::Tensor output,       // Pre-allocated output tensor (Float32)
    double delta_lsb            // Activation scaling factor (double for precision from Python)
) {
    // Input validation
    TORCH_CHECK(a_quant.device().is_cuda(), "a_quant must be a CUDA tensor");
    TORCH_CHECK(w_exp.device().is_cuda(), "w_exp must be a CUDA tensor");
    TORCH_CHECK(w_sign.device().is_cuda(), "w_sign must be a CUDA tensor");
    TORCH_CHECK(output.device().is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(a_quant.is_contiguous(), "a_quant must be contiguous");
    TORCH_CHECK(w_exp.is_contiguous(), "w_exp must be contiguous");
    TORCH_CHECK(w_sign.is_contiguous(), "w_sign must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    TORCH_CHECK(a_quant.dtype() == torch::kInt32, "a_quant must be Int32");
    TORCH_CHECK(w_exp.dtype() == torch::kInt32, "w_exp must be Int32");
    TORCH_CHECK(w_sign.dtype() == torch::kInt8, "w_sign must be Int8"); // Check for kInt8
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output must be Float32");

    TORCH_CHECK(w_exp.dim() == 2, "w_exp must be 2D");
    TORCH_CHECK(w_sign.dim() == 2, "w_sign must be 2D");
    TORCH_CHECK(a_quant.dim() == 1, "a_quant must be 1D");
    TORCH_CHECK(output.dim() == 1, "output must be 1D");

    const int out_features = w_exp.size(0);
    const int in_features = w_exp.size(1);

    TORCH_CHECK(w_sign.size(0) == out_features, "w_sign dim 0 mismatch");
    TORCH_CHECK(w_sign.size(1) == in_features, "w_sign dim 1 mismatch");
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

    // Launch Kernel via the launcher function
    LogMatVecKernelLauncher(
        a_quant.data_ptr<int>(),
        w_exp.data_ptr<int>(),
        // PyTorch kInt8 corresponds to int8_t, cast to signed char* for the kernel
        reinterpret_cast<const signed char*>(w_sign.data_ptr<int8_t>()),
        output.data_ptr<float>(),
        static_cast<float>(delta_lsb), // Cast double to float for the kernel
        in_features,
        out_features,
        blocks,
        threads,
        shared_mem_size,
        stream // Pass the stream
    );

    // Check for CUDA errors after kernel launch (optional but recommended for debugging)
    // cudaError_t err = cudaGetLastError();
    // TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Boilerplate for Python binding using Pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", // Python function name
        &logmatvec_forward, // C++ function pointer
        "LogNet Style Matrix-Vector Forward Pass (CUDA)" // Docstring
    );
}
