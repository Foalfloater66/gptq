#include <cuda_runtime.h>
#include <stdint.h> // For int64_t
#include <algorithm> // For std::max, std::min

// Clamping helper (optional, if exponents aren't pre-clamped)
// __device__ inline int clamp_exp(int exp, int min_exp, int max_exp) {
//     return max(min_exp, min(exp, max_exp));
// }

// Core Bitshift Operation
// x: quantized activation (int)
// y: weight exponent (int) - Handles positive (left shift) and negative (right shift)
// z: weight sign (signed char: +1, -1, 0)
// accumulator: pointer to the 64-bit accumulator
__device__ inline void accumulate_bitshift(const int x, const int y, const signed char z, int64_t* accumulator) {
    if (z == 0 || x == 0) { // Handle zero weight or activation
        return;
    }

    int64_t shifted_val;
    // Use standard C++ casting for clarity
    int64_t x_64 = static_cast<int64_t>(x);

    if (y >= 0) {
        // Left shift for positive exponents
        // Basic overflow check: if y is too large, result is likely 0 or incorrect anyway
        // A more robust check might compare y against (63 - position of highest set bit in x)
        if (y < 64) { // Avoid shifting by 64 or more bits
           shifted_val = (x_64 << y);
        } else {
           shifted_val = 0; // Or handle as error/saturation
        }
    } else {
        // Right shift for negative exponents
        // Shifting by negative amount is UB in C++, use positive amount
        int shift_amount = -y;
        if (shift_amount < 64) { // Avoid shifting by 64 or more bits
            shifted_val = (x_64 >> shift_amount);
        } else {
            // If shifting by 64 or more, result depends on sign of x
            shifted_val = (x_64 < 0) ? -1 : 0;
        }
    }

    // Apply sign using conditional expression
    *accumulator += (z > 0) ? shifted_val : -shifted_val;
}


// Kernel for Log-Quantized Matrix (W) x Linear-Quantized Vector (a)
// Assumes W is OutFeatures x InFeatures, a is InFeatures x 1
__global__ void LogMatVecKernel(
    const int* __restrict__ a_quant,      // Quantized activations [InFeatures]
    const int* __restrict__ w_exp,        // Weight exponents [OutFeatures * InFeatures]
    const signed char* __restrict__ w_sign, // Weight signs [OutFeatures * InFeatures]
    float* __restrict__ output,           // Output vector [OutFeatures]
    const float delta_lsb,               // Activation scaling factor
    const int in_features,
    const int out_features
    // Optional: min_exp, max_exp if clamping needed inside kernel
) {
    // Each block computes one output feature
    const int output_row = blockIdx.x;

    if (output_row >= out_features) {
        return;
    }

    // Accumulator for the dot product (use 64-bit to prevent overflow)
    int64_t accumulator = 0;

    // Each thread sums a portion of the dot product
    // Simple loop strategy (can be optimized with shared memory for 'a_quant')
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        // Calculate index for weight matrix (assuming row-major)
        int weight_idx = output_row * in_features + i;

        int activation_val = a_quant[i];
        int exponent_val = w_exp[weight_idx];
        // Directly cast to signed char when reading
        signed char sign_val = w_sign[weight_idx];

        // --- Optional: Clamp exponent here if not pre-clamped ---
        // exponent_val = clamp_exp(exponent_val, min_exp_val, max_exp_val);

        accumulate_bitshift(activation_val, exponent_val, sign_val, &accumulator);
    }

    // --- Block-level reduction using shared memory ---
    // Allocate shared memory dynamically based on kernel launch parameter
    extern __shared__ int64_t sdata[];
    sdata[threadIdx.x] = accumulator;
    __syncthreads(); // Wait for all threads to write to shared memory

    // Reduce within the block (works for any blockDim.x)
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        // Only threads in the first half of the current range participate
        if (threadIdx.x < offset) {
            sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        }
        __syncthreads(); // Synchronize after each reduction step
    }

    // Lead thread (threadIdx.x == 0) writes the final scaled result
    if (threadIdx.x == 0) {
        // Cast the final 64-bit integer result to float before scaling
        output[output_row] = static_cast<float>(sdata[0]) * delta_lsb;
    }
}


// --- Kernel Launcher ---
// This function is defined in the .cu file and called by the .cpp file.
// It sets up the <<<...>>> kernel launch syntax.
void LogMatVecKernelLauncher(
    const int* a_quant,
    const int* w_exp,
    const signed char* w_sign,
    float* output,
    const float delta_lsb,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size,
    cudaStream_t stream)
{
    // Launch the __global__ kernel
    LogMatVecKernel<<<blocks, threads, shared_mem_size, stream>>>(
        a_quant,
        w_exp,
        w_sign,
        output,
        delta_lsb,
        in_features,
        out_features
    );
}
