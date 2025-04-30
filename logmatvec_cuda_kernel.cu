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


// Helper function to unpack two 4-bit codes (sign+exponent) from an int8 byte
__device__ inline void unpack_4bit_codes(const int8_t packed_byte, uint8_t& code1, uint8_t& code2) {
    // Cast to unsigned char to avoid sign extension issues with right shift
    uint8_t unsigned_byte = static_cast<uint8_t>(packed_byte);
    code1 = (unsigned_byte >> 4) & 0x0F; // High nibble (first weight code)
    code2 = unsigned_byte & 0x0F;        // Low nibble (second weight code)
}

// Kernel for Log-Quantized Matrix (W) x Linear-Quantized Vector (a)
// Uses bundled packed 4-bit codes (1 sign + 3 exponent)
__global__ void LogMatVecKernelPacked4bit(
    const int* __restrict__ a_quant,          // Quantized activations [InFeatures]
    const int8_t* __restrict__ w_packed_4bit, // Packed 4-bit codes [OutFeatures * InFeatures/2]
    float* __restrict__ output,               // Output vector [OutFeatures]
    const float delta_lsb,                   // Activation scaling factor
    const int min_exp,                       // Minimum exponent value for unmapping (maps to exp_map=0)
    const int in_features,
    const int out_features
) {
    // Each block computes one output feature
    const int output_row = blockIdx.x;

    if (output_row >= out_features) {
        return;
    }

    // Accumulator for the dot product (use 64-bit to prevent overflow)
    int64_t accumulator = 0;

    // Each thread sums a portion of the dot product, processing two weights at a time
    // The loop iterates over packed exponent bytes
    int packed_in_features = in_features / 2; // Number of packed bytes per row
    for (int packed_idx = threadIdx.x; packed_idx < packed_in_features; packed_idx += blockDim.x) {
        // Calculate index for packed exponent matrix (row-major)
        int packed_weight_idx = output_row * packed_in_features + packed_idx;
        // Calculate base index for signs and activations (corresponding to first weight in pair)
        int base_idx = packed_idx * 2;

        // Calculate index for packed weight matrix (row-major)
        // int packed_weight_idx = output_row * packed_in_features + packed_idx; // Duplicate removed
        // Calculate base index for activations (corresponding to first weight in pair)
        // int base_idx = packed_idx * 2; // Duplicate removed

        // Read packed byte and unpack 4-bit codes
        int8_t packed_byte = w_packed_4bit[packed_weight_idx];
        uint8_t code1, code2;
        unpack_4bit_codes(packed_byte, code1, code2);

        // Read corresponding activations
        int activation1 = a_quant[base_idx];
        int activation2 = a_quant[base_idx + 1];

        // --- Process first weight (code1) ---
        if (code1 != 0) { // Check for special zero code
            // Decode sign and exponent map
            signed char sign1 = (code1 & 0x08) ? -1 : 1; // MSB (bit 3) is sign (1=neg, 0=pos)
            uint8_t exp_map1 = code1 & 0x07; // 3 LSBs are exponent map
            // Unmap exponent (handle the offset for positive values)
            // Positive codes 1-7 map to exp_map 0-6 -> exponents min_exp to max_exp-1
            // Negative codes 8-15 map to exp_map 0-7 -> exponents min_exp to max_exp
            int exponent1;
            if (sign1 > 0) { // Positive (codes 1-7 map to exp_map 0-6)
                 exponent1 = static_cast<int>(exp_map1 - 1) + min_exp; // Map 1->min_exp, 7->max_exp-1
            } else { // Negative (codes 8-15 map to exp_map 0-7)
                 exponent1 = static_cast<int>(exp_map1) + min_exp; // Map 0->min_exp, 7->max_exp
            }
            accumulate_bitshift(activation1, exponent1, sign1, &accumulator);
        }

        // --- Process second weight (code2) ---
        if (code2 != 0) { // Check for special zero code
            // Decode sign and exponent map
            signed char sign2 = (code2 & 0x08) ? -1 : 1;
            uint8_t exp_map2 = code2 & 0x07;
            // Unmap exponent
            int exponent2;
             if (sign2 > 0) {
                 exponent2 = static_cast<int>(exp_map2 - 1) + min_exp;
            } else {
                 exponent2 = static_cast<int>(exp_map2) + min_exp;
            }
            accumulate_bitshift(activation2, exponent2, sign2, &accumulator);
        }
    }

    // --- Block-level reduction using shared memory (remains the same) ---
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
    const int8_t* w_packed_4bit, // Changed parameter name
    float* output,
    const float delta_lsb,
    const int min_exp,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size,
    // const int min_exp, // Remove from the end
    cudaStream_t stream)
{
    // Launch the __global__ kernel for packed 4-bit codes
    LogMatVecKernelPacked4bit<<<blocks, threads, shared_mem_size, stream>>>(
        a_quant,
        w_packed_4bit, // Pass packed codes
        output,
        delta_lsb,
        min_exp,       // Pass min_exp for unmapping
        in_features,
        out_features
    );
}


// ============================================================================
// Kernel Version using Bundled 4-bit Codes but Float Multiplication
// ============================================================================

// Kernel for Log-Quantized Matrix (W) x Linear-Quantized Vector (a)
// Uses bundled packed 4-bit codes BUT performs float multiplication internally
__global__ void LogMatVecKernelBundled4bit_FloatMul( // New kernel name
    const int* __restrict__ a_quant,          // Quantized activations [InFeatures]
    const int8_t* __restrict__ w_packed_4bit, // Packed 4-bit codes [OutFeatures * InFeatures/2]
    float* __restrict__ output,               // Output vector [OutFeatures]
    const float delta_lsb,                   // Activation scaling factor
    const int min_exp,                       // Minimum exponent value for unmapping
    const int in_features,
    const int out_features
) {
    // Each block computes one output feature
    const int output_row = blockIdx.x;

    if (output_row >= out_features) {
        return;
    }

    // Accumulator for the dot product (use double for float accumulation precision)
    double accumulator = 0.0;

    // Each thread sums a portion of the dot product, processing two weights at a time
    int packed_in_features = in_features / 2;
    for (int packed_idx = threadIdx.x; packed_idx < packed_in_features; packed_idx += blockDim.x) {
        int packed_weight_idx = output_row * packed_in_features + packed_idx;
        int base_idx = packed_idx * 2;

        // Read packed byte and unpack 4-bit codes
        int8_t packed_byte = w_packed_4bit[packed_weight_idx];
        uint8_t code1, code2;
        unpack_4bit_codes(packed_byte, code1, code2); // Use the same unpacker

        // Read corresponding activations
        int activation1_int = a_quant[base_idx];
        int activation2_int = a_quant[base_idx + 1];

        // --- Process first weight (code1) using float multiplication ---
        if (code1 != 0) { // Check for special zero code
            // Decode sign and exponent map
            signed char sign1_char = (code1 & 0x08) ? -1 : 1;
            uint8_t exp_map1 = code1 & 0x07;
            // Unmap exponent
            int exponent1;
            if (sign1_char > 0) { exponent1 = static_cast<int>(exp_map1 - 1) + min_exp; }
            else { exponent1 = static_cast<int>(exp_map1) + min_exp; }

            // Calculate float weight value
            float weight1_float = powf(2.0f, static_cast<float>(exponent1));
            // Multiply float weight by activation (cast activation to double)
            double term1 = static_cast<double>(activation1_int) * static_cast<double>(weight1_float);
            accumulator += (sign1_char > 0) ? term1 : -term1;
        }

        // --- Process second weight (code2) using float multiplication ---
        if (code2 != 0) { // Check for special zero code
            // Decode sign and exponent map
            signed char sign2_char = (code2 & 0x08) ? -1 : 1;
            uint8_t exp_map2 = code2 & 0x07;
            // Unmap exponent
            int exponent2;
            if (sign2_char > 0) { exponent2 = static_cast<int>(exp_map2 - 1) + min_exp; }
            else { exponent2 = static_cast<int>(exp_map2) + min_exp; }

            // Calculate float weight value
            float weight2_float = powf(2.0f, static_cast<float>(exponent2));
            // Multiply float weight by activation
            double term2 = static_cast<double>(activation2_int) * static_cast<double>(weight2_float);
            accumulator += (sign2_char > 0) ? term2 : -term2;
        }
    }

    // --- Block-level reduction using shared memory (using double) ---
    extern __shared__ double sdata_double[]; // Use double for shared memory
    sdata_double[threadIdx.x] = accumulator;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sdata_double[threadIdx.x] += sdata_double[threadIdx.x + offset];
        }
        __syncthreads();
    }

    // Lead thread writes the final scaled result
    if (threadIdx.x == 0) {
        // Cast the final double result to float before scaling
        output[output_row] = static_cast<float>(sdata_double[0]) * delta_lsb;
    }
}


// --- Kernel Launcher for Bundled 4-bit Float Multiplication Version ---
void LogMatVecKernelLauncher_BundledFloatMul( // New launcher name
    const int* a_quant,
    const int8_t* w_packed_4bit,
    float* output,
    const float delta_lsb,
    const int min_exp,
    const int in_features,
    const int out_features,
    const dim3 blocks,
    const dim3 threads,
    const size_t shared_mem_size, // Base size, will be adjusted
    cudaStream_t stream)
{
    // Adjust shared memory size for double accumulator
    size_t shared_mem_double_size = threads.x * sizeof(double);

    // Launch the __global__ kernel for bundled 4-bit with float multiplication
    LogMatVecKernelBundled4bit_FloatMul<<<blocks, threads, shared_mem_double_size, stream>>>( // Call new kernel
        a_quant,
        w_packed_4bit,
        output,
        delta_lsb,
        min_exp,
        in_features,
        out_features
    );
}
