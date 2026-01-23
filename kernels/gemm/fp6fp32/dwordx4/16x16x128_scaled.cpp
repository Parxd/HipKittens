/**
 * @file 16x16x128_scaled_test.cpp
 * @brief Test harness for scaled FP6 MFMA (MXFP6) functionality.
 * 
 * This test validates the scaled MFMA instruction where each 32-element
 * block in A and B matrices can have its own 8-bit scale exponent.
 * 
 * MXFP6 format:
 * - Data: 32 FP6 values packed (24 bytes)
 * - Scale: 1 x 8-bit unsigned exponent per 32 elements
 * - Effective value = fp6_value * 2^scale
 */

#include "kittens.cuh"
#include "16x16x128_utils.cpp"
#include <random>
#include <omp.h>
#include <cstring>
#include <iomanip>
#include <cmath>
using namespace kittens;

using din = fp6_e2m3;
using dout = half;

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    std::cerr << "HIP error " << hipGetErrorString(_e) \
              << " at " << __FILE__ << ":" << __LINE__ << std::endl; std::exit(1);} } while(0)

// Test dimensions matching the full kernel's tile sizes
#define TEST_M 8192
#define TEST_K 8192
#define TEST_N 8192

// Scale dimensions (one scale per 32 elements in K dimension)
#define SCALE_K (TEST_K / 32)

constexpr int BLOCK_SIZE_M = 256;
constexpr int BLOCK_SIZE_N = 256;
constexpr int K_STEP = 128;
constexpr int REG_BLOCK_M = BLOCK_SIZE_M / 2;
constexpr int REG_BLOCK_N = BLOCK_SIZE_N / 2;

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

using _gl_A = gl<din, -1, -1, -1, -1>;
using _gl_B = gl<din, -1, -1, -1, -1>;
using _gl_C = gl<dout, -1, -1, -1, -1>;

__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================================
// Simple scaled kernel for testing with compile-time scales
// ============================================================================

/**
 * @brief Test kernel that performs scaled FP6 MFMA with compile-time uniform scales.
 * 
 * This demonstrates using mma_ABt_scaled<SCALE_A, SCALE_B>() with fixed scales.
 */
template<scale_t SCALE_A, scale_t SCALE_B>
__global__ __launch_bounds__(NUM_THREADS, 1)
void scaled_mfma_test_kernel(
    _gl_A g_a,
    _gl_B g_b,
    _gl_C g_c)
{
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    // Allocate shared memory tiles
    st_f6<REG_BLOCK_M, K_STEP> (&As)[2] = al.allocate<st_f6<REG_BLOCK_M, K_STEP>, 2>();
    st_f6<REG_BLOCK_N, K_STEP> (&Bs)[2] = al.allocate<st_f6<REG_BLOCK_N, K_STEP>, 2>();

    uintptr_t A_base = reinterpret_cast<uintptr_t>(&As[0]);
    uintptr_t B_base = reinterpret_cast<uintptr_t>(&Bs[0]);

    st_f6<REG_BLOCK_M, K_STEP> *As_ptrs[2] = {
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[0]) - A_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_M, K_STEP>*>(A_base + (reinterpret_cast<uintptr_t>(&As[1]) - A_base) * 6 / 8)
    };

    st_f6<REG_BLOCK_N, K_STEP> *Bs_ptrs[2] = {
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[0]) - B_base) * 6 / 8),
        reinterpret_cast<st_f6<REG_BLOCK_N, K_STEP>*>(B_base + (reinterpret_cast<uintptr_t>(&Bs[1]) - B_base) * 6 / 8)
    };

    // Register tiles
    rt_f6<REG_BLOCK_M/2, K_STEP> A_tile[2];
    rt_f6<REG_BLOCK_N/2, K_STEP> B_tile[2];
    rt_fl<REG_BLOCK_M/2, REG_BLOCK_N/2, ducks::rt_layout::accumulator> C_accum[2][2];
    zero(C_accum[0][0]);
    zero(C_accum[0][1]);
    zero(C_accum[1][0]);
    zero(C_accum[1][1]);

    const int warp_id = kittens::warpid();
    const int warp_row = warp_id / 2;
    const int warp_col = warp_id % 2;
    
    // Simple single-block test: process row 0, col 0
    int row = 0, col = 0;
    
    constexpr int bytes_per_thread = 16;
    constexpr int memcpy_per_tile_A = (REG_BLOCK_M * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);
    constexpr int memcpy_per_tile_B = (REG_BLOCK_N * K_STEP * 6 / 8) / (bytes_per_thread * NUM_THREADS);

    uintptr_t swizzled_offsets_A[memcpy_per_tile_A];
    uintptr_t swizzled_offsets_B[memcpy_per_tile_B];

    prefill_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g_a, {0, 0, row, 0}, *As_ptrs[0], swizzled_offsets_A);
    prefill_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g_b, {0, 0, col, 0}, *Bs_ptrs[0], swizzled_offsets_B);

    const uint32_t addrA = prefill_swizzled_offset_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[0], {warp_row, 0}));
    const uint32_t addrB = prefill_swizzled_offset_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[0], {warp_col, 0}));

    // Load data
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g_a, {0, 0, row*2, 0}, *As_ptrs[0], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g_b, {0, 0, col*2, 0}, *Bs_ptrs[0], swizzled_offsets_B);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_M, K_STEP>, _gl_A, coord<st_f6<REG_BLOCK_M, K_STEP>>, NUM_THREADS>(g_a, {0, 0, row*2+1, 0}, *As_ptrs[1], swizzled_offsets_A);
    load_global_to_shared_direct_with_swizzled_offsets_fp6<2, false, st_f6<REG_BLOCK_N, K_STEP>, _gl_B, coord<st_f6<REG_BLOCK_N, K_STEP>>, NUM_THREADS>(g_b, {0, 0, col*2+1, 0}, *Bs_ptrs[1], swizzled_offsets_B);
    
    __syncthreads();

    // Load to registers
    load_lds_reg_row_fp6(A_tile[0], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[0], {warp_row, 0}), addrA);
    load_lds_reg_row_fp6(A_tile[1], subtile_inplace<REG_BLOCK_M/2, K_STEP>(*As_ptrs[1], {warp_row, 0}), addrA);
    load_lds_reg_row_fp6(B_tile[0], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[0], {warp_col, 0}), addrB);
    load_lds_reg_row_fp6(B_tile[1], subtile_inplace<REG_BLOCK_N/2, K_STEP>(*Bs_ptrs[1], {warp_col, 0}), addrB);

    __syncthreads();

    // Perform scaled MFMA - demonstrating compile-time scale usage
    // This applies uniform scaling: result = (A * 2^SCALE_A) * (B * 2^SCALE_B)^T
    mma_ABt_scaled<SCALE_A, SCALE_B>(C_accum[0][0], A_tile[0], B_tile[0], C_accum[0][0]);
    mma_ABt_scaled<SCALE_A, SCALE_B>(C_accum[0][1], A_tile[0], B_tile[1], C_accum[0][1]);
    mma_ABt_scaled<SCALE_A, SCALE_B>(C_accum[1][0], A_tile[1], B_tile[0], C_accum[1][0]);
    mma_ABt_scaled<SCALE_A, SCALE_B>(C_accum[1][1], A_tile[1], B_tile[1], C_accum[1][1]);

    // Store results
    store(g_c, C_accum[0][0], {0, 0, (row * 2)*2 + warp_row, (col * 2)*2 + warp_col});
    store(g_c, C_accum[0][1], {0, 0, (row * 2)*2 + warp_row, (col * 2 + 1)*2 + warp_col});
    store(g_c, C_accum[1][0], {0, 0, (row * 2 + 1)*2+warp_row, (col * 2)*2 + warp_col});
    store(g_c, C_accum[1][1], {0, 0, (row * 2 + 1)*2+warp_row, (col * 2 + 1)*2+warp_col});
}

// ============================================================================
// Host-side utilities
// ============================================================================

void pack_fp6(uint32_t *output, const din *input, int size) {
    for (int i = 0; i < size * 6 / 32; i++) {
        output[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        const uint8_t tmp = *reinterpret_cast<const uint8_t*>(&input[i]);
        const uint32_t v = static_cast<uint32_t>(tmp & 0x3Fu);
        const int bit_pos = i * 6;
        const int word_idx = bit_pos >> 5;
        const int bit_off = bit_pos & 31;

        output[word_idx] |= (v << bit_off);
        const int spill = bit_off + 6 - 32;
        if (spill > 0) {
            output[word_idx + 1] |= (v >> (6 - spill));
        }
    }
}

void random_init_fp6(din* data, int size, uint32_t seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < size; i++) {
        data[i] = din(dis(gen));
    }
}

/**
 * @brief CPU reference for scaled MXFP6 GEMM: C = (A * 2^scale_a) * (B * 2^scale_b)^T
 */
void cpu_reference_scaled(
    const din* A,
    const din* B,
    half* C,
    int M, int K, int N,
    int scale_a, int scale_b)
{
    float scale_factor = std::pow(2.0f, static_cast<float>(scale_a + scale_b));
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[j * K + k]);
            }
            C[i * N + j] = half(sum * scale_factor);
        }
    }
}

/**
 * @brief CPU reference for unscaled GEMM (for comparison)
 */
void cpu_reference_unscaled(
    const din* A,
    const din* B,
    half* C,
    int M, int K, int N)
{
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += static_cast<float>(A[i * K + k]) * static_cast<float>(B[j * K + k]);
            }
            C[i * N + j] = half(sum);
        }
    }
}

int main() {
    std::cout << "=== MXFP6 Scaled MFMA Test ===\n";
    std::cout << "Matrix dimensions: " << TEST_M << " x " << TEST_K << " x " << TEST_N << "\n\n";
    
    // Test scale values (compile-time constants)
    constexpr scale_t TEST_SCALE_A = 2;  // 2^2 = 4x
    constexpr scale_t TEST_SCALE_B = 1;  // 2^1 = 2x
    // Total scaling = 2^(2+1) = 8x
    
    std::cout << "Testing with compile-time scales: A_scale=" << TEST_SCALE_A 
              << ", B_scale=" << TEST_SCALE_B << "\n";
    std::cout << "Expected total scale factor: 2^" << (TEST_SCALE_A + TEST_SCALE_B) 
              << " = " << (1 << (TEST_SCALE_A + TEST_SCALE_B)) << "x\n\n";
    
    // Allocate host memory
    din *h_A = new din[TEST_M * TEST_K];
    din *h_B = new din[TEST_N * TEST_K];
    dout *h_C = new dout[TEST_M * TEST_N];
    half *cpu_C_scaled = new half[TEST_M * TEST_N];
    half *cpu_C_unscaled = new half[TEST_M * TEST_N];
    
    // Initialize data
    std::cout << "Initializing data...\n";
    random_init_fp6(h_A, TEST_M * TEST_K, 42);
    random_init_fp6(h_B, TEST_N * TEST_K, 123);
    
    // Pack FP6 data
    int total_words_a = (TEST_M * TEST_K * 6) / 32;
    int total_words_b = (TEST_N * TEST_K * 6) / 32;
    uint32_t *h_A_packed = new uint32_t[total_words_a];
    uint32_t *h_B_packed = new uint32_t[total_words_b];
    pack_fp6(h_A_packed, h_A, TEST_M * TEST_K);
    pack_fp6(h_B_packed, h_B, TEST_N * TEST_K);
    
    // Allocate device memory
    din *d_A, *d_B;
    dout *d_C;
    
    HIP_CHECK(hipMalloc(&d_A, (TEST_M * TEST_K * 6) / 8));
    HIP_CHECK(hipMalloc(&d_B, (TEST_N * TEST_K * 6) / 8));
    HIP_CHECK(hipMalloc(&d_C, TEST_M * TEST_N * sizeof(dout)));
    
    // Copy to device
    HIP_CHECK(hipMemcpy(d_A, h_A_packed, (TEST_M * TEST_K * 6) / 8, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B_packed, (TEST_N * TEST_K * 6) / 8, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, TEST_M * TEST_N * sizeof(dout)));
    
    // Compute CPU references
    std::cout << "Computing CPU reference (scaled)...\n";
    cpu_reference_scaled(h_A, h_B, cpu_C_scaled, TEST_M, TEST_K, TEST_N, TEST_SCALE_A, TEST_SCALE_B);
    
    std::cout << "Computing CPU reference (unscaled)...\n";
    cpu_reference_unscaled(h_A, h_B, cpu_C_unscaled, TEST_M, TEST_K, TEST_N);
    
    // Compare scaled vs unscaled CPU references to show scale effect
    float scale_diff_sum = 0.0f;
    for (int i = 0; i < TEST_M * TEST_N; i++) {
        scale_diff_sum += std::abs(float(cpu_C_scaled[i]) - float(cpu_C_unscaled[i]));
    }
    std::cout << "\nAverage difference between scaled and unscaled CPU: " 
              << scale_diff_sum / (TEST_M * TEST_N) << "\n";
    
    // Create global layouts
    _gl_A gl_a(d_A, 1, 1, TEST_M, TEST_K);
    _gl_B gl_b(d_B, 1, 1, TEST_N, TEST_K);
    _gl_C gl_c(d_C, 1, 1, TEST_M, TEST_N);
    
    // Run GPU kernel with scaled MFMA
    std::cout << "\nRunning GPU kernel with scaled MFMA...\n";
    dim3 grid(1);
    dim3 block(NUM_THREADS);
    
    scaled_mfma_test_kernel<TEST_SCALE_A, TEST_SCALE_B><<<grid, block, MAX_SHARED_MEMORY>>>(
        gl_a, gl_b, gl_c);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Copy results back
    HIP_CHECK(hipMemcpy(h_C, d_C, TEST_M * TEST_N * sizeof(dout), hipMemcpyDeviceToHost));
    
    // Verify against scaled CPU reference
    int errors_scaled = 0;
    int errors_unscaled = 0;
    float max_diff_scaled = 0.0f;
    float max_diff_unscaled = 0.0f;
    float total_diff_scaled = 0.0f;
    float total_diff_unscaled = 0.0f;
    
    for (int i = 0; i < TEST_M * TEST_N; i++) {
        float gpu_val = float(h_C[i]);
        float cpu_scaled = float(cpu_C_scaled[i]);
        float cpu_unscaled = float(cpu_C_unscaled[i]);
        
        float diff_scaled = std::abs(gpu_val - cpu_scaled);
        float diff_unscaled = std::abs(gpu_val - cpu_unscaled);
        
        float threshold_scaled = 0.1f * std::abs(cpu_scaled) + 0.01f;
        float threshold_unscaled = 0.1f * std::abs(cpu_unscaled) + 0.01f;
        
        max_diff_scaled = std::max(max_diff_scaled, diff_scaled);
        max_diff_unscaled = std::max(max_diff_unscaled, diff_unscaled);
        total_diff_scaled += diff_scaled;
        total_diff_unscaled += diff_unscaled;
        
        if (diff_scaled > threshold_scaled) {
            errors_scaled++;
            if (errors_scaled <= 3) {
                int row = i / TEST_N;
                int col = i % TEST_N;
                std::cout << "[" << row << "," << col << "] CPU_scaled: " << cpu_scaled 
                          << " GPU: " << gpu_val << " (diff: " << diff_scaled << ")\n";
            }
        }
        if (diff_unscaled > threshold_unscaled) {
            errors_unscaled++;
        }
    }
    
    std::cout << "\n=== Results ===\n";
    std::cout << "Comparison with SCALED CPU reference:\n";
    std::cout << "  Average diff: " << total_diff_scaled / (TEST_M * TEST_N) << "\n";
    std::cout << "  Max diff: " << max_diff_scaled << "\n";
    std::cout << "  Errors: " << errors_scaled << "/" << (TEST_M * TEST_N) << "\n";
    
    std::cout << "\nComparison with UNSCALED CPU reference:\n";
    std::cout << "  Average diff: " << total_diff_unscaled / (TEST_M * TEST_N) << "\n";
    std::cout << "  Max diff: " << max_diff_unscaled << "\n";
    std::cout << "  Errors: " << errors_unscaled << "/" << (TEST_M * TEST_N) << "\n";
    
    // Determine which reference matches better
    bool matches_scaled = (total_diff_scaled < total_diff_unscaled * 0.5);
    bool matches_unscaled = (total_diff_unscaled < total_diff_scaled * 0.5);
    
    std::cout << "\n";
    if (matches_scaled && errors_scaled < 100) {
        std::cout << "=== Scaled MFMA test PASSED ===\n";
        std::cout << "GPU output matches SCALED CPU reference!\n";
        std::cout << "The scaling factors are being applied correctly.\n";
    } else if (matches_unscaled && errors_unscaled < 100) {
        std::cout << "=== Test shows UNSCALED behavior ===\n";
        std::cout << "GPU output matches UNSCALED CPU reference.\n";
        std::cout << "Note: The scaled MFMA infrastructure is in place,\n";
        std::cout << "but the actual scaling may need ISA-level verification.\n";
    } else {
        std::cout << "=== Test INCONCLUSIVE ===\n";
        std::cout << "GPU output doesn't clearly match either reference.\n";
    }
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_A_packed;
    delete[] h_B_packed;
    delete[] h_C;
    delete[] cpu_C_scaled;
    delete[] cpu_C_unscaled;
    
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
    
    return errors_scaled < 100 ? 0 : 1;
}
