#include <cstdio>
#include <random>
#include "amd_detail/amd_hip_runtime.h"
#include "common/base_types.cuh"
#include "common/util.cuh"
#include "kittens.cuh" 
#include "ops/warp/register/tile/conversions.cuh"
#include "types/register/rt_layout.cuh"
#include "types/shared/st.cuh"
#include "types/shared/st_layout.cuh"

using namespace kittens;

#define NUM_WARPS 1
#define BLOCK_M 256
#define BLOCK_N 256
#define BLOCK_K 128
#define REG_MN 128
#define REG_K 64

using G = kittens::group<NUM_WARPS>;
using _gl = gl<fp8e4m3,-1,-1,-1,-1>;
using _gl_c = gl<float,-1,-1,-1,-1>;

__global__ __launch_bounds__(NUM_WARPS * kittens::WARP_THREADS, 2)  // launch_bounds(max_threads_per_block, min_warps_per_simd)
void matmul_device(const _gl A, const _gl B, _gl_c C) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // lds is 256 x 128 col-layout -- underlying subtiles should be 8 x 8
    // reg is 128 x 64 col-layout  -- underlying subtiles should be 4 x 4
    auto (&lds_a) = al.allocate<st<fp8e4m3, 16, 32>>();
    auto (&lds_b) = al.allocate<st<fp8e4m3, 16, 32>>();
    rt<fp8e4m3, 16, 32> reg_a;
    rt<fp8e4m3, 16, 32> reg_b;
    rt<float, 16, 16, ducks::rt_layout::col> reg_c;

    G::load(lds_a, A, {0, 0, 0, 0});
    // G::load(lds_b, B, {0, 0, 0, 0});

    // load(reg_a, subtile_inplace<16, 32>(lds_a, {0, 0}));
    // load(reg_b, subtile_inplace<16, 32>(lds_b, {0, 0}));
    // mma_ABt(reg_c, reg_a, reg_b, reg_c);
    // store(C, reg_c, {0, 0});

    if (threadIdx.x == 0) {
        uint8_t* shm_bytes = reinterpret_cast<uint8_t*>(&lds_a);
        printf("lds_a[0..31] raw bytes: ");
        for (int i = 0; i < 32; i++) {
            printf("0x%02x ", shm_bytes[i]);
        }
        printf("\n");
    }
}

int main() {
    constexpr int m = 16, n = 16, k = 32;
    fp8e4m3* a, *b;
    float* c;
    hipMallocManaged((void**)(&a), sizeof(fp8e4m3) * m * k);
    hipMallocManaged((void**)(&b), sizeof(fp8e4m3) * n * k);
    hipMallocManaged((void**)(&c), sizeof(float) * m * n);

    // std::mt19937 gen(1);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    // for (int i = 0; i < m * k; i++) { a[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    // for (int i = 0; i < n * k; i++) { b[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    float val = 0.0f;
    for (int i = 0; i < m * k; ++i) {
        a[i] = base_types::convertor<fp8e4m3, float>::convert(val);
        b[i] = base_types::convertor<fp8e4m3, float>::convert(val);
        val += 0.0625f;
    }

    printf("a[0..31] raw bytes: ");
    for (int i = 0; i < 32; i++) {
        printf("0x%02x ", *reinterpret_cast<uint8_t*>(&a[i]));
    }
    printf("\n");

    _gl GL_A(a, 1, 1, m, k);
    _gl GL_B(b, 1, 1, k, n);
    _gl_c GL_C(c, 1, 1, m, n);
    matmul_device<<<1, 64, 0, nullptr>>>(GL_A, GL_B, GL_C);
    hipDeviceSynchronize();

#if 0
    float c_ref[m * n];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int kk = 0; kk < k; kk++) {
                float av = base_types::convertor<float, fp8e4m3>::convert(a[i * k + kk]);
                float bv = base_types::convertor<float, fp8e4m3>::convert(b[j * k + kk]);
                acc += av * bv;
            }
            c_ref[i * n + j] = acc;
        }
    }
    auto print_f32_matrix = [&](const char* name, float* mat, int rows, int cols) {
        printf("%s:\n", name);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%6.2f ", mat[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    };
    print_f32_matrix("C (GPU)", c, m, n);
    print_f32_matrix("C (CPU)", c_ref, m, n);
#endif

    hipFree(a); hipFree(b); hipFree(c);

    return 0;
}