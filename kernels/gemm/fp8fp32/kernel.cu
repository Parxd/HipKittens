#include <cstdio>
#include <vector>
#include <random>
#include "amd_detail/amd_hip_runtime.h"
#include "common/base_types.cuh"
#include "common/util.cuh"
#include "kittens.cuh" 
#include "ops/warp/memory/tile/global_to_shared.cuh"
#include "ops/warp/register/tile/conversions.cuh"
#include "ops/warp/shared/tile/conversions.cuh"
#include "types/shared/st_layout.cuh"

using namespace kittens;

#define NUM_WARPS 8
#define BLOCK_M 256
#define BLOCK_N 256
#define BLOCK_K 128
#define REG_MN 128
#define REG_MN 64

using G = kittens::group<NUM_WARPS>;
using _gl = gl<fp8e4m3,-1,-1,-1,-1>;
using _gl_c = gl<float,-1,-1,-1,-1>;

__global__ __launch_bounds__(NUM_WARPS * kittens::WARP_THREADS, 2)  // launch_bounds(max_threads_per_block, min_warps_per_simd)
void matmul_device(const _gl A, const _gl B, _gl_c C) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    auto (&lds_a) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&lds_b) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    rt<fp8e4m3, REG_MN, REG_MN> tiles[1];

    const int k_iter = A.cols() / BLOCK_K;
}

int main() {
    int M = 8192, N = 8192, K = 8192;
    fp8e4m3* a, *b;
    float* c;
    hipMallocManaged((void**)(&a), sizeof(fp8e4m3) * M * K);
    hipMallocManaged((void**)(&b), sizeof(fp8e4m3) * N * K);
    hipMallocManaged((void**)(&c), sizeof(float) * M * N);

    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) { a[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    for (int i = 0; i < N * K; i++) { b[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    float val = 0.0;
    // for (int i = 0; i < M * K; ++i) {
    //     a[i] = base_types::convertor<fp8e4m3, float>::convert(1);
    //     b[i] = base_types::convertor<fp8e4m3, float>::convert(1);
    //     val += 0.0625;
    // }

    _gl   GL_A(a, 1, 1, M, K);
    _gl   GL_B(b, 1, 1, N, K);
    _gl_c GL_C(c, 1, 1, M, N);
    matmul_device<<<1, NUM_WARPS * kittens::WARP_THREADS, sizeof(fp8e4m3) * (M * K) + (K * N), nullptr>>>(GL_A, GL_B, GL_C);
    hipDeviceSynchronize();

#if 0
    float c_ref[M * N];
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += float(a[k * M + m]) * float(b[k * N + n]);
            }
            c_ref[m * N + n] = acc;
        }
    }

    int total = M * N;
    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float sum_rel_diff = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float ref  = c_ref[m * N + n];
            float got  = c[m * N + n];
            float diff = fabsf(ref - got);
            float rel  = (fabsf(ref) > 1e-5f) ? diff / fabsf(ref) : diff;

            max_abs_diff  = fmaxf(max_abs_diff, diff);
            sum_abs_diff += diff;
            max_rel_diff  = fmaxf(max_rel_diff, rel);
            sum_rel_diff += rel;
        }
    }
    printf("Max abs error:  %e\n", max_abs_diff);
    printf("Mean abs error: %e\n", sum_abs_diff / total);
    printf("Max rel error:  %e\n", max_rel_diff);
    printf("Mean rel error: %e\n", sum_rel_diff / total);
#endif

    hipFree(a); hipFree(b); hipFree(c);
    return 0;
}