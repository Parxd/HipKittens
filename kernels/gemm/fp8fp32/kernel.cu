#include <cstdio>
#include <vector>
#include <random>
#include "amd_detail/amd_hip_runtime.h"
#include "common/base_types.cuh"
#include "common/util.cuh"
#include "kittens.cuh" 
#include "ops/warp/register/tile/conversions.cuh"

using namespace kittens;

#define NUM_WARPS 8
#define BLOCK_M 256
#define BLOCK_N 256
#define BLOCK_K 64
#define REG_MN   64
#define REG_K    32

using G = kittens::group<NUM_WARPS>;
using _gl = gl<fp8e4m3,-1,-1,-1,-1>;
using _gl_c = gl<float,-1,-1,-1,-1>;

__global__ __launch_bounds__(NUM_WARPS * WARP_THREADS, 2)  // launch_bounds(max_threads_per_block, min_warps_per_simd)
void matmul_device(const _gl A, const _gl B, _gl_c C) {
    constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    rt<fp8e4m3, REG_MN, REG_K> tiles[6];
    rt_fl<REG_MN, REG_MN, ducks::rt_layout::col> C_accum[2];
    for (int i = 0; i < 2; i++) { zero(C_accum[i]); }

    const int warp_id = warpid();
    const int output_m = blockIdx.y, output_n = blockIdx.x;  // TODO: use cache-aware assignments
    const int warp_row = warp_id / 4, warp_col = warp_id % 4;
    const int k_iters = A.cols() / BLOCK_K;

    G::load(As, A, {0, 0, output_m, 0});
    G::load(Bs, B, {0, 0, output_n, 0});
    __builtin_amdgcn_s_barrier();

    // if (warp_row == 1) {
    //     __builtin_amdgcn_s_barrier();
    // }

    for (int K_TILE = 0; K_TILE < k_iters; ++K_TILE) {
        // constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
        // constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
        // float4 a_buffer_next[BUFFER_SIZE_A];
        // float4 b_buffer_next[BUFFER_SIZE_A];
        // load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE_A, A, {0, 0, output_m, K_TILE + 1}, As);
        // load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE_B, B, {0, 0, output_n, K_TILE + 1}, Bs);

        load(tiles[0], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 0}));
        load(tiles[2], subtile_inplace<REG_MN, REG_K>(As, {warp_row, 1}));
        load(tiles[1], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 0}));
        load(tiles[3], subtile_inplace<REG_MN, REG_K>(As, {warp_row + 2, 1}));
        load(tiles[4], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 0}));
        load(tiles[5], subtile_inplace<REG_MN, REG_K>(Bs, {warp_col, 1}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        mma_ABt(C_accum[0], tiles[0], tiles[4], C_accum[0]);
        mma_ABt(C_accum[0], tiles[2], tiles[5], C_accum[0]);

        mma_ABt(C_accum[1], tiles[1], tiles[4], C_accum[1]);
        mma_ABt(C_accum[1], tiles[3], tiles[5], C_accum[1]);

        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
    store(C, C_accum[0], {0, 0, warp_row, warp_col});
    store(C, C_accum[1], {0, 0, warp_row + 2, warp_col});
}

int main() {
    const int M = 256, N = 256, K = 64;
    fp8e4m3* a, *b;
    float* c;
    hipMallocManaged((void**)(&a), sizeof(fp8e4m3) * M * K);
    hipMallocManaged((void**)(&b), sizeof(fp8e4m3) * N * K);
    hipMallocManaged((void**)(&c), sizeof(float) * M * N);

    auto host_a = new fp8e4m3[M * K];
    auto host_b = new fp8e4m3[N * K];

    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < M * K; i++) {
        float val = dis(gen);
        a[i]      = base_types::convertor<fp8e4m3, float>::convert(val);
        host_a[i] = base_types::convertor<fp8e4m3, float>::convert(val);
    }
    for (int i = 0; i < N * K; i++) {
        float val = dis(gen);
        b[i]      = base_types::convertor<fp8e4m3, float>::convert(val);
        host_b[i] = base_types::convertor<fp8e4m3, float>::convert(val);
    }

    _gl   GL_A(a, 1, 1, M, K);
    _gl   GL_B(b, 1, 1, N, K);
    _gl_c GL_C(c, 1, 1, M, N);
    dim3 gridDim(ceil_div(N, BLOCK_N), ceil_div(M, BLOCK_M));
    matmul_device<<<gridDim, NUM_WARPS * WARP_THREADS, sizeof(fp8e4m3) * (BLOCK_M * BLOCK_K) + (BLOCK_K * BLOCK_N), nullptr>>>(GL_A, GL_B, GL_C);
    hipError_t err = hipGetLastError();
    printf("launch err?: %s\n", hipGetErrorString(err));
    hipDeviceSynchronize();

#if 1
    auto host_c = new float[M * N];

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += float(host_a[m * K + k]) * float(host_b[n * K + k]);
            }
            host_c[m * N + n] = acc;
        }
    }

    int total = M * N;
    float max_abs_diff = 0.0f;
    float sum_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    float sum_rel_diff = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float ref  = host_c[m * N + n];
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

    printf("ref[0]=%f got[0]=%f\n", host_c[0], c[0]);
    printf("ref[1]=%f got[1]=%f\n", host_c[1], c[1]);
    printf("ref[256]=%f got[256]=%f\n", host_c[256], c[256]);

    int zeros = 0;
    for (int i = 0; i < M * N; i++)
        if (host_c[i] == 0.0f) zeros++;
    printf("zero count: %d / %d\n", zeros, M*N);
    delete[] host_c;
#endif

    hipFree(a); hipFree(b); hipFree(c);
    delete[] host_a; delete[] host_b;

    return 0;
}