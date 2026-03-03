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

#define NUM_WARPS 4
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

    // auto (&lds_a) = al.allocate<st<fp8e4m3, 128, 256, ducks::st_layout::col>>();
    // auto (&lds_b) = al.allocate<st<fp8e4m3, 128, 256, ducks::st_layout::col>>();
    // rt<fp8e4m3, 128, 64, ducks::rt_layout::col> reg_a;
    // rt<fp8e4m3, 128, 64, ducks::rt_layout::col> reg_b;
    // rt<float,   64, 64, ducks::rt_layout::col> reg_c;
    // zero(reg_c);

    // G::load(lds_a, A, {0, 0, 0, 0});
    // G::load(lds_b, B, {0, 0, 0, 0});
    // __builtin_amdgcn_s_barrier();

    // auto warp_id = warpid();
    // auto warp_row = warp_id / 4, warp_col = warp_id % 4;
    // load(reg_a, subtile_inplace<128, 64>(lds_a, {0, warp_row}));
    // load(reg_b, subtile_inplace<128, 64>(lds_b, {0, warp_col}));

    // asm volatile("s_waitcnt lgkmcnt(0)");
    // __builtin_amdgcn_s_barrier();
    // mma_AtB(reg_c, reg_a, reg_b, reg_c);
    // __builtin_amdgcn_s_barrier();
    // __builtin_amdgcn_sched_barrier(0);

    // store(C, reg_c, {warp_row, warp_col});

    auto (&lds) = al.allocate<st<fp8e4m3, 256, 128, ducks::st_layout::row>>();
    constexpr auto reg_buffer_sz = (256 * 128) / (NUM_WARPS * kittens::WARP_THREADS) / (sizeof(float4) / sizeof(fp8e4m3));
    float4 reg_buffer[reg_buffer_sz];
    load_global_to_register_buffer(reg_buffer, reg_buffer_sz, A, {0, 0}, lds);
    __builtin_amdgcn_s_barrier();
    store_register_buffer_to_shared(lds, reg_buffer);
    __builtin_amdgcn_s_barrier();
    asm volatile("s_waitcnt lgkmcnt(0)");

    if (threadIdx.x == 0) {
        printf("LDS: \n");
        for (int i = 0; i < 256 * 128; ++i) {
            if (float(lds.data[i]) - 1.00000f > 0.00001) {
                printf("%i, %f\n", i, float(lds.data[i]));
            }
        }
        printf("\n");
    }
}

int main() {
    constexpr int M = 256, N = 256, K = 128;
    fp8e4m3* a, *b;
    float* c;
    hipMallocManaged((void**)(&a), sizeof(fp8e4m3) * M * K);
    hipMallocManaged((void**)(&b), sizeof(fp8e4m3) * N * K);
    hipMallocManaged((void**)(&c), sizeof(float) * M * N);

    // std::mt19937 gen(1);
    // std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    // for (int i = 0; i < M * K; i++) { a[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    // for (int i = 0; i < N * K; i++) { b[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    float val = 0.0;
    for (int i = 0; i < M * K; ++i) {
        a[i] = base_types::convertor<fp8e4m3, float>::convert(1);
        b[i] = base_types::convertor<fp8e4m3, float>::convert(1);
        val += 0.0625;
    }

    _gl   GL_A(a, 1, 1, K, M);
    _gl   GL_B(b, 1, 1, K, N);
    _gl_c GL_C(c, 1, 1, M, N);
    matmul_device<<<1, NUM_WARPS * kittens::WARP_THREADS, sizeof(fp8e4m3) * M * K * 2, nullptr>>>(GL_A, GL_B, GL_C);
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