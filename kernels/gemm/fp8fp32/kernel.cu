#include <cstdio>
#include <vector>
#include <random>
#include "amd_detail/amd_hip_runtime.h"
#include "kittens.cuh" 
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

__global__   // launch_bounds(max_threads_per_block, min_warps_per_simd)
void matmul_device(const _gl A, const _gl B, _gl_c C) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    // lds is 256 x 128 col-layout -- underlying subtiles should be 8 x 8
    // reg is 128 x 64 col-layout  -- underlying subtiles should be 4 x 4
    auto (&lds_a) = al.allocate<st<fp8e4m3, 32, 16, ducks::st_layout::col>>();
    auto (&lds_b) = al.allocate<st<fp8e4m3, 32, 16, ducks::st_layout::col>>();
    rt<fp8e4m3, 32, 16, ducks::rt_layout::col> reg_a;
    rt<fp8e4m3, 32, 16, ducks::rt_layout::col> reg_b;
    rt<float,   16, 16, ducks::rt_layout::col> reg_c;
    zero(reg_c);

    G::load(lds_a, A, {0, 0, 0, 0});
    G::load(lds_b, B, {0, 0, 0, 0});

    load(reg_a, subtile_inplace<32, 16>(lds_a, {0, 0}));
    load(reg_b, subtile_inplace<32, 16>(lds_b, {0, 0}));
    asm volatile("s_waitcnt lgkmcnt(0)");
    mma_AtB(reg_c, reg_a, reg_b, reg_c);
    store(C, reg_c, {0, 0});

    /*
    FP32 DUMP LDS
    */
    // if (threadIdx.x == 0) {
    //     printf("lds_a: ");
    //     for (int i = 0; i < 8; ++i) {
    //         printf("%f ", float(lds_a.data[i]));
    //     }
    //     printf("\n");
    // }

    /*
    FP32 DUMP REG
    */
    __builtin_amdgcn_s_barrier();
    if (threadIdx.x == 32) {
        auto [a_tile0_0, a_tile0_1, a_tile0_2, a_tile0_3] = static_cast<float4>(reg_a.tiles[0][0].data[0]);
        printf("%f\n", a_tile0_0);
        printf("%f\n", a_tile0_1);
        printf("%f\n", a_tile0_2);
        printf("%f\n", a_tile0_3);

        auto [a_tile1_0, a_tile1_1, a_tile1_2, a_tile1_3] = static_cast<float4>(reg_a.tiles[0][0].data[1]);
        printf("%f\n", a_tile1_0);
        printf("%f\n", a_tile1_1);
        printf("%f\n", a_tile1_2);
        printf("%f\n", a_tile1_3);

        auto [tile0_0, tile0_1, tile0_2, tile0_3] = static_cast<float4>(reg_b.tiles[0][0].data[0]);
        printf("%f\n", tile0_0);
        printf("%f\n", tile0_1);
        printf("%f\n", tile0_2);
        printf("%f\n", tile0_3);

        auto [tile1_0, tile1_1, tile1_2, tile1_3] = static_cast<float4>(reg_b.tiles[0][0].data[1]);
        printf("%f\n", tile1_0);
        printf("%f\n", tile1_1);
        printf("%f\n", tile1_2);
        printf("%f\n", tile1_3);

        // auto [res_0, res_1] = reg_c.tiles[0][0].data[0];
        // auto [res_2, res_3] = reg_c.tiles[0][0].data[1];
        // printf("%f\n", res_0);
        // printf("%f\n", res_1);
        // printf("%f\n", res_2);
        // printf("%f\n", res_3);
    }
}

int main() {
    constexpr int M = 16, N = 16, K = 32;
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
        // val += 0.0625;
    }

    _gl   GL_A(a, 1, 1, K, M);
    _gl   GL_B(b, 1, 1, K, N);
    _gl_c GL_C(c, 1, 1, M, N);
    matmul_device<<<1, 64, sizeof(fp8e4m3) * M * K * 2, nullptr>>>(GL_A, GL_B, GL_C);
    hipDeviceSynchronize();

#if 1
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
    auto print_f32_matrix = [&](const char* name, float* mat, int rows, int cols) {
        printf("\n");
        printf("%s:\n", name);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%6.2f ", mat[i * cols + j]);
            }
            printf("\n");
        }
        printf("\n");
    };
    print_f32_matrix("C (CPU)", c_ref, M, N);
#endif

    hipFree(a); hipFree(b); hipFree(c);
    return 0;
}