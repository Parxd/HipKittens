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

__global__ __launch_bounds__(NUM_WARPS * kittens::WARP_THREADS, 2)  // launch_bounds(max_threads_per_block, min_warps_per_simd)
void matmul_device(const _gl A, const _gl B, _gl C) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    auto (&lds) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K, ducks::st_layout::col>>();
    rt<fp8e4m3, REG_MN, REG_K, ducks::rt_layout::col> reg;
    
    load(reg, subtile_inplace<REG_MN, REG_K>(lds, {0, 0}));
}

int main() {
    int m = 16, n = 16, k = 32;
    fp8e4m3* a, *b, *c;
    hipMallocManaged((void**)(&a), sizeof(fp8e4m3) * m * k);
    hipMallocManaged((void**)(&b), sizeof(fp8e4m3) * n * k);
    hipMallocManaged((void**)(&c), sizeof(fp8e4m3) * m * n);

    std::mt19937 gen(1);
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    for (int i = 0; i < m * k; i++) { a[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }
    for (int i = 0; i < n * k; i++) { b[i] = base_types::convertor<fp8e4m3, float>::convert(dis(gen)); }

    _gl GL_A(a, 1, 1, m, k);
    _gl GL_B(b, 1, 1, k, n);
    _gl GL_C(c, 1, 1, m, n);
    matmul_device<<<1, 64, 0, nullptr>>>(GL_A, GL_B, GL_C);
    hipDeviceSynchronize();

    hipFree(a); hipFree(b); hipFree(c);

    return 0;
}