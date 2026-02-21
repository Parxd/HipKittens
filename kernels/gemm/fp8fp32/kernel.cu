#include <cstdio>
#include <random>
#include "amd_detail/amd_hip_runtime.h"
#include "common/base_types.cuh"
#include "common/util.cuh"
#include "kittens.cuh" 
#include "types/shared/st.cuh"

using namespace kittens;

#define NUM_WARPS 1
using G = kittens::group<NUM_WARPS>;
using _gl = gl<fp8e4m3,-1,-1,-1,-1>;

__global__ __launch_bounds__(kittens::WARP_THREADS, 1)
void matmul_device(const _gl A, const _gl B, _gl C) {
    // __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8();
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