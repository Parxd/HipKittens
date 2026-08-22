#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
#include <hip/hip_runtime.h>

using namespace kittens;

#define NUM_WARPS 8

// MoE constants
constexpr int EXPERTS = 256;
constexpr int D_HIDDEN = 7168;
constexpr int D_EXPERT = 512;

// intra-gemm constants
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 256;
constexpr int BLOCK_K = 256;
constexpr int REG_MN = 16;
constexpr int REG_K = 32;

using _gl_A = gl<fp8e4m3,1,1,-1,-1>;
using _gl_B = gl<fp8e4m3,1,-1,-1,-1>;
using _gl_C = gl<float,-1,-1,-1,-1>;
using _gl_meta = gl<int,1,1,1,-1>;

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    _gl_meta token_to_slot;
    _gl_meta tile_to_expert;
    _gl_meta tile_to_m_limit;
    int num_valid_tiles;
    hipStream_t stream;

    dim3 block() { return dim3(NUM_WARPS * WARP_THREADS); }
    size_t dynamic_shared_memory() { return 65536; }
};

__global__ __launch_bounds__(NUM_WARPS * WARP_THREADS, 2)
void kernel(const micro_globals g) {
    // TODO: add tile swizzling--stack intra-XCD SMs along intra-expert M-tiles first
    const int total_tiles = g.num_valid_tiles * (D_EXPERT / BLOCK_N);
    for (int lt = blockIdx.x; lt < total_tiles; lt += gridDim.x) {
        int gl_m_tile = lt % g.num_valid_tiles, n_tile = lt / g.num_valid_tiles;
        int expert = g.tile_to_expert[gl_m_tile];
        int token_idx_beg = gl_m_tile * BLOCK_M, token_idx_end = g.tile_to_m_limit[gl_m_tile];
    }
}

void call(micro_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    auto grid_dim = dim3(prop.multiProcessorCount);  // TODO: confirm this is actually in units of CUs
    kernel<<<grid_dim, g.block(), mem_size, g.stream>>>(g);
}
