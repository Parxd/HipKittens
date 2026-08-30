#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
#include <hip/hip_runtime.h>
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 8

// MoE constants
constexpr int EXPERTS = 256;
constexpr int D_HIDDEN = 7168;
constexpr int D_EXPERT = 512;

// intra-gemm constants
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;
constexpr int REG_M = 16;
constexpr int REG_N = 16;
constexpr int REG_K = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WEIGHT_SWIZZLE_GRANULARITY = BLOCK_N / 2;
constexpr size_t SMEM_BYTES = BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K;
static_assert(SMEM_BYTES <= 64 * 1024, "SMEM_BYTES exceeds gfx942 LDS size (64 KiB)");

using G = kittens::group<NUM_WARPS>;
using _gl_A = gl<fp8e4m3,1,1,-1,-1>;
using _gl_B = gl<fp8e4m3,1,-1,-1,-1>;  // [expert, d_expert * 2, d_hidden]
using _gl_C = gl<float,1,-1,-1,-1>;  // [M, topK_slot, d_expert]
using _gl_sf_A = gl<float,1,1,1,-1>;
using _gl_sf_B = gl<float,1,1,-1,-1>;  // [expert, d_expert * 2]
using _gl_meta = gl<int,1,1,1,-1>;

struct moe_stage1_globals {
    _gl_A A;
    _gl_sf_A sf_A;
    _gl_B B;
    _gl_sf_B sf_B;
    _gl_C C;
    _gl_meta sorted_token_ids;
    _gl_meta sorted_expert_ids;
    int num_valid_tiles;  // equivalent to (num_valid_ids[0] / BLOCK_M)
    hipStream_t stream;

    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return SMEM_BYTES; }
};

__global__ __launch_bounds__(NUM_THREADS, 2)
void kernel(const moe_stage1_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    auto (&sf_A) = al.allocate<sv_fl<BLOCK_M>>();
    auto (&sf_B) = al.allocate<sv_fl<BLOCK_N>>();
    rt<fp8e4m3, REG_M, REG_K> tiles_a[4];
    rt<fp8e4m3, REG_N, REG_K> tiles_b[8];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> accum[2];
    for (int i = 0; i < 2; i++) { zero(accum[i]); }
    
    const int warp_id = warpid();
    const int warp_row = warp_id / 4, warp_col = warp_id % 4;
    constexpr int k_iters = D_HIDDEN / BLOCK_K;

    // TODO: add tile swizzling--stack intra-XCD SMs along intra-expert M-tiles first
    const int total_tiles = g.num_valid_tiles * (D_EXPERT / BLOCK_N);
    for (int lt = blockIdx.x; lt < total_tiles; lt += gridDim.x) {
        int gl_m_tile = lt % g.num_valid_tiles, n_tile = lt / g.num_valid_tiles;
        int expert = g.sorted_expert_ids[gl_m_tile];

        gather_load<NUM_THREADS>(As, g.A, {0, 0, gl_m_tile, 0}, g.sorted_token_ids);
        G::load(Bs, g.B, {0, expert, n_tile, 0});
        __builtin_amdgcn_s_barrier();

        if (warp_row == 1) {
            __builtin_amdgcn_s_barrier();
        }

        #pragma unroll 2
        for (int K_TILE = 0; K_TILE < k_iters - 1; ++K_TILE) {
            constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
            constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K) / NUM_THREADS / sizeof(float4) / sizeof(fp8e4m3);
            float4 a_buffer_next[BUFFER_SIZE_A];
            float4 b_buffer_next[BUFFER_SIZE_B];
            
            // Cluster 0
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE_B, g.B, {0, expert, n_tile, K_TILE + 1}, Bs);
            load(tiles_a[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
            load(tiles_a[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
            load(tiles_b[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
            load(tiles_b[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 1
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[0], a_tiles[0], b_tiles[0], accum[0]);
            mma_ABt(accum[0], a_tiles[0], b_tiles[1], accum[0]);
            mma_ABt(accum[0], a_tiles[1], b_tiles[0], accum[0]);
            mma_ABt(accum[0], a_tiles[1], b_tiles[1], accum[0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 2
            load(tiles_b[2], subtile_inplace<REG_M, REG_K>(Bs, {warp_col, 2}));
            load(tiles_b[3], subtile_inplace<REG_M, REG_K>(Bs, {warp_col, 3}));
            load(tiles_a[2], subtile_inplace<REG_M, REG_K>(As, {warp_row, 2}));
            load(tiles_a[3], subtile_inplace<REG_M, REG_K>(As, {warp_row, 3}));
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 3
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[0], tiles_a[2], tiles_b[2], accum[0]);
            mma_ABt(accum[0], tiles_a[2], tiles_b[3], accum[0]);
            mma_ABt(accum[0], tiles_a[3], tiles_b[2], accum[0]);
            mma_ABt(accum[0], tiles_a[3], tiles_b[3], accum[0]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 4
            gather_load_global_to_register_buffer<2, false, NUM_THREADS>(a_buffer_next, BUFFER_SIZE_A, g.A, {0, 0, gl_m_tile, K_TILE + 1}, g.sorted_token_ids);
            load(tiles_b[4], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 0}));
            load(tiles_b[5], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 1}));
            load(tiles_b[6], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 2}));
            load(tiles_b[7], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 3}));
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 5
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[1], tiles_a[0], tiles_b[4], accum[1]);
            mma_ABt(accum[1], tiles_a[0], tiles_b[5], accum[1]);
            mma_ABt(accum[1], tiles_a[1], tiles_b[4], accum[1]);
            mma_ABt(accum[1], tiles_a[1], tiles_b[5], accum[1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 6
            asm volatile("s_waitcnt lgkmcnt(0)");
            store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
            store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            // Cluster 7
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[1], tiles_a[0], tiles_b[6], accum[1]);
            mma_ABt(accum[1], tiles_a[0], tiles_b[7], accum[1]);
            mma_ABt(accum[1], tiles_a[1], tiles_b[6], accum[1]);
            mma_ABt(accum[1], tiles_a[1], tiles_b[7], accum[1]);
            __builtin_amdgcn_s_setprio(0);
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }
        // TODO: add PTPC quantization epilogue
        __builtin_amdgcn_sched_barrier(0);
        load(tiles_a[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
        load(tiles_a[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
        load(tiles_b[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
        load(tiles_b[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(accum[0], a_tiles[0], b_tiles[0], accum[0]);
        mma_ABt(accum[0], a_tiles[0], b_tiles[1], accum[0]);
        mma_ABt(accum[0], a_tiles[1], b_tiles[0], accum[0]);
        mma_ABt(accum[0], a_tiles[1], b_tiles[1], accum[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(tiles_b[2], subtile_inplace<REG_M, REG_K>(Bs, {warp_col, 2}));
        load(tiles_b[3], subtile_inplace<REG_M, REG_K>(Bs, {warp_col, 3}));
        load(tiles_a[2], subtile_inplace<REG_M, REG_K>(As, {warp_row, 2}));
        load(tiles_a[3], subtile_inplace<REG_M, REG_K>(As, {warp_row, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(accum[0], tiles_a[2], tiles_b[2], accum[0]);
        mma_ABt(accum[0], tiles_a[2], tiles_b[3], accum[0]);
        mma_ABt(accum[0], tiles_a[3], tiles_b[2], accum[0]);
        mma_ABt(accum[0], tiles_a[3], tiles_b[3], accum[0]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load(tiles_b[4], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 0}));
        load(tiles_b[5], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 1}));
        load(tiles_b[6], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 2}));
        load(tiles_b[7], subtile_inplace<REG_M, REG_K>(Bs, {warp_col + 4, 3}));
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        mma_ABt(accum[1], tiles_a[0], tiles_b[4], accum[1]);
        mma_ABt(accum[1], tiles_a[0], tiles_b[5], accum[1]);
        mma_ABt(accum[1], tiles_a[1], tiles_b[4], accum[1]);
        mma_ABt(accum[1], tiles_a[1], tiles_b[5], accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(accum[1], tiles_a[0], tiles_b[6], accum[1]);
        mma_ABt(accum[1], tiles_a[0], tiles_b[7], accum[1]);
        mma_ABt(accum[1], tiles_a[1], tiles_b[6], accum[1]);
        mma_ABt(accum[1], tiles_a[1], tiles_b[7], accum[1]);
        __builtin_amdgcn_s_setprio(0);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
    }
}

void call(moe_stage1_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    auto grid_dim = dim3(prop.multiProcessorCount);  // TODO: confirm this is actually in units of CUs
    kernel<<<grid_dim, g.block(), mem_size, g.stream>>>(g);
}
