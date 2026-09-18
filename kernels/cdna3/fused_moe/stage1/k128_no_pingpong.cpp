#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
#include <hip/hip_runtime.h>
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 8
#define SPLIT_K False

// MoE constants
constexpr int D_INTER = 512;
constexpr int D_MODEL = 2048;
constexpr int TOP_K = 8;

// intra-gemm constants
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 128;
constexpr int REG_M = 16;
constexpr int REG_N = 16;
constexpr int REG_K = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WEIGHT_SWIZZLE_GRANULARITY = BLOCK_N / 2;
constexpr size_t SMEM_BYTES = (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(fp8e4m3) + 
                            ((BLOCK_M + WEIGHT_SWIZZLE_GRANULARITY * 2) * sizeof(float));
constexpr int OCCUPANCY = 2;  // persistent CTAs/CU targeted by __launch_bounds__ below
static_assert(SMEM_BYTES * OCCUPANCY <= 64 * 1024, "SMEM_BYTES * OCCUPANCY exceeds gfx942 LDS size (64 KiB)");

using out_dtype = bf16;
using G = kittens::group<NUM_WARPS>;
using _gl_A = gl<fp8e4m3,1,1,-1,-1>;  // [M, d_model]
using _gl_B = gl<fp8e4m3,1,-1,-1,-1>;  // [expert, d_inter * 2, d_model]
using _gl_C = gl<out_dtype,1,1,-1,-1>;  // [M * topK, d_model]
using _gl_sf_A = gl<float,1,1,1,-1>;  // [M]
using _gl_sf_B = gl<float,1,1,-1,-1>;  // [expert, d_inter * 2]
using _gl_meta = gl<int,1,1,1,-1>;

struct moe_stage1_globals {
    _gl_A A;
    _gl_sf_A sf_A;
    _gl_B B;
    _gl_sf_B sf_B;
    _gl_C C;
    _gl_meta sorted_token_ids;
    _gl_meta sorted_expert_ids;
    _gl_meta num_valid_ids;
    hipStream_t stream;

    dim3 grid() { return 0; }  // dummy 
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return SMEM_BYTES; }
};

__global__ __launch_bounds__(NUM_THREADS, 4)
void kernel(const moe_stage1_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
    auto (&sf_A) = al.allocate<sv_fl<BLOCK_M>>();
    auto (&sf_gate) = al.allocate<sv_fl<WEIGHT_SWIZZLE_GRANULARITY>>();
    auto (&sf_up) = al.allocate<sv_fl<WEIGHT_SWIZZLE_GRANULARITY>>();
    rt_fp8e4m3<REG_M, REG_K> a_tiles[2];
    rt_fp8e4m3<REG_N, REG_K> b_tiles[4];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> accum[2];  // 0: gate accum., 1: up accum.
    rv_fl<REG_M, ducks::rv_layout::align> reg_sf_A;
    rv_fl<REG_N, ducks::rv_layout::ortho> reg_sf_W[2];

    const int warp_id = warpid();
    const int warp_row = warp_id / 4, warp_col = warp_id % 4;
    constexpr int k_iters = D_MODEL / BLOCK_K;

    const int num_valid_m_tiles = g.num_valid_ids[0] / BLOCK_M;
    constexpr int num_n_tiles = 2 * D_INTER / BLOCK_N;
    const int total_tiles = num_valid_m_tiles * num_n_tiles;
    const int num_tiles_per_cu = ceil_div(total_tiles, gridDim.x);
    const int chunk_size = 1;
    const int window_size = 1;
    const int base_bidx = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, chunk_size);

    for (int tile = 0; tile < num_tiles_per_cu && base_bidx + tile * gridDim.x < total_tiles; ++tile) {
    // for (int lt = blockIdx.x; lt < total_tiles; lt += gridDim.x) {
        for (int i = 0; i < 2; i++) { zero(accum[i]); }

        const int remap_bidx = base_bidx + tile * gridDim.x;
        int num_wgid_in_group = window_size * num_n_tiles;
        int group_id = remap_bidx / num_wgid_in_group;
        int first_pid_m = group_id * window_size;
        int group_size_m = min(num_valid_m_tiles - first_pid_m, window_size);
        int output_m = first_pid_m + ((remap_bidx % num_wgid_in_group) % group_size_m);
        int output_n = (remap_bidx % num_wgid_in_group) / group_size_m;
        // int output_m = lt % num_valid_m_tiles, output_n = lt / num_valid_m_tiles;
        int expert = g.sorted_expert_ids[output_m];

        gather_load<NUM_THREADS>(As, g.A, {0, 0, output_m, 0}, g.sorted_token_ids);
        G::load(Bs, g.B, {0, expert, output_n, 0});
        __builtin_amdgcn_s_barrier();

        for (int K_TILE = 0; K_TILE < k_iters - 1; ++K_TILE) {
            constexpr int BYTES_PER_MEMCPY = NUM_THREADS * sizeof(float4) / sizeof(fp8e4m3);
            constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;
            constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;
            float4 a_buffer_next[BUFFER_SIZE_A];
            float4 b_buffer_next[BUFFER_SIZE_B];
            
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_buffer_next, BUFFER_SIZE_B, g.B, {0, expert, output_n, K_TILE + 1}, Bs);
            gather_load_global_to_register_buffer<NUM_THREADS>(a_buffer_next, BUFFER_SIZE_A, g.A, {0, 0, output_m, K_TILE + 1}, g.sorted_token_ids, As);
            load(a_tiles[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
            load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));
            load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col + 4, 0}));
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[0], a_tiles[0], b_tiles[0], accum[0]);
            mma_ABt(accum[1], a_tiles[0], b_tiles[1], accum[1]);
            __builtin_amdgcn_s_setprio(0);

            load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
            load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));
            load(b_tiles[3], subtile_inplace<REG_N, REG_K>(Bs, {warp_col + 4, 1}));

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_setprio(1);
            mma_ABt(accum[0], a_tiles[1], b_tiles[2], accum[0]);
            mma_ABt(accum[1], a_tiles[1], b_tiles[3], accum[1]);
            __builtin_amdgcn_s_setprio(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            store_register_buffer_to_shared<NUM_THREADS>(As, a_buffer_next);
            store_register_buffer_to_shared<NUM_THREADS>(Bs, b_buffer_next);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
        }
        __builtin_amdgcn_sched_barrier(0);
        gather_f32_sf_a<NUM_THREADS>(sf_A, g.sf_A, {output_m}, g.sorted_token_ids);
        if (warp_id == 0) {
            load(sf_gate, g.sf_B, {expert, output_n});
            load(sf_up, g.sf_B, {expert, output_n + (D_INTER / WEIGHT_SWIZZLE_GRANULARITY)});
        }
        load(a_tiles[0], subtile_inplace<REG_M, REG_K>(As, {warp_row, 0}));
        load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As, {warp_row, 1}));
        load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 0}));      // gate
        load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs, {warp_col, 1}));      // gate
        load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs, {warp_col + 4, 0}));  // up
        load(b_tiles[3], subtile_inplace<REG_N, REG_K>(Bs, {warp_col + 4, 1}));  // up
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();  // for block-wide visibility of scale factors

        __builtin_amdgcn_s_setprio(1);
        mma_ABt(accum[0], a_tiles[0], b_tiles[0], accum[0]);
        mma_ABt(accum[0], a_tiles[1], b_tiles[1], accum[0]);
        mma_ABt(accum[1], a_tiles[0], b_tiles[2], accum[1]);
        mma_ABt(accum[1], a_tiles[1], b_tiles[3], accum[1]);
        __builtin_amdgcn_s_setprio(0);

        load_sv_to_rv(reg_sf_A, subvec_inplace<REG_M>(sf_A, warp_row));
        load_sv_to_rv(reg_sf_W[0], subvec_inplace<REG_N>(sf_gate, warp_col));
        load_sv_to_rv(reg_sf_W[1], subvec_inplace<REG_N>(sf_up, warp_col));
        apply_row_sf(accum[0], accum[0], reg_sf_A);
        apply_col_sf(accum[0], accum[0], reg_sf_W[0]);
        apply_row_sf(accum[1], accum[1], reg_sf_A);
        apply_col_sf(accum[1], accum[1], reg_sf_W[1]);
        silu(accum[0], accum[0]);
        mul(accum[0], accum[0], accum[1]);

        scatter_store<TOP_K>(g.C, accum[0], {0, 0, output_m * 2 + warp_row, output_n * 4 + warp_col}, g.sorted_token_ids);
    }
}

void call(moe_stage1_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    auto grid_dim = dim3(OCCUPANCY * prop.multiProcessorCount);
    kernel<<<grid_dim, g.block(), mem_size, g.stream>>>(g);
}

PYBIND11_MODULE(tk_kernel, m) {
    m.doc() = "tk_kernel python module";
    py::bind_kernel<kernel>(
        m,
        "kernel",
        &moe_stage1_globals::A,
        &moe_stage1_globals::sf_A,
        &moe_stage1_globals::B,
        &moe_stage1_globals::sf_B,
        &moe_stage1_globals::C,
        &moe_stage1_globals::sorted_token_ids,
        &moe_stage1_globals::sorted_expert_ids,
        &moe_stage1_globals::num_valid_ids
    );
    py::bind_function<call>(
        m,
        "call",
        &moe_stage1_globals::A,
        &moe_stage1_globals::sf_A,
        &moe_stage1_globals::B,
        &moe_stage1_globals::sf_B,
        &moe_stage1_globals::C,
        &moe_stage1_globals::sorted_token_ids,
        &moe_stage1_globals::sorted_expert_ids,
        &moe_stage1_globals::num_valid_ids
    );
}
