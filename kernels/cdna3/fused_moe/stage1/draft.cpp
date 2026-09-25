#include "cdna3/common/util.cuh"
#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
#include <hip/hip_runtime.h>
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 4
#define SPLIT_K False

// MoE constants
constexpr int D_INTER = 2048;
constexpr int D_MODEL = 7168;
constexpr int TOP_K = 8;

// intra-gemm constants
constexpr int BLOCK_M = 32;
constexpr int BLOCK_N = 64;
constexpr int BLOCK_K = 128;
constexpr int REG_M = 16;
constexpr int REG_N = 16;
constexpr int REG_K = 64;
constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;
constexpr int WEIGHT_SWIZZLE_GRANULARITY = BLOCK_N / 2;
constexpr size_t SMEM_BYTES = 2 * (BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K) * sizeof(fp8e4m3) + 
                            ((BLOCK_M + WEIGHT_SWIZZLE_GRANULARITY * 2) * sizeof(float));
constexpr int OCCUPANCY = 2;
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

__global__ __launch_bounds__(NUM_THREADS, OCCUPANCY)
void kernel(const moe_stage1_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);

    using a_st = st<fp8e4m3, BLOCK_M, BLOCK_K>;
    using b_st = st<fp8e4m3, BLOCK_N, BLOCK_K>;
    auto (&As)[2] = al.allocate<a_st, 2>();
    auto (&Bs)[2] = al.allocate<b_st, 2>();
    auto (&sf_A) = al.allocate<sv_fl<BLOCK_M>>();
    auto (&sf_gate) = al.allocate<sv_fl<WEIGHT_SWIZZLE_GRANULARITY>>();
    auto (&sf_up) = al.allocate<sv_fl<WEIGHT_SWIZZLE_GRANULARITY>>();
    rt_fp8e4m3<REG_M, REG_K> a_tiles[2];
    rt_fp8e4m3<REG_N, REG_K> b_tiles[4];
    rt_fl<REG_M, REG_N, ducks::rt_layout::col> accum[2];  // 0: gate accum., 1: up accum.
    rv_fl<REG_M, ducks::rv_layout::align> reg_sf_A;
    rv_fl<REG_N, ducks::rv_layout::ortho> reg_sf_W[2];

    const int warp_id = warpid();
    const int warp_row = warp_id / 2, warp_col = warp_id % 2;
    constexpr int k_iters = D_MODEL / BLOCK_K;

    const int num_valid_m_tiles = g.num_valid_ids[0] / BLOCK_M;
    constexpr int num_n_tiles = 2 * D_INTER / BLOCK_N;
    const int total_tiles = num_valid_m_tiles * num_n_tiles;
    const int num_tiles_per_cu = ceil_div(total_tiles, gridDim.x);
    const int chunk_size = 1;
    const int window_size = 1;
    const int base_bidx = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, chunk_size);

    for (int tile = 0; tile < num_tiles_per_cu && base_bidx + tile * gridDim.x < total_tiles; ++tile) {

        const int remap_bidx = base_bidx + tile * gridDim.x;
        int num_wgid_in_group = window_size * num_n_tiles;
        int group_id = remap_bidx / num_wgid_in_group;
        int first_pid_m = group_id * window_size;
        int group_size_m = min(num_valid_m_tiles - first_pid_m, window_size);
        int output_m = first_pid_m + ((remap_bidx % num_wgid_in_group) % group_size_m);
        int output_n = (remap_bidx % num_wgid_in_group) / group_size_m;
        int expert = g.sorted_expert_ids[output_m];

        constexpr int BYTES_PER_MEMCPY = NUM_THREADS * sizeof(float4) / sizeof(fp8e4m3);
        constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;
        constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;
        int tokens[BUFFER_SIZE_A];
        float4 a_buf[2][BUFFER_SIZE_A];
        float4 b_buf[2][BUFFER_SIZE_B];

        int lds_idx = 0;
        int reg_idx = 1;

        gather_tokens<NUM_THREADS>(tokens, g.sorted_token_ids, {0, 0, output_m, 0}, As[lds_idx]);
        for (int i = 0; i < 2; i++) { zero(accum[i]); }
        gather_load<NUM_THREADS>(As[lds_idx], g.A, {0, 0, output_m, 0}, tokens);
        G::load(Bs[lds_idx], g.B, {0, expert, output_n, 0});

        auto prefetch = [&](int k_tile, float4* a_dst, float4* b_dst) {
            load_global_to_register_buffer<2, false, NUM_THREADS>(b_dst, BUFFER_SIZE_B, g.B, {0, expert, output_n, k_tile}, Bs0);
            gather_load_global_to_register_buffer<NUM_THREADS>(a_dst, BUFFER_SIZE_A, g.A, {0, 0, output_m, k_tile}, tokens, As0);
        };
        auto commit = [&](a_st& As_d, b_st& Bs_d, const float4* a_src, const float4* b_src) {
            store_register_buffer_to_shared<NUM_THREADS>(As_d, a_src);
            store_register_buffer_to_shared<NUM_THREADS>(Bs_d, b_src);
        };
        auto compute = [&](a_st& As_c, b_st& Bs_c) {
            load(a_tiles[0], subtile_inplace<REG_M, REG_K>(As_c, {warp_row, 0}));
            load(b_tiles[0], subtile_inplace<REG_N, REG_K>(Bs_c, {warp_col, 0}));
            load(b_tiles[1], subtile_inplace<REG_N, REG_K>(Bs_c, {warp_col + 2, 0}));

            load(a_tiles[1], subtile_inplace<REG_M, REG_K>(As_c, {warp_row, 1}));
            load(b_tiles[2], subtile_inplace<REG_N, REG_K>(Bs_c, {warp_col, 1}));
            load(b_tiles[3], subtile_inplace<REG_N, REG_K>(Bs_c, {warp_col + 2, 1}));
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(6)");
            __builtin_amdgcn_sched_barrier(0);
            mma_ABt(accum[0], a_tiles[0], b_tiles[0], accum[0]);
            mma_ABt(accum[1], a_tiles[0], b_tiles[1], accum[1]);
            __builtin_amdgcn_sched_barrier(0);

            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            mma_ABt(accum[0], a_tiles[1], b_tiles[2], accum[0]);
            mma_ABt(accum[1], a_tiles[1], b_tiles[3], accum[1]);
            __builtin_amdgcn_sched_barrier(0);
        };
        
        prefetch(1, a_buf[reg_idx], b_buf[reg_idx]);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);
        
        #pragma unroll 2
        for (int K_TILE = 0; K_TILE + 2 < k_iters; ++K_TILE) {
            prefetch(K_TILE + 2, a_buf[reg_idx^1], b_buf[reg_idx^1]);
            compute(As[lds_idx], Bs[lds_idx]);
            asm volatile("s_waitcnt vmcnt(3)");
            __builtin_amdgcn_sched_barrier(0);
            
            commit(As[lds_idx^1], Bs[lds_idx^1], a_buf[reg_idx], b_buf[reg_idx]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            lds_idx^=1; reg_idx^=1;
        }
        // k_iter - 2
        compute(As[lds_idx], Bs[lds_idx]);
        asm volatile("s_waitcnt vmcnt(0)");
        __builtin_amdgcn_sched_barrier(0);
        
        commit(As[lds_idx^1], Bs[lds_idx^1], a_buf[reg_idx], b_buf[reg_idx]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        lds_idx^=1; reg_idx^=1;

        // k_iter - 1
        if (warp_id == 0) {
            gather_f32_sf_a<WARP_THREADS>(sf_A, g.sf_A, {output_m}, g.sorted_token_ids);
            load(sf_gate, g.sf_B, {expert, output_n});
            load(sf_up, g.sf_B, {expert, output_n + (D_INTER / WEIGHT_SWIZZLE_GRANULARITY)});
        }
        compute(As[lds_idx], Bs[lds_idx]);
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        load_sv_to_rv(reg_sf_A, subvec_inplace<REG_M>(sf_A, warp_row));
        load_sv_to_rv(reg_sf_W[0], subvec_inplace<REG_N>(sf_gate, warp_col));
        load_sv_to_rv(reg_sf_W[1], subvec_inplace<REG_N>(sf_up, warp_col));
        apply_row_sf(accum[0], accum[0], reg_sf_A);
        apply_col_sf(accum[0], accum[0], reg_sf_W[0]);
        apply_row_sf(accum[1], accum[1], reg_sf_A);
        apply_col_sf(accum[1], accum[1], reg_sf_W[1]);
        silu(accum[0], accum[0]);
        mul(accum[0], accum[0], accum[1]);
        scatter_store<TOP_K>(g.C, accum[0], {0, 0, output_m * 2 + warp_row, output_n * 2 + warp_col}, g.sorted_token_ids);
    }
}

void call(moe_stage1_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    // hipDeviceProp_t prop;
    // hipGetDeviceProperties(&prop, 0);
    // auto grid_dim = dim3(OCCUPANCY * multiProcessorCount);
    auto grid_dim = dim3(OCCUPANCY * 304);
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
