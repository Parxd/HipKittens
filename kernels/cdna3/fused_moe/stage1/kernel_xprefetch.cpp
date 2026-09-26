#include "kittens.cuh" 
#include "pyutils/pyutils.cuh"
#include "utils.cpp"

using namespace kittens;

#define NUM_WARPS 4
#define SPLIT_K False
// 1: emit the epilogue scatter stores via inline asm so the compiler's waitcnt pass doesn't see
// pending stores mixed with the cross-tile prefetch loads (which on gfx942 can force vmcnt(0)
// and drain the next tile's in-flight k=1 prefetch).
#define SCATTER_STORE_ASM 0

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
static_assert(BLOCK_M == WEIGHT_SWIZZLE_GRANULARITY, "scale-factor loads map one lane per row/channel");

constexpr int BYTES_PER_MEMCPY = NUM_THREADS * sizeof(float4) / sizeof(fp8e4m3);
constexpr int BUFFER_SIZE_A = (BLOCK_M * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;
constexpr int BUFFER_SIZE_B = (BLOCK_N * BLOCK_K + BYTES_PER_MEMCPY - 1) / BYTES_PER_MEMCPY;

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

// Per-thread routing metadata for one output tile. Loaded one tile ahead so that none of the
// dependent (token id -> activation row / scale) loads sit on the critical path.
struct tile_meta {
    int m, n;
    int expert_raw;              // VGPR copy; readfirstlane at use so the load stays a VMEM load
    int tokens[BUFFER_SIZE_A];   // A-gather rows for this thread
    int sf_token;                // per-token scale row (warp 0, lanes < BLOCK_M)
    int packed[4];               // sorted_token_ids entries for this lane's 4 output rows
};

__device__ inline void buffer_store_b16_nt(uint32_t bits, i32x4 srsrc, uint32_t byte_offset) {
    asm volatile("buffer_store_short %0, %1, %2, 0 offen nt"
                 :
                 : "v"(bits), "v"(byte_offset), "s"(srsrc)
                 : "memory");
}

template<int TOP_K_, ducks::rt::all RT, ducks::gl::all GL>
__device__ inline void scatter_store_preloaded(const GL& dst, const RT& src, int col_base, const int (&packed)[4]) {
    static_assert(RT::height == 1 && RT::width == 1, "packed[] holds exactly one 16x16 tile's rows per lane");
    using T = bf16;

    T* base_ptr = (T*)dst.raw_ptr + col_base;
    const int row_stride = dst.cols();
    const int total_bytes = row_stride * dst.rows() * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride * sizeof(T));

    const int sentinel = dst.rows() / TOP_K_;
    const int col_offset = kittens::laneid() % 16;
    const float* flat = reinterpret_cast<const float*>(src.tiles[0][0].data);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int token_id = packed[k] & 0x00FFFFFF;
        const int topk_slot = (packed[k] & 0xFF000000) >> 24;
        const int flat_offset = (token_id * TOP_K_ + topk_slot) * row_stride + col_offset;
        if (token_id != sentinel) {
            uint16_t bits = __builtin_bit_cast(uint16_t, __float2bfloat16(flat[k]));
#if SCATTER_STORE_ASM
            buffer_store_b16_nt(bits, srsrc, flat_offset * sizeof(T));
#else
            llvm_amdgcn_raw_buffer_store_b16(bits, srsrc, flat_offset * sizeof(T), 0, 0b10);
#endif
        }
    }
}

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
    const int lane = laneid();
    const int warp_row = warp_id / 2, warp_col = warp_id % 2;
    constexpr int k_iters = D_MODEL / BLOCK_K;
    static_assert(k_iters % 2 == 0 && k_iters >= 4);

    const int num_valid_m_tiles = g.num_valid_ids[0] / BLOCK_M;
    constexpr int num_n_tiles = 2 * D_INTER / BLOCK_N;
    const int total_tiles = num_valid_m_tiles * num_n_tiles;
    const int chunk_size = CUS_PER_XCD;
    constexpr int window_size = 1;

    int remap_bidx = chiplet_transform_chunked(blockIdx.x, gridDim.x, NUM_XCDS, chunk_size);
    if (remap_bidx >= total_tiles) return;

    const i32x4 expert_rsrc = make_srsrc(g.sorted_expert_ids.raw_ptr, g.sorted_expert_ids.cols() * sizeof(int));
    const i32x4 sf_a_rsrc   = make_srsrc(g.sf_A.raw_ptr, g.sf_A.cols() * sizeof(float));  // padding token id == M lands OOB -> 0
    const i32x4 sf_b_rsrc   = make_srsrc(g.sf_B.raw_ptr, g.sf_B.rows() * g.sf_B.cols() * sizeof(float));
    const uint32_t sf_A_ptr    = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sf_A.data[0]));
    const uint32_t sf_gate_ptr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sf_gate.data[0]));
    const uint32_t sf_up_ptr   = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&sf_up.data[0]));

    auto load_tile_meta = [&](int bidx) {
        tile_meta t;
        const int num_wgid_in_group = window_size * num_n_tiles;
        const int group_id = bidx / num_wgid_in_group;
        const int first_pid_m = group_id * window_size;
        const int group_size_m = min(num_valid_m_tiles - first_pid_m, window_size);
        t.m = first_pid_m + ((bidx % num_wgid_in_group) % group_size_m);
        t.n = (bidx % num_wgid_in_group) / group_size_m;

        const int* sorted_ids = g.sorted_token_ids.raw_ptr;
        const int row0 = t.m * BLOCK_M;
        gather_tokens<NUM_THREADS>(t.tokens, g.sorted_token_ids, {0, 0, t.m, 0}, As[0]);
        t.expert_raw = __builtin_bit_cast(int, llvm_amdgcn_raw_buffer_load_f32(expert_rsrc, t.m * sizeof(int), 0, 0));
        t.sf_token = sorted_ids[row0 + lane % BLOCK_M] & 0x00FFFFFF;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            t.packed[k] = sorted_ids[row0 + warp_row * REG_M + (lane / 16) * 4 + k];
        }
        return t;
    };

    auto issue_sf_load = [&](const tile_meta& t, int expert) -> float {
        if (warp_id == 0) {
            return llvm_amdgcn_raw_buffer_load_f32(sf_a_rsrc, t.sf_token * sizeof(float), 0, 0);
        } else if (warp_id <= 2) {
            const int channel = t.n * WEIGHT_SWIZZLE_GRANULARITY + lane % WEIGHT_SWIZZLE_GRANULARITY
                              + (warp_id == 2 ? D_INTER : 0);
            return llvm_amdgcn_raw_buffer_load_f32(sf_b_rsrc, (expert * g.sf_B.cols() + channel) * sizeof(float), 0, 0);
        }
        return 0.0f;
    };

    auto commit_sf = [&](float v) {
        if (lane < BLOCK_M) {
            if      (warp_id == 0) store_shared_f32(sf_A.idx(sf_A_ptr, lane), v);
            else if (warp_id == 1) store_shared_f32(sf_gate.idx(sf_gate_ptr, lane), v);
            else if (warp_id == 2) store_shared_f32(sf_up.idx(sf_up_ptr, lane), v);
        }
    };

    float4 a_buf[2][BUFFER_SIZE_A];
    float4 b_buf[2][BUFFER_SIZE_B];

    auto prefetch = [&](const tile_meta& t, int k_tile, float4* a_dst, float4* b_dst) {
        const int expert = __builtin_amdgcn_readfirstlane(t.expert_raw);
        load_global_to_register_buffer<2, false, NUM_THREADS>(b_dst, BUFFER_SIZE_B, g.B, {0, expert, t.n, k_tile}, Bs[0]);
        gather_load_global_to_register_buffer<NUM_THREADS>(a_dst, BUFFER_SIZE_A, g.A, {0, 0, t.m, k_tile}, t.tokens, As[0]);
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
        // __builtin_amdgcn_sched_barrier(0);  // no compiler barrier here to overlap w/ commit for LDS[K_TILE + 1]
    };

    // Invariant at the top of every tile: As[0]/Bs[0] hold k=0, b_buf[1]/a_buf[1] have k=1 in flight.
    tile_meta cur = load_tile_meta(remap_bidx);
    prefetch(cur, 0, a_buf[0], b_buf[0]);
    prefetch(cur, 1, a_buf[1], b_buf[1]);
    commit(As[0], Bs[0], a_buf[0], b_buf[0]);
    asm volatile("s_waitcnt lgkmcnt(0)");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    while (true) {
        const int next_bidx = remap_bidx + gridDim.x;
        const bool has_next = next_bidx < total_tiles;
        const int expert = __builtin_amdgcn_readfirstlane(cur.expert_raw);

        const float sf_val = issue_sf_load(cur, expert);
        tile_meta nxt = cur;
        if (has_next) nxt = load_tile_meta(next_bidx);

        for (int i = 0; i < 2; i++) { zero(accum[i]); }

        for (int K_TILE = 0; K_TILE + 2 < k_iters; K_TILE += 2) {
            prefetch(cur, K_TILE + 2, a_buf[0], b_buf[0]);
            compute(As[0], Bs[0]);
            commit(As[1], Bs[1], a_buf[1], b_buf[1]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);

            prefetch(cur, K_TILE + 3, a_buf[1], b_buf[1]);
            compute(As[1], Bs[1]);
            commit(As[0], Bs[0], a_buf[0], b_buf[0]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_s_barrier();
            __builtin_amdgcn_sched_barrier(0);
        }

        // k_iters - 2: a_buf[0]/b_buf[0] are free, so start the next tile's k=0.
        commit_sf(sf_val);
        if (has_next) prefetch(nxt, 0, a_buf[0], b_buf[0]);
        __builtin_amdgcn_sched_barrier(0);

        compute(As[0], Bs[0]);
        commit(As[1], Bs[1], a_buf[1], b_buf[1]);
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();  // for K-1 LDS tile & scale factors
        __builtin_amdgcn_sched_barrier(0);

        // k_iters - 1: a_buf[1]/b_buf[1] are free, so start the next tile's k=1.
        if (has_next) prefetch(nxt, 1, a_buf[1], b_buf[1]);
        load_sv_to_rv(reg_sf_A, subvec_inplace<REG_M>(sf_A, warp_row));
        load_sv_to_rv(reg_sf_W[0], subvec_inplace<REG_N>(sf_gate, warp_col));
        load_sv_to_rv(reg_sf_W[1], subvec_inplace<REG_N>(sf_up, warp_col));
        compute(As[1], Bs[1]);
        __builtin_amdgcn_sched_barrier(0);

        apply_row_sf(accum[0], accum[0], reg_sf_A);
        apply_col_sf(accum[0], accum[0], reg_sf_W[0]);
        apply_row_sf(accum[1], accum[1], reg_sf_A);
        apply_col_sf(accum[1], accum[1], reg_sf_W[1]);
        silu(accum[0], accum[0]);
        mul(accum[0], accum[0], accum[1]);

        // As[0]/Bs[0] were last read by compute(k_iters - 2), which is behind the previous barrier.
        if (has_next) commit(As[0], Bs[0], a_buf[0], b_buf[0]);
        scatter_store_preloaded<TOP_K>(g.C, accum[0], (cur.n * 2 + warp_col) * REG_N, cur.packed);

        if (!has_next) break;
        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_barrier();
        __builtin_amdgcn_sched_barrier(0);

        remap_bidx = next_bidx;
        cur = nxt;
    }
}

void call(moe_stage1_globals g) {
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)kernel, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    auto grid_dim = dim3(OCCUPANCY * NUM_CUS);
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
