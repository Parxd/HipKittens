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
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 256;
constexpr int REG_M = 16;
constexpr int REG_N = 16;
constexpr int REG_K = 32;
constexpr int WEIGHT_SWIZZLE_GRANULARITY = BLOCK_N / 2;
constexpr size_t SMEM_BYTES = BLOCK_M * BLOCK_K + BLOCK_N * BLOCK_K;
static_assert(SMEM_BYTES <= 64 * 1024, "SMEM_BYTES exceeds gfx942 LDS size (64 KiB)");

using G = kittens::group<NUM_WARPS>;
using _gl_A = gl<fp8e4m3,1,1,-1,-1>;
using _gl_B = gl<fp8e4m3,1,-1,-1,-1>;
using _gl_C = gl<float,-1,-1,-1,-1>;
using _gl_meta = gl<int,1,1,1,-1>;

/**
 * @brief Gathers non-contiguous rows from a global tile into a shared tile selected by a mapping.
 *
 * @tparam ST The shared tile type.
 * @tparam GL The global tile type for the physical activations matrix.
 * @tparam GL_IDX The global tile type for the token mappings array.
 * @tparam COORD Coord type.
 * @param dst[out]  The destination shared tile.
 * @param src[in]   The source global tensor (unpermuted activations), shape [M, d_hidden].
 * @param idx[in]   Tile coordinate (m_tile, k_tile): m_tile is the M-tile index into the
 *                   permuted row space; k_tile is the column-block index into src's actual K axis.
 * @param token_mapping[in]  Array mapping each absolute permuted-space row index to its
 *                   corresponding row index in src. Only reads indices [m_tile * BLOCK_M, end_idx).
 * @param end_idx[in]  Absolute row index in permuted space up to which rows in this
 *                   M-tile are valid / non-padding.
 */
template
    int axis,
    int N_THREADS,
    ducks::st::all ST,
    ducks::gl::all GL,
    ducks::gl::all GL_IDX,
    ducks::coord::tile COORD = coord<ST>
>
__device__ inline void gather_load(
    ST& dst, const GL& src, const COORD& idx, const GL_IDX& token_mapping, int end_idx
) {
    using T = typename ST::dtype;
    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // coord. for token mapping, uses scaled permuted M index as column index, since it's a 1D tile
    coord<> tm_coord(0, 0, 0, unit_coord.r);
    // already-scaled physical K index w/o permuted M index
    // src_ptr is only offset in K-axis based on BLOCK_K coord., we do our own M-axis offset based on token IDs inside load loop 
    coord<> k_coord(0, 0, 0, unit_coord.c);

    const int valid_rows = end_idx - unit_coord.r;
    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[k_coord];
    typename GL_IDX::dtype *tm_ptr = (typename GL_IDX::dtype*)&token_mapping[tm_coord];

    uint32_t dst_ptr = reinterpret_cast<uintptr_t>(&dst.data[0]);
    const int laneid = threadIdx.x % N_THREADS;

    const int small_calls = 4;  // this should vary based on batch size?
    const int big_calls = (total_calls + small_calls - 1) / small_calls;
    float4    buf[small_calls];

    for (int i = 0; i < big_calls; i++) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < valid_rows) {
                int token_id = tm_ptr[row];  // TODO: use as-is or decode
                buf[j] = load_global_vec4_async((float4*) (src_ptr + (token_id * row_stride + col)));
            }
        }
        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt vmcnt(0)");
        #endif

        #pragma unroll
        for(int j = 0; j < small_calls; j++) {
            int load_idx = (offset + j) * N_THREADS + laneid;
            int row = load_idx / memcpy_per_row;
            int col = (load_idx % memcpy_per_row) * elem_per_memcpy;

            if (row < valid_rows) {
                store_shared_vec(dst.idx(dst_ptr, {row, col}), {buf[j].x, buf[j].y});
                store_shared_vec(dst.idx(dst_ptr, {row, col + elem_per_half_memcpy}), {buf[j].z, buf[j].w});
            }
        }

        #ifdef BUILTINS_ONLY
        __builtin_amdgcn_s_waitcnt(0);
        #else
        asm volatile("s_waitcnt lgkmcnt(0)");
        #endif
    }
}

struct micro_globals {
    _gl_A A;
    _gl_B B;
    _gl_C C;
    _gl_meta token_mapping;
    _gl_meta tile_to_expert;
    _gl_meta tile_to_m_limit;
    int num_valid_tiles;
    hipStream_t stream;

    dim3 block() { return dim3(NUM_WARPS * WARP_THREADS); }
    size_t dynamic_shared_memory() { return SMEM_BYTES; }
};

__global__ __launch_bounds__(NUM_WARPS * WARP_THREADS, 2)
void kernel(const micro_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    
    auto (&As) = al.allocate<st<fp8e4m3, BLOCK_M, BLOCK_K>>();
    auto (&Bs) = al.allocate<st<fp8e4m3, BLOCK_N, BLOCK_K>>();
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
        int expert = g.tile_to_expert[gl_m_tile];
        int end_token_idx = g.tile_to_m_limit[gl_m_tile];

        gather_load<2, NUM_WARPS * NUM_THREADS>(As, g.A, {0, 0, gl_m_tile, 0}, g.token_mapping, end_token_idx);
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

        }
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
