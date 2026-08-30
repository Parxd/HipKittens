#include "kittens.cuh"

using namespace kittens;

extern "C" __device__ float
llvm_amdgcn_raw_buffer_load_f32(i32x4 srsrc, uint32_t voffset, uint32_t soffset, uint32_t coherency)
    __asm("llvm.amdgcn.raw.buffer.load.f32");

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
 *                  PERMUTED row space; k_tile is the column-block index into src's actual K axis.
 * @param sorted_token_ids[in]  Array mapping each permuted-space row index to its
 *                              corresponding bit-packed int, which encodes the token index in 
 *                              src. in the lower 24-bits, and its top-K slot in the upper 8.
 *                              Padding tokens are marked as M, as valid tokens range from [0, M - 1].
 *                              Only reads indices [m_tile * BLOCK_M, m_tile * 2 * BLOCK_M).
 */
template<int N_THREADS,
        ducks::st::all ST,
        ducks::gl::all GL,
        ducks::gl::all GL_IDX,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void gather_load(
    ST& dst, const GL& src, const COORD& idx, const GL_IDX& sorted_token_ids
) {
    using T = typename ST::dtype;
    constexpr int axis = 2;

    const int row_stride = src.template stride<axis>();
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename ST::dtype);
    constexpr int elem_per_half_memcpy = sizeof(float2) / sizeof(typename ST::dtype);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_calls = (ST::cols * ST::rows + N_THREADS*elem_per_memcpy-1) / (N_THREADS*elem_per_memcpy);

    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    // coord. for token mapping, uses scaled permuted M index as column index, since it's a 1D tile
    coord<> tm_coord(0, 0, 0, unit_coord.r);
    // already-scaled physical K index w/o permuted M index
    // src_ptr only offset in K-axis based on BLOCK_K coord., we do our own M-axis offset based on token IDs inside load loop 
    coord<> k_coord(0, 0, 0, unit_coord.c);

    typename GL::dtype *src_ptr = (typename GL::dtype*)&src[k_coord];
    typename GL_IDX::dtype *tm_ptr = (typename GL_IDX::dtype*)&sorted_token_ids[tm_coord];

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
            // int topk_slot = packed >> 24;  // don't need if not fusing weight reduction

            // TODO: compare against raw_buffer_load w/ hardware supported OOB reads to avoid this conditional
            if (token_id != src.rows()) {
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
            int token_id = tm_ptr[row] & 0xFFFFFF;

            if (token_id != src.rows()) {
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

/**
 * @brief Gathers non-contiguous rows from a global tile into register buffers selected by a mapping.
 *
 * @tparam ST The shared tile type.
 * @tparam GL The global tile type for the physical activations matrix.
 * @tparam GL_IDX The global tile type for the token mappings array.
 * @tparam COORD Coord type.
 * @param reg_buffer[out]  The destination register buffers.
 * @param buffer_size[in]  Size of register buffers, in units of float4's (i.e. 128-bits)
 * @param src[in]   The source global tensor (unpermuted activations), shape [M, d_hidden].
 * @param idx[in]   Tile coordinate (m_tile, k_tile): m_tile is the M-tile index into the
 *                  PERMUTED row space; k_tile is the column-block index into src's actual K axis.
 * @param sorted_token_ids[in]  Array mapping each permuted-space row index to its
 *                              corresponding bit-packed int, which encodes the token index in 
 *                              src. in the lower 24-bits, and its top-K slot in the upper 8.
 *                              Padding tokens are marked as M, as valid tokens range from [0, M - 1].
 *                              Only reads indices [m_tile * BLOCK_M, m_tile * 2 * BLOCK_M).
 */
template<int N_THREADS,
        ducks::st::all ST, 
        ducks::gl::all GL,
        ducks::gl::all GL_IDX,
        ducks::coord::tile COORD = coord<ST>
>
__device__ inline void gather_load_global_to_register_buffer(
    float4* reg_buffer, const int buffer_size, const GL& src, const COORD& idx, const GL_IDX& sorted_token_ids
) {
    using T = typename ST::dtype;
    constexpr int axis = 2;
    
    constexpr int elem_per_memcpy = sizeof(float4) / sizeof(T);
    constexpr int memcpy_per_row = ST::cols / elem_per_memcpy;
    constexpr int total_chunks = (ST::rows * ST::cols) / elem_per_memcpy;
    constexpr int total_calls = (total_chunks + N_THREADS - 1) / N_THREADS;
    constexpr int small_calls = 4;
    const int big_calls = (total_calls + small_calls - 1) / small_calls;

    const int row_stride = src.template stride<axis>();
    const int row_stride_bytes = row_stride * sizeof(T);
    coord<> unit_coord = idx.template unit_coord<axis, 3>();
    coord<> tm_coord(0, 0, 0, unit_coord.r);
    coord<> k_coord(0, 0, 0, unit_coord.c);
    T* base_ptr = (T*)&src[k_coord];
    typename GL_IDX::dtype *tm_ptr = (typename GL_IDX::dtype*)&sorted_token_ids[tm_coord];
    const int laneid = threadIdx.x % N_THREADS;

    const int total_bytes = row_stride * src.rows() * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, row_stride_bytes);

    int buf_idx = 0;
    for (int i = 0; i < big_calls && buf_idx < buffer_size; ++i) {
        const int offset = i * small_calls;
        #pragma unroll
        for (int j = 0; j < small_calls; ++j) {
            const int chunk_idx = (offset + j) * N_THREADS + laneid;
            if (chunk_idx < total_chunks && buf_idx < buffer_size) {
                int row = chunk_idx / memcpy_per_row;
                int col = (chunk_idx % memcpy_per_row) * elem_per_memcpy;
                int token_id = tm_ptr[row] & 0xFFFFFF;
                int flat_offset = token_id * row_stride + col;
                int byte_offset = flat_offset * sizeof(T);
                __uint128_t raw = llvm_amdgcn_raw_buffer_load_b128(srsrc, byte_offset, 0, 0);
                reg_buffer[buf_idx] = *reinterpret_cast<float4*>(&raw);
                buf_idx++;
            }
        }
    }
}

template<int N_THREADS,
        ducks::sv::all SV, 
        ducks::gl::all GL,
        ducks::gl::all GL_IDX,
        ducks::coord::tile COORD = coord<SV>
>
__device__ inline void gather_sf_a(
    SV& dst, const GL& src, const COORD& idx, const GL_IDX& sorted_token_ids
) {
    using T = float;

    int total_calls = (dst.length + N_THREADS - 1) / N_THREADS;
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    T* base_ptr = (T*)&src[unit_coord];
    typename GL_IDX::dtype* tm_ptr = (typename GL_IDX::dtype*)&sorted_token_ids[unit_coord];
    int laneid = threadIdx.x % N_THREADS;

    int total_bytes = dst.length * sizeof(T);
    i32x4 srsrc = make_srsrc(base_ptr, total_bytes, sizeof(T));
    
}