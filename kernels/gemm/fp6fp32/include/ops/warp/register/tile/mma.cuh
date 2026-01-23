/**
 * @file
 * @brief Matrix multiply-accumulate operations for tiles stored in registers.
 */

#pragma once

#include "../../../../common/common.cuh"
#include "../../../../types/types.cuh"

namespace kittens {

#ifdef KITTENS_CDNA4

// Type alias for 8-bit scale exponent used in MXFP6
using scale_t = uint32_t;

/**
 * @brief Non-scaled FP6 MFMA instruction (16x16x128).
 * 
 * Performs D = A * B + C where A and B are FP6 E2M3 format.
 * No per-block scaling is applied.
 */
__device__ static inline void mfma1616128_fp6(float2 (&D)[2],
                                             const fp6_e2m3_32 (&A)[1], 
                                             const fp6_e2m3_32 (&B)[1],
                                             const float2 (&C)[2]) {

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8x_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)D = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(const int8x_t*)&A[0],
        *(const int8x_t*)&B[0],
        *(floatx4_t*)C,
        2, // cbsz - A format (2 = FP6)
        2, // blgp - B format (2 = FP6)
        0, 0, 0, 0  // scale_a, scale_b, negate_a, negate_b
    );
}

/**
 * @brief Scaled FP6 MFMA instruction (16x16x128) for MXFP6 format with compile-time scales.
 * 
 * Performs D = (A * 2^SCALE_A) * (B * 2^SCALE_B) + C
 * where A and B are FP6 E2M3 format with per-32-element block scaling.
 * 
 * The scale factors are 8-bit unsigned exponents that multiply the
 * corresponding 32-element blocks by 2^scale.
 *
 * @tparam SCALE_A Compile-time scale exponent for A block (0-255)
 * @tparam SCALE_B Compile-time scale exponent for B block (0-255)
 * @param D Output accumulator (4 floats as 2 x float2)
 * @param A Input FP6 matrix block (32 packed FP6 values)
 * @param B Input FP6 matrix block (32 packed FP6 values)
 * @param C Input accumulator
 */
template<scale_t SCALE_A = 0, scale_t SCALE_B = 0>
__device__ static inline void mfma1616128_fp6_scaled(float2 (&D)[2],
                                                     const fp6_e2m3_32 (&A)[1], 
                                                     const fp6_e2m3_32 (&B)[1],
                                                     const float2 (&C)[2]) {

    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8x_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    *(floatx4_t*)D = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        *(const int8x_t*)&A[0],
        *(const int8x_t*)&B[0],
        *(floatx4_t*)C,
        2, // cbsz - A format (2 = FP6)
        2, // blgp - B format (2 = FP6)
        SCALE_A,    // Per-32 block scale for A (compile-time constant)
        SCALE_B,    // Per-32 block scale for B (compile-time constant)
        0, 0        // negate flags (not used)
    );
}

/**
 * @brief Scaled FP6 MFMA with runtime scales using inline assembly.
 * 
 * This version supports dynamic per-block scales loaded at runtime.
 * The scales are passed as VGPRs to the MFMA instruction.
 *
 * @param D Output accumulator (4 floats as 2 x float2)
 * @param A Input FP6 matrix block (32 packed FP6 values)
 * @param B Input FP6 matrix block (32 packed FP6 values)
 * @param C Input accumulator
 * @param scale_a Runtime scale exponent for A block
 * @param scale_b Runtime scale exponent for B block
 */
__device__ static inline void mfma1616128_fp6_scaled_dynamic(float2 (&D)[2],
                                                              const fp6_e2m3_32 (&A)[1], 
                                                              const fp6_e2m3_32 (&B)[1],
                                                              const float2 (&C)[2],
                                                              scale_t scale_a,
                                                              scale_t scale_b) {
    typedef __attribute__((__vector_size__(8 * sizeof(int)))) int int8x_t;
    typedef __attribute__((__vector_size__(4 * sizeof(float)))) float floatx4_t;

    // Use inline assembly for dynamic scales
    // The MFMA scale instruction takes scales from VGPRs when using op_sel
    floatx4_t c_vec = *(floatx4_t*)C;
    floatx4_t d_vec;
    
    // Pack scales into a single register pair for the instruction
    // The instruction expects scales in specific VGPR format
    uint32_t packed_scales = (scale_b << 16) | scale_a;
    
    asm volatile(
        "v_mfma_scale_f32_16x16x128_f8f6f4 %0, %1, %2, %3, 2, 2, %4, %5, 0, 0"
        : "=v"(d_vec)
        : "v"(*(const int8x_t*)&A[0]), 
          "v"(*(const int8x_t*)&B[0]),
          "v"(c_vec),
          "v"(scale_a),
          "v"(scale_b)
    );
    
    *(floatx4_t*)D = d_vec;
}

/**
 * @brief Base non-scaled FP6 MMA operation on register tiles.
 */
__device__ static inline void mma_ABt_base(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &a,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
       mfma1616128_fp6(d.data, a.data, b.data, c.data);
}

/**
 * @brief Base scaled FP6 MMA operation on register tiles (MXFP6) with compile-time scales.
 *
 * Each 32-element block in A and B is scaled by 2^SCALE_A and 2^SCALE_B respectively.
 */
template<scale_t SCALE_A = 0, scale_t SCALE_B = 0>
__device__ static inline void mma_ABt_base_scaled(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &a,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::accumulator> &c) {
       mfma1616128_fp6_scaled<SCALE_A, SCALE_B>(d.data, a.data, b.data, c.data);
}

/**
 * @brief Base scaled FP6 MMA operation with runtime dynamic scales.
 */
__device__ static inline void mma_ABt_base_scaled_dynamic(rt_base<float, ducks::rt_layout::accumulator> &d,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &a,
                                     const rt_base<fp6_e2m3, ducks::rt_layout::row> &b,
                                     const rt_base<float, ducks::rt_layout::accumulator> &c,
                                     scale_t scale_a,
                                     scale_t scale_b) {
       mfma1616128_fp6_scaled_dynamic(d.data, a.data, b.data, c.data, scale_a, scale_b);
}

#endif // KITTENS_CDNA4


/**
 * @brief Dot product operation for row layout.
 *
 * This function performs the dot product operation
 * using the `hmma16816` function.
 *
 * @tparam N The number of row tiles.
 * @tparam K The number of column tiles for the A matrix and row tiles for the B matrix.
 * @tparam M The number of column tiles for the B matrix.
 * @param[out] d The output rt_fl<N, M, row_layout> accumulator.
 * @param[in] a The first input rt_bf<N, K, row_layout> matrix.
 * @param[in] b The second input rt_bf<M, K, row_layout> matrix in row-major mode.
 * @param[in] c The input rt_fl<N, M, row_layout> accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<ducks::rt::accumulator_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
#else
template<ducks::rt::col_layout D, ducks::rt::row_layout A, ducks::rt::row_layout B, ducks::rt::col_layout C>
#endif
__device__ static inline void mma_ABt(D &d,
                                const A &a,
                                const B &b, // notice row and (M, K) instead of col and (K, M)
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows); // Check D matches A, B
    static_assert(A::cols == B::cols); // Check reduction dim is same
    static_assert(D::rows == C::rows && D::cols == C::cols); // Check D matches C

    #ifdef KITTENS_CDNA4
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>) ||
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp6_e2m3> &&
            std::is_same_v<typename B::T, fp6_e2m3> && std::is_same_v<typename C::T, float>)
    );
    #else
    static_assert(
        (std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, bf16> &&
            std::is_same_v<typename B::T, bf16> && std::is_same_v<typename C::T, float>) ||
        (std::is_same_v<typename D::T, half> && std::is_same_v<typename A::T, half> &&
            std::is_same_v<typename B::T, half> && std::is_same_v<typename C::T, half>)
    );
    #endif

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Scaled dot product operation for MXFP6 format with compile-time uniform scales.
 *
 * Performs D = (A * 2^SCALE_A) * (B * 2^SCALE_B)^T + C where scale factors
 * are applied uniformly to all blocks.
 *
 * @tparam SCALE_A Compile-time scale exponent for all A blocks.
 * @tparam SCALE_B Compile-time scale exponent for all B blocks.
 * @param[out] d The output accumulator.
 * @param[in] a The first input FP6 matrix.
 * @param[in] b The second input FP6 matrix.
 * @param[in] c The input accumulator matrix.
 */
#ifdef KITTENS_CDNA4
template<scale_t SCALE_A, scale_t SCALE_B,
         ducks::rt::accumulator_layout D, ducks::rt::row_layout A, 
         ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
__device__ static inline void mma_ABt_scaled(D &d,
                                const A &a,
                                const B &b,
                                const C &c) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp6_e2m3> &&
                  std::is_same_v<typename B::T, fp6_e2m3> && std::is_same_v<typename C::T, float>);

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base_scaled<SCALE_A, SCALE_B>(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base_scaled<SCALE_A, SCALE_B>(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m]
                );
            }
        }
    }
}

/**
 * @brief Scaled dot product with dynamic per-tile scales.
 *
 * Uses inline assembly to support runtime-variable scale factors.
 * This is useful for MXFP6 where different blocks have different scales.
 */
template<ducks::rt::accumulator_layout D, ducks::rt::row_layout A, 
         ducks::rt::row_layout B, ducks::rt::accumulator_layout C>
__device__ static inline void mma_ABt_scaled_dynamic(D &d,
                                const A &a,
                                const B &b,
                                const C &c,
                                const scale_t scales_a[][A::width],
                                const scale_t scales_b[][B::width]) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    static_assert(std::is_same_v<typename D::T, float> && std::is_same_v<typename A::T, fp6_e2m3> &&
                  std::is_same_v<typename B::T, fp6_e2m3> && std::is_same_v<typename C::T, float>);

    #pragma unroll
    for(int n = 0; n < D::height; n++) {
        #pragma unroll
        for(int m = 0; m < D::width; m++) {
            mma_ABt_base_scaled_dynamic(
                d.tiles[n][m],
                a.tiles[n][0],
                b.tiles[m][0],
                c.tiles[n][m],
                scales_a[n][0],
                scales_b[m][0]
            );
            #pragma unroll
            for(int k = 1; k < A::width; k++) {
                mma_ABt_base_scaled_dynamic(
                    d.tiles[n][m],
                    a.tiles[n][k],
                    b.tiles[m][k],
                    d.tiles[n][m],
                    scales_a[n][k],
                    scales_b[m][k]
                );
            }
        }
    }
}

/**
 * @brief Single-tile scaled MMA operation with compile-time scales.
 */
template<int n, int m, int k, scale_t SCALE_A, scale_t SCALE_B, 
         typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_one_scaled(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    
    mma_ABt_base_scaled<SCALE_A, SCALE_B>(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m]
    );
}

/**
 * @brief Single-tile scaled MMA with dynamic runtime scales.
 */
template<int n, int m, int k, typename D, typename A, typename B, typename C>
__device__ inline void mma_ABt_one_scaled_dynamic(D& d_mma, const A& a_mma, const B& b_mma, const C& c_mma,
                                                   scale_t scale_a, scale_t scale_b) {
    static_assert(D::rows == A::rows && D::cols == B::rows);
    static_assert(A::cols == B::cols);
    static_assert(D::rows == C::rows && D::cols == C::cols);
    
    mma_ABt_base_scaled_dynamic(
        d_mma.tiles[n][m],
        a_mma.tiles[n][k],
        b_mma.tiles[m][k],
        c_mma.tiles[n][m],
        scale_a,
        scale_b
    );
}
#endif

}