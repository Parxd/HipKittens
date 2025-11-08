#include "global_to_register.cuh"

#ifdef TEST_WARP_MEMORY_VEC_GLOBAL_TO_REGISTER

template<typename T>
struct reg_vec_load_store {
    using dtype = T;
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::rv_layout::all L> using valid = std::bool_constant<NW == 1 && S<=64 
    >; // this is warp-level
    static inline const std::string test_identifier = std::is_same_v<dtype, kittens::bf16> ? "reg_vec_loadstore_gmem=bf16" :
                                                      std::is_same_v<dtype, kittens::half> ? "reg_vec_loadstore_gmem=half" :
                                                                                             "reg_vec_loadstore_gmem=float";
    template<typename RT_SHAPE, typename ST_SHAPE, int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __host__ static void host_func(const std::vector<float> &i_ref, std::vector<float> &o_ref) {
        o_ref = i_ref; // overwrite the whole thing
    }
    template<typename RT_SHAPE, typename ST_SHAPE, typename dtype, int S, int NW, kittens::ducks::gl::all GL, kittens::ducks::rv_layout::all L> __device__ static void device_func(const GL &input, const GL &output) {
        // Only testing row layout tile vectors
        kittens::rv<dtype, RT_SHAPE::cols*S, RT_SHAPE::cols, RT_SHAPE, L> reg_vec;
        kittens::load(reg_vec, input, {});
        __builtin_amdgcn_s_waitcnt(0);
        __builtin_amdgcn_s_barrier();
        kittens::store(output, reg_vec, {});
    }
};

void warp::memory::vec::global_to_register::tests(test_data &results) {
    std::cout << "\n ----- Starting ops/warp/memory/vec/global_to_register tests! -----\n" << std::endl;
    constexpr int SIZE = INTENSITY_0 ? 1  :
                         INTENSITY_1 ? 2  :
                         INTENSITY_2 ? 4  : 
                         INTENSITY_3 ? 8  :
                         INTENSITY_4 ? 16 : -1;
                         
    using DEFAULT_ST_SHAPE = kittens::ducks::st_shape::st_16x16;

    using RT_SHAPE_1 = kittens::ducks::rt_shape::rt_16x32;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_1, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_1, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_1, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);

    using RT_SHAPE_2 = kittens::ducks::rt_shape::rt_32x16;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_2, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_2, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_2, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);

    using RT_SHAPE_3 = kittens::ducks::rt_shape::rt_16x16;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_3, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_3, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_3, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);

    using RT_SHAPE_4 = kittens::ducks::rt_shape::rt_32x32;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_4, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_4, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_4, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);


    using RT_SHAPE_5 = kittens::ducks::rt_shape::rt_32x32_8;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_5, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_5, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_5, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);

    using RT_SHAPE_6 = kittens::ducks::rt_shape::rt_16x32_4;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_6, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_6, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_6, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);

    using RT_SHAPE_7 = kittens::ducks::rt_shape::rt_32x16_4;
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_7, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::naive>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_7, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::ortho>::run(results);
    sweep_gmem_type_1d_warp<reg_vec_load_store, RT_SHAPE_7, DEFAULT_ST_SHAPE, SIZE, kittens::ducks::rv_layout::align>::run(results);
}
#endif