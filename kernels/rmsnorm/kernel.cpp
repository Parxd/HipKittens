#include "kittens.cuh"
#include <hip/hip_cooperative_groups.h>
#include "pyutils/pyutils.cuh"


constexpr int B = 16;
constexpr int H = 16;
constexpr int N = 4096;
constexpr int D = 128; 

#define NUM_WORKERS (4) 
#define NUM_THREADS (NUM_WORKERS*kittens::WARP_THREADS)

using G = kittens::group<NUM_WORKERS>;
using namespace kittens;


template <int _N> struct rmsnorm_globals{
    using x_gl = gl<bf16 , -1,-1,-1,-1>;
    using o_gl = gl<bf16 , -1,-1 , -1, -1>;
    using gamma_gl = gl<bf16 , -1 , -1 , -1 , -1>;

    x_gl x;
    o_gl o;
    gamma_gl gamma;
    float epsilon;

    const int n_per_tile = 4;
    const int n_tile_size = N / n_per_tile;

    dim3 grid() { return dim3(n_tile_size, B, 1); }
    dim3 block() { return dim3(NUM_THREADS); }
    size_t dynamic_shared_memory() { return 0; }
}; 

template<int _D>
__global__ void rmsnorm_hk(
    const rmsnorm_globals<_D> g 
){
    auto warpid = kittens::warpid();
    const int batch = blockIdx.y;
    const int seq_start = blockIdx.x * g.n_per_tile;
    const int seq_idx = seq_start + warpid; 

    rv<bf16 , _D> x_reg , gamma_reg , x_reg_squared;

    load(x_reg, g.x, {0, batch, seq_idx, 0});
    load(gamma_reg , g.gamma, {0, batch, seq_idx, 0});
    asm volatile("s_waitcnt vmcnt(0)"); 

    mul(x_reg_squared, x_reg, x_reg);

    bf16 x_var;
    sum(x_var, x_reg_squared);

    float var_f32 = __bfloat162float(x_var) / float(_D);
    float inv_rms_f32 = rsqrtf(var_f32 + g.epsilon);
    bf16 inv_rms = __float2bfloat16(inv_rms_f32);

    mul(x_reg, x_reg, inv_rms);
    mul(x_reg, x_reg, gamma_reg); 

    store(g.o, x_reg, {0, batch, seq_idx, 0});
}


template<int _D>
void dispatch_rmsnorm(rmsnorm_globals<_D> g) {  
    unsigned long mem_size = g.dynamic_shared_memory();
    hipFuncSetAttribute((void*)rmsnorm_hk<_D>, hipFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    rmsnorm_hk<_D><<<g.grid(), g.block(), mem_size>>>(g);
    hipDeviceSynchronize();
}

PYBIND11_MODULE(rms_norm_kernel, m) {
    m.doc() = "rms_norm_kernel python module";
    py::bind_function<dispatch_rmsnorm<D>>(m, "dispatch_rmsnorm", 
        &rmsnorm_globals<D>::x, 
        &rmsnorm_globals<D>::o, 
        &rmsnorm_globals<D>::gamma,
        &rmsnorm_globals<D>::epsilon
    );
}