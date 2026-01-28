import torch
import torch.nn as nn
import rms_norm_kernel

B = 16    
N = 4096      
D = 128      
EPSILON = 1e-6

def pytorch_rmsnorm(x, gamma, eps=1e-6):
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return x_normed * gamma

def test_rmsnorm():
    x = torch.randn(1, B, N, D, dtype=torch.bfloat16, device='cuda')
    
    gamma_1d = torch.ones(D, dtype=torch.bfloat16, device='cuda')
    gamma = gamma_1d.view(1, 1, 1, D).expand(1, B, N, D).contiguous()
    o = torch.zeros_like(x)
    ref_output = pytorch_rmsnorm(x, gamma_1d.view(1, 1, 1, D), EPSILON)
    
    rms_norm_kernel.dispatch_rmsnorm(x, o, gamma, EPSILON)
    
    torch.cuda.synchronize()
    
    max_diff = (o - ref_output).abs().max().item()
    mean_diff = (o - ref_output).abs().mean().item()
    
    print(f"Max absolute difference:  {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    rtol, atol = 1e-1, 3e-2
    if torch.allclose(o, ref_output, rtol=rtol, atol=atol):
        print("✓ PASSED: Results match within tolerance")
    else:
        print("✗ FAILED: Results differ too much")
        
        # Debug info
        print(f"\nSample values (first 5 elements of [0,0,0,:]):")
        print(f"  HK output: {o[0,0,0,:5]}")
        print(f"  Reference: {ref_output[0,0,0,:5]}")

def benchmark_rmsnorm(warmup=10, iters=100):
    x = torch.randn(1, B, N, D, dtype=torch.bfloat16, device='cuda')
    gamma = torch.ones(1, B, N, D, dtype=torch.bfloat16, device='cuda')
    o = torch.zeros_like(x)
    
    # Warmup
    for _ in range(warmup):
        rms_norm_kernel.dispatch_rmsnorm(x, o, gamma, EPSILON)
    torch.cuda.synchronize()
    
    # Benchmark HK
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        rms_norm_kernel.dispatch_rmsnorm(x, o, gamma, EPSILON)
    end.record()
    torch.cuda.synchronize()
    
    hk_time = start.elapsed_time(end) / iters
    
    # Benchmark PyTorch
    start.record()
    for _ in range(iters):
        _ = pytorch_rmsnorm(x, gamma, EPSILON)
    end.record()
    torch.cuda.synchronize()
    
    pt_time = start.elapsed_time(end) / iters
    
    print(f"\nBenchmark ({iters} iterations):")
    print(f"  HipKittens: {hk_time:.4f} ms")
    print(f"  PyTorch:    {pt_time:.4f} ms")
    print(f"  Speedup:    {pt_time/hk_time:.2f}x")

if __name__ == "__main__":
    print("=== RMSNorm Correctness Test ===")
    test_rmsnorm()
    
    print("\n=== RMSNorm Benchmark ===")
    benchmark_rmsnorm()