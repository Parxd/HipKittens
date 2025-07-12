# AMD-benchmarking-harness


## Setup

Install PyTorch. Need to install special PyTorch that targets AMD. From the [PyTorch website](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3
```

Get the ThunderKittens-HIP repo
```
git submodule update --init --recursive
cd ThunderKittens-HIP
source env.src
```

Load rocm on your gpu: 
```bash
module avail
module load rocm/6.3.3
```
If there are version mismatches, the kernel results will be incorrect. 


### Directory structure
```
kernels/
    - HIP and TK implementations of various kernels
utils/
    - helper functions for benchmarking or profiling
```

### Usage

```bash 
cd AMD-benchmarking-harness/kernels/TK/gemm/bf16fp32/mi325x/256_256_64_16/
make 
python test_python.py
```


