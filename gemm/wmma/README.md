# warp matrix multiply and accumulate

GEMM (General Matrix Multiply) code. Example of usage of WMMA (Warp Matrix Multiply and Accumulate).

## Setup

nvcc -arch=sm_80 -o gemm wmma.cu

or

```
mkdir build
cd build
cmake ..
make
```
