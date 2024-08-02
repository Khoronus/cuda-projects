#ifndef WMMA_IMPL_H
#define WMMA_IMPL_H

#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_TILE_SIZE 16

template <typename T>
__global__ void wmma_gemm(const T *a, const T *b, float *c, int M, int N, int K);

template <>
__global__ void wmma_gemm<__half>(const __half *a, const __half *b, float *c, int M, int N, int K);

template <>
__global__ void wmma_gemm<float>(const float *a, const float *b, float *c, int M, int N, int K);

template <typename T>
void run_wmma_gemm(int M, int N, int K);

#include "wmma_impl.cuh"

#endif // WMMA_IMPL_H
