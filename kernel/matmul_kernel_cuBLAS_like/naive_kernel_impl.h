#ifndef NAIVE_KERNEL_IMPL_H
#define NAIVE_KERNEL_IMPL_H

// https://gist.github.com/ekzhang/32f5ad7123359f2a1e0b143250742211

#include <cassert>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line);

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line);

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);

void matmul_naive(const float alpha, const float beta, const int N, const int M, const int K, float *a, float *b, float *c, float *d_a, float *d_b, float *d_c);

#include "naive_kernel_impl.cuh"

#endif // NAIVE_KERNEL_IMPL_H