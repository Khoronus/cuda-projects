#include <iostream>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_TILE_SIZE 16

template <typename T>
__global__ void wmma_gemm(const T *a, const T *b, float *c, int M, int N, int K);

template <>
__global__ void wmma_gemm<__half>(const __half *a, const __half *b, float *c, int M, int N, int K) {
    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, __half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_TILE_SIZE, WMMA_TILE_SIZE, WMMA_TILE_SIZE, float> c_frag;

    // Initialize the output to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Load the inputs
    wmma::load_matrix_sync(a_frag, a, WMMA_TILE_SIZE);
    wmma::load_matrix_sync(b_frag, b, WMMA_TILE_SIZE);

    // Perform the matrix multiplication
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    // Store the output
    wmma::store_matrix_sync(c, c_frag, WMMA_TILE_SIZE, wmma::mem_row_major);
}

template <>
__global__ void wmma_gemm<float>(const float *a, const float *b, float *c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

template <typename T>
void run_wmma_gemm(int M, int N, int K) {
    // Allocate and initialize host memory
    T *h_a = (T*)malloc(M * K * sizeof(T));
    T *h_b = (T*)malloc(K * N * sizeof(T));
    float *h_c = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            h_a[i] = static_cast<float>(rand()) / RAND_MAX;
        } else if constexpr (std::is_same<T, __half>::value) {
            h_a[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        }
    }

    for (int i = 0; i < K * N; ++i) {
        if constexpr (std::is_same<T, float>::value) {
            h_b[i] = static_cast<float>(rand()) / RAND_MAX;
        } else if constexpr (std::is_same<T, __half>::value) {
            h_b[i] = __float2half(static_cast<float>(rand()) / RAND_MAX);
        }
    }

    // Allocate device memory
    T *d_a, *d_b;
    float *d_c;
    cudaMalloc((void**)&d_a, M * K * sizeof(T));
    cudaMalloc((void**)&d_b, K * N * sizeof(T));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, M * K * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(T), cudaMemcpyHostToDevice);

    // Launch the kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    wmma_gemm<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, M, N, K);

    // Copy result from device to host
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

int main() {
    // Matrix size
    int M = 16;
    int N = 16;
    int K = 16;

    std::cout << "Running WMMA GEMM with float precision:" << std::endl;
    run_wmma_gemm<float>(M, N, K);

    std::cout << "Running WMMA GEMM with half precision:" << std::endl;
    run_wmma_gemm<__half>(M, N, K);

    std::cout << "GEMM completed successfully." << std::endl;
    return 0;
}
