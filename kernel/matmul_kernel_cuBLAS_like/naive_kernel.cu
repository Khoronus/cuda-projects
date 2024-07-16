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
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i{0}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (int i{0}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

#define MAX_ERR 1e-4

#define CEIL_DIV(x, y) ((x) + (y) - 1) / (y)
#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
    // compute position in C that this thread is responsible for
    const uint x = blockIdx.x * blockDim.y + threadIdx.y;
    const uint y = blockIdx.y * blockDim.x + threadIdx.x;

    // `if` condition is necessary for when M or N aren't multiples of 32.
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        // C = α*(A@B)+β*C
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

void matmul_naive(const float alpha, const float beta, const int N, const int M, const int K, float *a, float *b, float *c, float *d_a, float *d_b, float *d_c) {

    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    // create as many blocks as necessary to map all of C
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);
    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    // Performance measurement.
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_a, d_b, beta, d_c);
    checkCudaErrors(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
}

void test_naive() {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    const int N = 4096;
    const int M = 4096;
    const int K = 4096;

    const float alpha = 0.5;
    const float beta = 0.7;

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**) &d_a, sizeof(float) * M * K));
    checkCudaErrors(cudaMalloc((void**) &d_b, sizeof(float) * K * N));
    checkCudaErrors(cudaMalloc((void**) &d_c, sizeof(float) * M * N));

    // Allocate memory
    a = (float*)malloc(sizeof(float) * M * K);
    b = (float*)malloc(sizeof(float) * K * N);
    c = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = 1.0;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = 1.0;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 1.0;
        }
    }

    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    // Performance measurement.
    constexpr int num_repeats{10};
    constexpr int num_warmups{10};
    std::function<void(cudaStream_t)> const function_sgemm_naive{std::bind(
        matmul_naive, alpha, beta, N, M, K, a, b, c, d_a, d_b, d_c)};
    float const latency_sgemm_naive{measure_performance(
        function_sgemm_naive, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
                << "sgemm_naive Latency: " << latency_sgemm_naive << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Verification
    printf("Done!\n");
    for(int i = 0; i < M * N; i++){
        assert(fabs(c[i] - alpha * K - beta) < MAX_ERR);
    }
    printf("c[0] = %f\n", c[0]);
    printf("PASSED\n");

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
}

int main() {
    test_naive();
    return 0;
}