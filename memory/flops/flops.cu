#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(result) << std::endl; \
        exit(result); \
    } \
} while (0)

// CUDA kernel to perform a highly intensive computational task
__global__ void computeKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float value = data[idx];
        // Increase compute intensity
        for (int i = 0; i < 10000000; ++i) { // Increase computational workload
            value = sqrt(sqrt(sqrt(sqrt(sqrt(value)))));
            value = value * value * value * value * value;
        }
        data[idx] = value;
    }
}

// CUDA kernel to perform a simple computation for FLOPS measurement
__global__ void computeFLOPS(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float value = data[idx];
        
        // Example: Use FMA for maximum FLOPS
        #pragma unroll
        for (int i = 0; i < 10000000; ++i) { // Adjust to increase computational workload
            // Example of FMA operation
            value = fma(value, 2.0f, 3.0f); // value = value * 2.0f + 3.0f;
        }

        // Store the result back to global memory
        data[idx] = value;
    }
}


void runComputeKernel(size_t N) {
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_data = (float*)malloc(size);

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_data;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    computeFLOPS<<<numBlocks, blockSize>>>(d_data, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Calculate GFLOPS
    // Each kernel invocation performs 1000000 floating-point operations
    double totalOperations = (double)(10000000 * N);
    double executionTimeSeconds = gpuTime / 1000.0; // Convert ms to seconds
    double gflops = (totalOperations / executionTimeSeconds) / 1.0e9; // 1.0e9 for Giga (10^9)

    std::cout << "Elapsed GPU Time: " << gpuTime << " ms" << std::endl;
    std::cout << "Computed GFLOPS: " << gflops << " GFLOPS" << std::endl;

    // Free allocated memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
}

int main() {
    size_t N = 1000000; // Adjust N as needed for the size of the problem
    runComputeKernel(N);
    return 0;
}
