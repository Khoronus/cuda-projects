#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(result) << std::endl; \
        exit(result); \
    } \
} while (0)

// CUDA kernel to perform a simple computation for FLOPS measurement
__global__ void computeFLOPS(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float value = data[idx];
        
        // Example: Use FMA for maximum FLOPS
        #pragma unroll
        for (int i = 0; i < 1000000; ++i) { // Adjust to increase computational workload
            // Example of FMA operation
            value = fma(value, 2.0f, 3.0f); // value = value * 2.0f + 3.0f;
        }

        // Store the result back to global memory
        data[idx] = value;
    }
}

void measureFLOPSOnAllGPUs(size_t N) {
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));

    std::vector<float*> h_data(numDevices);
    std::vector<float*> d_data(numDevices);
    std::vector<float*> results(numDevices);
    std::vector<cudaEvent_t> startEvents(numDevices);
    std::vector<cudaEvent_t> stopEvents(numDevices);
    std::vector<float> elapsedTimes(numDevices);

    // Allocate and initialize memory on each GPU
    for (int device = 0; device < numDevices; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        
        // Allocate host memory
        h_data[device] = (float*)malloc(N * sizeof(float));
        
        // Initialize host data
        for (size_t i = 0; i < N; ++i) {
            h_data[device][i] = static_cast<float>(i);
        }

        // Allocate device memory
        CUDA_CHECK(cudaMalloc((void**)&d_data[device], N * sizeof(float)));
        
        // Copy data from host to device
        CUDA_CHECK(cudaMemcpy(d_data[device], h_data[device], N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Create CUDA events for timing
        CUDA_CHECK(cudaEventCreate(&startEvents[device]));
        CUDA_CHECK(cudaEventCreate(&stopEvents[device]));
    }

    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);

    // Launch the kernel on each GPU
    for (int device = 0; device < numDevices; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        
        // Record the start event
        CUDA_CHECK(cudaEventRecord(startEvents[device], 0));
        
        // Launch the kernel
        computeFLOPS<<<numBlocks, blockSize>>>(d_data[device], N);
        CUDA_CHECK(cudaGetLastError());
        
        // Record the stop event
        CUDA_CHECK(cudaEventRecord(stopEvents[device], 0));
    }

    // Synchronize and measure elapsed time
    for (int device = 0; device < numDevices; ++device) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaEventSynchronize(stopEvents[device]));
        
        // Calculate the elapsed time
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTimes[device], startEvents[device], stopEvents[device]));
    }

    // Calculate FLOPS
    for (int device = 0; device < numDevices; ++device) {
        float gflops = (2.0f * N * 1000000.0f) / (elapsedTimes[device] * 1e6f);
        std::cout << "Device " << device << ": " << elapsedTimes[device] << " ms, " << gflops << " GFLOPS" << std::endl;
    }

    // Clean up
    for (int device = 0; device < numDevices; ++device) {
        CUDA_CHECK(cudaFree(d_data[device]));
        free(h_data[device]);
        CUDA_CHECK(cudaEventDestroy(startEvents[device]));
        CUDA_CHECK(cudaEventDestroy(stopEvents[device]));
    }
}

int main() {
    size_t N = 100000000;
    measureFLOPSOnAllGPUs(N);
    return 0;
}
