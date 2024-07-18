#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t result = call; \
        if (result != cudaSuccess) { \
            cerr << "CUDA error: " << cudaGetErrorString(result) << " at line " << __LINE__ << endl; \
            exit(result); \
        } \
    } while(0)

// Kernel for matrix multiplication
__global__ void matrixMultiply(float *a, float *b, float *c, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[row * K + k] * b[k * M + col];
        }
        c[row * M + col] = sum;
    }
}

// Function to perform matrix multiplication on a GPU
void matrixMultiplyOnGPU(float *h_a, float *h_b, float *h_c, int N, int M, int K, int device) {
    // Set CUDA device
    CUDA_CHECK(cudaSetDevice(device));

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, K * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * M * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, K * M * sizeof(float), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matrixMultiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, N, M, K);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * M * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

// Main function
int main() {
    // Matrix dimensions
    int N = 1024;
    int M = 1024;
    int K = 1024;

    // Number of GPUs
    int numGPUs;
    CUDA_CHECK(cudaGetDeviceCount(&numGPUs));

    cout << "Number of GPUs detected: " << numGPUs << endl;

    // Generate random matrices
    float *h_a = new float[N * K];
    float *h_b = new float[K * M];
    float *h_c = new float[N * M];

    for (int i = 0; i < N * K; ++i) {
        h_a[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    for (int i = 0; i < K * M; ++i) {
        h_b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Calculate workload per GPU
    int workloadPerGPU = N / numGPUs;

    // Perform matrix multiplication on each GPU
    vector<cudaStream_t> streams(numGPUs);
    for (int i = 0; i < numGPUs; ++i) {
        // Initialize CUDA stream for each GPU
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams[i]));

        // Calculate start and end indices for the current GPU
        int startIdx = i * workloadPerGPU;
        int endIdx = (i == numGPUs - 1) ? N : startIdx + workloadPerGPU;

        // Launch matrix multiplication on current GPU
        matrixMultiplyOnGPU(&h_a[startIdx * K], h_b, &h_c[startIdx * M], endIdx - startIdx, M, K, i);
    }

    // Synchronize streams and clean up
    for (int i = 0; i < numGPUs; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    // Print results or further processing as needed
    cout << "Matrix multiplication completed successfully across " << numGPUs << " GPUs." << endl;

    // Clean up host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
