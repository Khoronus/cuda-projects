#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t result = call; \
    if (result != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(result) << std::endl; \
        exit(result); \
    } \
} while (0)

// CPU function to perform vector addition
void cpuVectorAdd(float *a, float *b, float *c, int start, int end) {
    for (int i = start; i < end; ++i) {
        c[i] = a[i] + b[i];
    }
}

// CUDA kernel to perform vector addition (GPU function)
__global__ void gpuVectorAdd(float *a, float *b, float *c, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + start < end) {
        c[idx + start] = a[idx + start] + b[idx + start];
    }
}


void example_addition_cpu_gpu(){
    const int N = 100000; // Number of elements in vectors
    const int blockSize = 256; // Threads per block

    // Compute division for CPU and GPU work
    int mid = N / 2; // Mid-point to divide work between CPU and GPU

    // Allocate memory for host vectors (CPU)
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(N * sizeof(float));
    h_b = (float*)malloc(N * sizeof(float));
    h_c = (float*)malloc(N * sizeof(float));

    // Initialize host input vectors
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate memory for device vectors (GPU)
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // Copy host input data to device for GPU part
    CUDA_CHECK(cudaMemcpy(d_a + mid, h_a + mid, (N - mid) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b + mid, h_b + mid, (N - mid) * sizeof(float), cudaMemcpyHostToDevice));

    // Launch GPU kernel for GPU part
    dim3 blockSizeGPU(blockSize);
    dim3 numBlocksGPU((N - mid + blockSize - 1) / blockSize);
    gpuVectorAdd<<<numBlocksGPU, blockSizeGPU>>>(d_a, d_b, d_c, mid, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result from device to host for GPU part
    CUDA_CHECK(cudaMemcpy(h_c + mid, d_c + mid, (N - mid) * sizeof(float), cudaMemcpyDeviceToHost));

    // Perform CPU vector addition for CPU part
    cpuVectorAdd(h_a, h_b, h_c, 0, mid);

    // Verify results
    bool success = true;
    int mismatchIndex = -1;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            success = false;
            mismatchIndex = i;
            std::cout << "Mismatch at index " << mismatchIndex << ": CPU=" << h_a[i] + h_b[i] << ", GPU=" << h_c[i] << std::endl;
            break;
        }
    }

    if (!success && mismatchIndex != -1) {
        std::cout << "CPU[" << mismatchIndex << "] = " << h_a[mismatchIndex] << ", " << h_b[mismatchIndex] << ", " << h_c[mismatchIndex] << std::endl;
        
        float gpu_a, gpu_b, gpu_c;
        CUDA_CHECK(cudaMemcpy(&gpu_a, d_a + mismatchIndex, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gpu_b, d_b + mismatchIndex, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&gpu_c, d_c + mismatchIndex, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << "GPU[" << mismatchIndex << "] = " << gpu_a << ", " << gpu_b << ", " << gpu_c << std::endl;
    }

    if (success) {
        std::cout << "CPU and GPU results match." << std::endl;
    } else {
        std::cout << "CPU and GPU results do not match." << std::endl;
    }

    // Free host and device memory
    free(h_a); free(h_b); free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}


// CUDA kernel to perform a dummy operation (addition) on device data
__global__ void dummyKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 10.0f; // Perform a dummy operation (addition by 10.0f)
    }
}

void example_async(){
    const int N = 10; // Number of elements
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

    // Allocate CUDA array
    cudaArray *cuArray;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, N, 1));

    // Use cudaMemcpyAsync for host to device transfer
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Use cudaMemcpy2DToArray for host to CUDA array transfer
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, N * sizeof(float), N * sizeof(float), 1, cudaMemcpyHostToDevice);

    // Perform dummy operation on device data using a CUDA kernel
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    dummyKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure all memory operations and kernel executions are completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy data back from device to host using cudaMemcpyAsync
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Synchronize again to ensure all asynchronous operations are completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    std::cout << "Results after copying from device to host and dummy operation:" << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
    }

    // Free allocated memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeArray(cuArray));
}


int main() {
    example_addition_cpu_gpu();
    example_async();
    return 0;
}
