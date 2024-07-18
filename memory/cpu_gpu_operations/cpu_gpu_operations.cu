#include <iostream>
#include <cstdlib>
#include <chrono>
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
    const int N = 100000000; // Increased number of elements in vectors
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
    auto startHtoD = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_a + mid, h_a + mid, (N - mid) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b + mid, h_b + mid, (N - mid) * sizeof(float), cudaMemcpyHostToDevice));
    auto endHtoD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hToDTime = endHtoD - startHtoD;

    // Launch GPU kernel for GPU part
    dim3 blockSizeGPU(blockSize);
    dim3 numBlocksGPU((N - mid + blockSize - 1) / blockSize);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    gpuVectorAdd<<<numBlocksGPU, blockSizeGPU>>>(d_a, d_b, d_c, mid, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy result from device to host for GPU part
    auto startDtoH = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(h_c + mid, d_c + mid, (N - mid) * sizeof(float), cudaMemcpyDeviceToHost));
    auto endDtoH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dToHTime = endDtoH - startDtoH;

    // Perform CPU vector addition for CPU part
    auto startCPU = std::chrono::high_resolution_clock::now();
    cpuVectorAdd(h_a, h_b, h_c, 0, mid);
    auto endCPU = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuTime = endCPU - startCPU;

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

    // GFLOPS calculation
    double gflops = (2.0 * (N - mid)) / (gpuTime / 1000.0) / 1e9;
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;
    std::cout << "GFLOPS: " << gflops << std::endl;

    // Bandwidth calculation
    double hToDBandwidth = (2.0 * (N - mid) * sizeof(float)) / (hToDTime.count() * 1e9);
    double dToHBandwidth = ((N - mid) * sizeof(float)) / (dToHTime.count() * 1e9);
    std::cout << "Host to Device Bandwidth: " << hToDBandwidth << " GB/s" << std::endl;
    std::cout << "Device to Host Bandwidth: " << dToHBandwidth << " GB/s" << std::endl;

    // CPU Time
    std::cout << "CPU Time: " << cpuTime.count() * 1000.0 << " ms" << std::endl;

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
        data[idx] += 10.0f;
    }
}

void example_async() {
    const int N = 100000; // Increased number of elements
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
    auto startHtoD = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice));
    auto endHtoD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hToDTime = endHtoD - startHtoD;

    // Use cudaMemcpy2DToArray for host to CUDA array transfer
    auto startHtoA = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_data, N * sizeof(float), N * sizeof(float), 1, cudaMemcpyHostToDevice));
    auto endHtoA = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hToATime = endHtoA - startHtoA;

    // Perform dummy operation on device data using a CUDA kernel
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    dummyKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy data back from device to host using cudaMemcpyAsync
    auto startDtoH = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost));
    auto endDtoH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dToHTime = endDtoH - startDtoH;

    // Synchronize again to ensure all asynchronous operations are completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    std::cout << "Results after copying from device to host and dummy operation:" << std::endl;
    for (int i = 0; i < N; ++i) {
        if (i < 10) { // Print only the first 10 results to avoid excessive output
            std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
        }
    }

    // Bandwidth calculation
    double hToDBandwidth = size / (hToDTime.count() * 1e9);
    double hToABandwidth = size / (hToATime.count() * 1e9);
    double dToHBandwidth = size / (dToHTime.count() * 1e9);
    std::cout << "Host to Device Bandwidth: " << hToDBandwidth << " GB/s" << std::endl;
    std::cout << "Host to Array Bandwidth: " << hToABandwidth << " GB/s" << std::endl;
    std::cout << "Device to Host Bandwidth: " << dToHBandwidth << " GB/s" << std::endl;

    // GPU Time
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;

    // Free allocated memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeArray(cuArray));
}

/**
CUDA error at /cpu_gpu_operations.cu:164: invalid argument
The error occurs because the array dimensions exceed the limits supported by the CUDA runtime. 
Specifically, the dimensions for cudaMallocArray and cudaMemcpy2DToArray must be within the maximum 
supported width and height for 2D arrays on the GPU.
 */
void example_async_1D() {
    const int N = 10000000; // Number of elements
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

    // Use cudaMemcpyAsync for host to device transfer
    auto startHtoD = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice));
    auto endHtoD = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hToDTime = endHtoD - startHtoD;

    // Perform dummy operation on device data using a CUDA kernel
    dim3 blockSize(256);
    dim3 numBlocks((N + blockSize.x - 1) / blockSize.x);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    dummyKernel<<<numBlocks, blockSize>>>(d_data, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpuTime;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTime, start, stop));

    // Copy data back from device to host using cudaMemcpyAsync
    auto startDtoH = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost));
    auto endDtoH = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dToHTime = endDtoH - startDtoH;

    // Synchronize again to ensure all asynchronous operations are completed
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify results
    std::cout << "Results after copying from device to host and dummy operation:" << std::endl;
    for (int i = 0; i < N; ++i) {
        if (i < 10) { // Print only the first 10 results to avoid excessive output
            std::cout << "h_data[" << i << "] = " << h_data[i] << std::endl;
        }
    }

    // Bandwidth calculation
    double hToDBandwidth = size / (hToDTime.count() * 1e9);
    double dToHBandwidth = size / (dToHTime.count() * 1e9);
    std::cout << "Host to Device Bandwidth: " << hToDBandwidth << " GB/s" << std::endl;
    std::cout << "Device to Host Bandwidth: " << dToHBandwidth << " GB/s" << std::endl;

    // GPU Time
    std::cout << "GPU Time: " << gpuTime << " ms" << std::endl;

    // Free allocated memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
}

int main() {
    example_addition_cpu_gpu();
    example_async();
    example_async_1D();
    return 0;
}
