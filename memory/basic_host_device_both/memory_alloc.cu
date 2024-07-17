#include <iostream>
#include <iomanip>
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

__global__ void kernel(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__device__ void deviceFunction(float& value) {
    value += 1; // Example device function modifying data
}

__global__ void kernel_mul2(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2; // Example operation on device data
        deviceFunction(data[idx]); // Call __device__ function
    }
}

double seconds_elapsed(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    return static_cast<double>(elapsed_time) / 1000.0; // Convert to seconds
}

double compute_gflops(int N, double kernel_time) {
    // Each kernel invocation processes N elements, with 2 floating point operations per element
    double flops = 2.0 * static_cast<double>(N);
    double gflops = flops / kernel_time * 1.0e-9; // 1.0e-9 converts to GFLOPS
    return gflops;
}

void print_memory_info() {
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "CUDA Memory Info:" << std::endl;
    std::cout << "  Free memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Total memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;
}

void print_bandwidth_info(int N, double kernel_time) {
    size_t total_bytes = N * sizeof(float);
    double bandwidth = total_bytes / kernel_time / (1024 * 1024 * 1024); // GB/s
    std::cout << "CUDA Bandwidth Info:" << std::endl;
    std::cout << "  Effective bandwidth: " << bandwidth << " GB/s" << std::endl;
}


// you must first call the cudaGetDeviceProperties() function, then pass 
// the devProp structure returned to this function:
int getSPcores(cudaDeviceProp devProp)
{  
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major){
     case 2: // Fermi
      if (devProp.minor == 1) cores = mp * 48;
      else cores = mp * 32;
      break;
     case 3: // Kepler
      cores = mp * 192;
      break;
     case 5: // Maxwell
      cores = mp * 128;
      break;
     case 6: // Pascal
      if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
      else if (devProp.minor == 0) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 7: // Volta and Turing
      if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
      else printf("Unknown device type\n");
      break;
     case 8: // Ampere
      if (devProp.minor == 0) cores = mp * 64;
      else if (devProp.minor == 6) cores = mp * 128;
      else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
      else printf("Unknown device type\n");
      break;
     case 9: // Hopper
      if (devProp.minor == 0) cores = mp * 128;
      else printf("Unknown device type\n");
      break;
     default:
      printf("Unknown device type\n"); 
      break;
      }
    return cores;
}

/**
 * Basic allocation in both host and device
 */
void example_basic(){
    std::cout << "example_basic" << std::endl;
    const int N = 1000000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_b, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_c, N * sizeof(float)));

    // Transfer data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transfer results from device to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results (optional)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cerr << "Verification error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "Verification passed!" << std::endl;
    } else {
        std::cout << "Verification failed!" << std::endl;
    }

    // Measure allocated memory
    size_t freeMem, totalMem;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    std::cout << "Free memory: " << freeMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Total memory: " << totalMem / (1024 * 1024) << " MB" << std::endl;

    // Free device and host memory
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}


/**
 * Example of unified memory and use
 */
int example_unified_memory() {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found." << std::endl;
        return 1;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Global memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Total CUDA cores: " << getSPcores(deviceProp) * deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock / 1024 << " KB" << std::endl;
        std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    }

    // Select first GPU
    CUDA_CHECK(cudaSetDevice(0));

    const int N = 1000000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate unified memory (accessible from both host and device)
    float *data;
    CUDA_CHECK(cudaMallocManaged(&data, N * sizeof(float)));

    // Initialize data (on host)
    for (int i = 0; i < N; ++i) {
        data[i] = static_cast<float>(i);
    }

    // Start measuring kernel execution time
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch kernel on device
    kernel_mul2<<<blocksPerGrid, threadsPerBlock>>>(data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Stop measuring time
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    double kernel_time = seconds_elapsed(start, stop);

    // Calculate and print GFLOPS
    double gflops = compute_gflops(N, kernel_time);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CUDA GFLOPS Info:" << std::endl;
    std::cout << "  GFLOPS: " << gflops << std::endl;

    // Print memory info
    print_memory_info();

    // Print bandwidth info
    print_bandwidth_info(N, kernel_time);

    // Verify results (on host)
    for (int i = 0; i < 10; ++i) {
        std::cout << "data[" << i << "] = " << data[i] << std::endl;
    }

    // Free unified memory
    CUDA_CHECK(cudaFree(data));

    return 0;
}

int main() {
    example_basic();
    example_unified_memory();
    return 0;
}
