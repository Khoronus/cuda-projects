#include <cuda_runtime.h>
#include <iostream>

__global__ void fusedKernel(float* a, float* b, float* d, float* e, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float temp = a[idx] + b[idx];
        e[idx] = temp * d[idx];
    }
}

void computeFused(float* a, float* b, float* d, float* e, int N) {
    float *d_a, *d_b, *d_d, *d_e;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_d, N * sizeof(float));
    cudaMalloc((void**)&d_e, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fusedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_d, d_e, N);
    cudaDeviceSynchronize();

    cudaMemcpy(e, d_e, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_d);
    cudaFree(d_e);
}
