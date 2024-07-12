#include <cuda_runtime.h>
#include <iostream>

__global__ void kernelA(float* a, float* b, float* c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void kernelB(float* c, float* d, float* e, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        e[idx] = c[idx] * d[idx];
    }
}

void computeSeparate(float* a, float* b, float* d, float* e, int N) {
    float *d_a, *d_b, *d_c, *d_d, *d_e;
    cudaMalloc((void**)&d_a, N * sizeof(float));
    cudaMalloc((void**)&d_b, N * sizeof(float));
    cudaMalloc((void**)&d_c, N * sizeof(float));
    cudaMalloc((void**)&d_d, N * sizeof(float));
    cudaMalloc((void**)&d_e, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, d, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernelA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    kernelB<<<blocksPerGrid, threadsPerBlock>>>(d_c, d_d, d_e, N);
    cudaDeviceSynchronize();

    cudaMemcpy(e, d_e, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);
    cudaFree(d_e);
}
