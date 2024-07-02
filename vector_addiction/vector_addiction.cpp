#include <math.h>
#include <time.h>
#include <iostream>
#include <stdexcept>
#include "cuda_runtime.h"
#include "kernel.h"

// declare the vectors' number of elements and their size in bytes
static const int n_el = 512;
static const size_t size = n_el * sizeof(float);

int main(){
  // declare and allocate input vectors h_A and h_B in the host (CPU) memory
  float* h_A = (float*)malloc(size);
  float* h_B = (float*)malloc(size);
  float* h_C = (float*)malloc(size);

  // declare device vectors in the device (GPU) memory
  float *d_A,*d_B,*d_C;

  // initialize input vectors
  for (int i=0; i<n_el; i++){
    h_A[i]=sin(i);
    h_B[i]=cos(i);
  }

  // allocate device vectors in the device (GPU) memory
  cudaMalloc(&d_A, size);
  cudaMalloc(&d_B, size);
  cudaMalloc(&d_C, size);

  // copy input vectors from the host (CPU) memory to the device (GPU) memory
  cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

  // call kernel function
  sum(d_A, d_B, d_C, n_el);

  // copy the output (results) vector from the device (GPU) memory to the host (CPU) memory
  cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  // free device memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // compute the cumulative error
  double err=0;
  for (int i=0; i<n_el; i++) {
    double diff=double((h_A[i]+h_B[i])-h_C[i]);
    err+=diff*diff;
    // print results for manual checking.
    std::cout << "A+B: " << h_A[i]+h_B[i] << "\n";
    std::cout << "C: " << h_C[i] << "\n";
  }
  err=sqrt(err);
  std::cout << "err: " << err << "\n";

  // free host memory
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;

  return cudaDeviceSynchronize();
}