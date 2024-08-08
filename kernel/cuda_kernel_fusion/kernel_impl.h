#include <iostream>
#include <cuda_runtime.h>

void computeSeparate(float* a, float* b, float* d, float* e, int N);
void computeFused(float* a, float* b, float* d, float* e, int N);
