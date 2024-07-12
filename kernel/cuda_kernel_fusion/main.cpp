#include <iostream>
#include <cuda_runtime.h>

void computeSeparate(float* a, float* b, float* d, float* e, int N);
void computeFused(float* a, float* b, float* d, float* e, int N);

int main() {
    const int N = 1000;
    float *a = new float[N];
    float *b = new float[N];
    float *d = new float[N];
    float *e = new float[N];
    float *e_fused = new float[N];

    // Initialize arrays
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
        d[i] = static_cast<float>(i * 3);
    }

    // Compute using separate kernels
    computeSeparate(a, b, d, e, N);

    // Compute using fused kernel
    computeFused(a, b, d, e_fused, N);

    // Compare results
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (e[i] != e_fused[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    delete[] a;
    delete[] b;
    delete[] d;
    delete[] e;
    delete[] e_fused;

    return 0;
}
