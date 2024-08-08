#include <gtest/gtest.h>
#include "kernel_impl.h"

// Test case
TEST(TestMain, Kernels) {
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
    for (int i = 0; i < N; ++i) {
        ASSERT_FLOAT_EQ(e[i], e_fused[i]) << "Mismatch at index " << i;
    }

    delete[] a;
    delete[] b;
    delete[] d;
    delete[] e;
    delete[] e_fused;

    std::cout << "TestMain completed successfully." << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
