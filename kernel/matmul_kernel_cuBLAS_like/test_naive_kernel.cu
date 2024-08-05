#include <gtest/gtest.h>
#include "naive_kernel_impl.h"

// Test case
TEST(AddNaiveKernelTest, Naive) {
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    const int N = 4096;
    const int M = 4096;
    const int K = 4096;

    const float alpha = 0.5;
    const float beta = 0.7;

    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc((void**) &d_a, sizeof(float) * M * K));
    checkCudaErrors(cudaMalloc((void**) &d_b, sizeof(float) * K * N));
    checkCudaErrors(cudaMalloc((void**) &d_c, sizeof(float) * M * N));

    // Allocate memory
    a = (float*)malloc(sizeof(float) * M * K);
    b = (float*)malloc(sizeof(float) * K * N);
    c = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = 1.0;
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = 1.0;
        }
    }
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i * N + j] = 1.0;
        }
    }

    // launch the asynchronous execution of the kernel on the device
    // The function call returns immediately on the host
    // Performance measurement.
    constexpr int num_repeats{10};
    constexpr int num_warmups{10};
    std::function<void(cudaStream_t)> const function_sgemm_naive{std::bind(
        matmul_naive, alpha, beta, N, M, K, a, b, c, d_a, d_b, d_c)};
    float const latency_sgemm_naive{measure_performance(
        function_sgemm_naive, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3)
                << "sgemm_naive Latency: " << latency_sgemm_naive << " ms" << std::endl;
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Verification
    printf("Done!\n");
    for(int i = 0; i < M * N; i++){
        assert(fabs(c[i] - alpha * K - beta) < MAX_ERR);
    }
    printf("c[0] = %f\n", c[0]);
    printf("PASSED\n");

    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    std::cout << "NaiveKernel completed successfully." << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
