#include <gtest/gtest.h>
#include "wmma_impl.h"

// Test case
TEST(AddKernelTest, SimpleRun) {
    // Matrix size
    int M = 16;
    int N = 16;
    int K = 16;

    std::cout << "Running WMMA GEMM with float precision:" << std::endl;
    EXPECT_NO_THROW(run_wmma_gemm<float>(M, N, K));

    std::cout << "Running WMMA GEMM with half precision:" << std::endl;
    EXPECT_NO_THROW(run_wmma_gemm<__half>(M, N, K));

    std::cout << "GEMM completed successfully." << std::endl;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}