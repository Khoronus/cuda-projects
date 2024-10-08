
#include "wmma_impl.h"

int main() {
    // Matrix size
    int M = 16;
    int N = 16;
    int K = 16;

    std::cout << "Running WMMA GEMM with float precision:" << std::endl;
    run_wmma_gemm<float>(M, N, K);

    std::cout << "Running WMMA GEMM with half precision:" << std::endl;
    run_wmma_gemm<__half>(M, N, K);

    std::cout << "GEMM completed successfully." << std::endl;
    return 0;
}
