#include <iostream>
#include <memory>
#include <vector>

template <typename T>
void bank_id_1d_mapping(int bank_size, int num_banks, int N)
{
    for (int i{0}; i < N; ++i)
    {
        int bank_idx = (i * sizeof(T) * 8 / bank_size) % num_banks;
        std::cout << "Array Idx: " << i << " "
                  << "Bank Idx: " << bank_idx << std::endl;
    }
}

template <typename T>
void bank_id_2d_mapping(int bank_size, int num_banks, int M, int N)
{
    for (int i{0}; i < M; ++i)
    {
        for (int j{0}; j < N; ++j)
        {
            int bank_idx =
                ((i * N + j) * sizeof(T) * 8 / bank_size) % num_banks;
            std::cout << "Matrix Idx: (" << i << ", " << j << ") "
                      << "Bank Idx: " << bank_idx << std::endl;
        }
    }
}

int main()
{

    constexpr const int bank_size{32}; // bits
    constexpr const int num_banks{32};

    const int M{4};
    const int N{33};

    std::cout << "Bank ID Mapping 1D: N = " << N << std::endl;
    bank_id_1d_mapping<float>(bank_size, num_banks, N);
    std::cout << "Bank 2D Mapping 1D: M = " << M << " N = " << N << std::endl;
    bank_id_2d_mapping<float>(bank_size, num_banks, M, N);
}
