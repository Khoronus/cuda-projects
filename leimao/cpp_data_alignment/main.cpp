#include <cstdio>
#include <cstdlib>
#include <iostream>

int main()
{
    unsigned char buf1[sizeof(int) / sizeof(char)];
    std::cout << "Default "
              << alignof(unsigned char[sizeof(int) / sizeof(char)]) << "-byte"
              << " aligned addr: " << static_cast<void*>(buf1) << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf1) % alignof(int) << std::endl;

    alignas(int) unsigned char buf2[sizeof(int) / sizeof(char)];
    std::cout << alignof(int)
              << "-byte aligned addr: " << static_cast<void*>(buf2)
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) %
                     alignof(unsigned char[sizeof(int) / sizeof(char)])
              << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(buf2) % alignof(int) << std::endl;

    void* p1 = malloc(sizeof(int));
    std::cout << "Default "
              << "16-byte"
              << " aligned addr: " << p1 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p1) % 1024 << std::endl;
    free(p1);

    void* p2 = aligned_alloc(1024, sizeof(int));
    std::cout << "1024-byte aligned addr: " << p2 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 16 << std::endl;
    std::cout << reinterpret_cast<uintptr_t>(p2) % 1024 << std::endl;
    free(p2);
}
