# cuda-projects
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Collection of cuda projects.

## How to build

Most of the projects have to be compiled locally. A common cmake is provided to test the projects (*partially completed*).

```
mkdir build
cd build
cmake -DBUILD_TESTS=ON ..
make
ctest
```

## References

[CUDA Programming](https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming))

