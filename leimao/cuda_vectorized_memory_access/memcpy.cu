#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(const char* const file, const int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

std::string std_string_centered(std::string const& s, size_t width,
                                char pad = ' ')
{
    size_t const l{s.length()};
    // Throw an exception if width is too small.
    if (width < l)
    {
        throw std::runtime_error("Width is too small.");
    }
    size_t const left_pad{(width - l) / 2};
    size_t const right_pad{width - l - left_pad};
    std::string const s_centered{std::string(left_pad, pad) + s +
                                 std::string(right_pad, pad)};
    return s_centered;
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> const& bound_function,
                          cudaStream_t stream, unsigned int num_repeats = 100,
                          unsigned int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (unsigned int i{0U}; i < num_warmups; ++i)
    {
        bound_function(stream);
    }

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));
    for (unsigned int i{0U}; i < num_repeats; ++i)
    {
        bound_function(stream);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

template <typename T>
__global__ void custom_device_memcpy(T* __restrict__ output,
                                     T const* __restrict__ input, size_t n)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        output[i] = input[i];
    }
}

template <typename T>
void launch_custom_device_memcpy(T* output, T const* input, size_t n,
                                 cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T, unsigned int BLOCK_DIM_X>
__global__ void custom_device_memcpy_shared_memory(T* __restrict__ output,
                                                   T const* __restrict__ input,
                                                   size_t n)
{
    // Using shared memory as intermediate buffer.
    __shared__ T shared_memory[BLOCK_DIM_X];
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i < n; i += stride)
    {
        shared_memory[threadIdx.x] = input[i];
        // Synchronization is not necessary in this case.
        // __syncthreads();
        output[i] = shared_memory[threadIdx.x];
    }
}

template <typename T>
void launch_custom_device_memcpy_shared_memory(T* output, T const* input,
                                               size_t n, cudaStream_t stream)
{
    constexpr dim3 threads_per_block{1024};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (n + threads_per_block.x - 1U) / threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_shared_memory<T, threads_per_block.x>
        <<<blocks_per_grid, threads_per_block, 0, stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

// One thread copies sizeof(R) bytes of data.
// One warp copies 32 x sizeof(R) bytes of data via one of few memory
// transactions.
template <typename T, typename R = uint64_t>
__global__ void custom_device_memcpy_optimized(T* __restrict__ output,
                                               T const* __restrict__ input,
                                               size_t n)
{
    size_t const idx{blockDim.x * blockIdx.x + threadIdx.x};
    size_t const stride{blockDim.x * gridDim.x};
    for (size_t i{idx}; i * sizeof(R) / sizeof(T) < n; i += stride)
    {
        if ((i + 1U) * sizeof(R) / sizeof(T) < n)
        {
            reinterpret_cast<R*>(output)[i] =
                reinterpret_cast<R const*>(input)[i];
        }
        else
        {
            // Remaining units to copy.
            size_t const start_index{i * sizeof(R) / sizeof(T)};
            size_t const remaining_units_to_copy{(n - start_index)};
            for (size_t j{0}; j < remaining_units_to_copy; ++j)
            {
                output[start_index + j] = input[start_index + j];
            }
        }
    }
}

template <typename T, typename R = uint64_t>
void launch_custom_device_memcpy_optimized(T* output, T const* input, size_t n,
                                           cudaStream_t stream)
{
    dim3 const threads_per_block{1024};
    size_t const num_units_to_copy_round_up{(n * sizeof(T) + sizeof(R) - 1U) /
                                            sizeof(R)};
    dim3 const blocks_per_grid{static_cast<unsigned int>(std::min(
        (num_units_to_copy_round_up + threads_per_block.x - 1U) /
            threads_per_block.x,
        static_cast<size_t>(std::numeric_limits<unsigned int>::max())))};
    custom_device_memcpy_optimized<<<blocks_per_grid, threads_per_block, 0,
                                     stream>>>(output, input, n);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_official_device_memcpy(T* output, T const* input, size_t n,
                                   cudaStream_t stream)
{
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output, input, n * sizeof(T),
                                     cudaMemcpyDeviceToDevice, stream));
}

// Initialize the buffer so that the unit of the data is the index of the data.
template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void initialize_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        buffer[i] = static_cast<T>(
            i % static_cast<size_t>(std::numeric_limits<T>::max()));
    }
}

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
void verify_buffer(T* buffer, size_t n)
{
    for (size_t i{0}; i < n; ++i)
    {
        if (buffer[i] != static_cast<T>(i % static_cast<size_t>(
                                                std::numeric_limits<T>::max())))
        {
            std::cerr << "Verification failed at index: " << i << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// Measure custom device memcpy performance given the number of units to copy,
// the device memcpy function to use, and the number of repeats and warmups.
template <typename T>
float measure_custom_device_memcpy_performance(
    size_t n,
    std::function<void(T*, T const*, size_t, cudaStream_t)> const&
        device_memcpy_function,
    int num_repeats = 100, int num_warmups = 100)
{
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<T> input(n);
    std::vector<T> output(n, static_cast<T>(0));
    initialize_buffer(input.data(), n);

    T* d_input;
    T* d_output;

    CHECK_CUDA_ERROR(cudaMalloc(&d_input, n * sizeof(T)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, n * sizeof(T)));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_input, input.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_output, output.data(), n * sizeof(T),
                                     cudaMemcpyHostToDevice, stream));
    // Run device memcpy once to check correcness.
    device_memcpy_function(d_output, d_input, n, stream);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(output.data(), d_output, n * sizeof(T),
                                     cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify the correctness of the device memcpy.
    verify_buffer(output.data(), n);

    size_t const num_bytes{n * sizeof(T)};
    float const num_giga_bytes{static_cast<float>(num_bytes) / (1 << 30)};

    std::function<void(cudaStream_t)> function{std::bind(
        device_memcpy_function, d_output, d_input, n, std::placeholders::_1)};

    float const latency{
        measure_performance(function, stream, num_repeats, num_warmups)};
    std::cout << std::fixed << std::setprecision(3) << "Latency: " << latency
              << " ms" << std::endl;
    std::cout << "Effective Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) << " GB/s"
              << std::endl;

    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    // Query deive name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    float const peak_bandwidth{
        static_cast<float>(2.0 * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Percentage of Peak Bandwitdh: "
              << 2.f * num_giga_bytes / (latency / 1000) / peak_bandwidth * 100
              << "%" << std::endl;

    return latency;
}

int main()
{
    constexpr unsigned int num_repeats{10U};
    constexpr unsigned int num_warmups{10U};

    constexpr size_t tensor_size_small{1U * 64U * 64U * 64U};
    constexpr size_t tensor_size_medium{1U * 128U * 128U * 128U};
    constexpr size_t tensor_size_large{1U * 512U * 512U * 512U};

    constexpr size_t string_width{50U};

    std::cout << std_string_centered("", string_width, '~') << std::endl;
    std::cout << std_string_centered("NVIDIA GPU Device Info", string_width,
                                     ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '~') << std::endl;

    // Query deive name and peak memory bandwidth.
    int device_id{0};
    cudaGetDevice(&device_id);
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device_id);
    std::cout << "Device Name: " << device_prop.name << std::endl;
    float const memory_size{static_cast<float>(device_prop.totalGlobalMem) /
                            (1 << 30)};
    std::cout << "Memory Size: " << memory_size << " GB" << std::endl;
    float const peak_bandwidth{
        static_cast<float>(2.0f * device_prop.memoryClockRate *
                           (device_prop.memoryBusWidth / 8) / 1.0e6)};
    std::cout << "Peak Bandwitdh: " << peak_bandwidth << " GB/s" << std::endl;
    std::cout << std::endl;

    // Measure CUDA official memcpy performance for different tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("CUDA Official Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_official_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_official_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_official_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_official_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Measure the latency and bandwidth of custom device memcpy for different
    // tensor sizes.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy", string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy<int8_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy<int16_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy<int32_t>, num_repeats,
            num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy<int64_t>, num_repeats,
            num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The more units of data we copy, the higher the bandwidth.
    // 2. The larger the unit of the data, the higher the bandwidth.

    // Check if shared memory can improve the latency of custom device memcpy.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered("Custom Device Memcpy with Shared Memory",
                                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int8_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int16_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_shared_memory<int64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. The effect of using shared memory for improving the latency of custom
    // device memcpy is not obvious.

    // Improve the latency of custom device memcpy when the unit of the data is
    // small.
    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 4-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint32_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint32_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 8-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int8_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int16_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int32_t, uint64_t>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size,
            launch_custom_device_memcpy_optimized<int64_t, uint64_t>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    std::cout << std_string_centered("", string_width, '*') << std::endl;
    std::cout << std_string_centered(
                     "Custom Device Memcpy 16-Byte Copy Per Thread",
                     string_width, ' ')
              << std::endl;
    std::cout << std_string_centered("", string_width, '*') << std::endl;

    for (size_t tensor_size :
         {tensor_size_small, tensor_size_medium, tensor_size_large})
    {
        std::string const tensor_size_string{std::string("Tensor Size: ") +
                                             std::to_string(tensor_size) +
                                             std::string(" Units")};
        std::cout << std_string_centered("", string_width, '=') << std::endl;
        std::cout << std_string_centered(tensor_size_string, string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '=') << std::endl;

        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 1 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int8_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int8_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 2 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int16_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int16_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 4 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int32_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int32_t, uint4>,
            num_repeats, num_warmups);
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        std::cout << std_string_centered("Unit Size: 8 Byte", string_width, ' ')
                  << std::endl;
        std::cout << std_string_centered("", string_width, '-') << std::endl;
        measure_custom_device_memcpy_performance<int64_t>(
            tensor_size, launch_custom_device_memcpy_optimized<int64_t, uint4>,
            num_repeats, num_warmups);
    }
    std::cout << std::endl;

    // Conclusions:
    // 1. Copying data in units of 8 bytes or 16 bytes can improve the latency
    // of custom device memcpy.
}
