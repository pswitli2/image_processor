#include "Kernels.hpp"

__device__ std::size_t __get_idx()
{
    return (blockIdx.x * blockDim.x) + threadIdx.x;
}
__global__ void __copy_image(const pixel64_t* input, pixel64_t* output)
{
    const auto idx = __get_idx();
    output[idx] = input[idx];
}

__global__ void __sum(const pixel64_t* input, pixel64_t* output, std::size_t length)
{
    const auto idx = __get_idx();
    const auto offset = idx * length;
    pixel64_t sum = 0;
    for (std::size_t i = offset; i < offset + length; i++)
    {
        const auto p = input[i];
        sum += p;
    }

    output[idx] = sum;
}

__global__ void __pixel_minus_mean_pow2(const pixel64_t* input, pixel64_t* output, pixel64_t mean)
{
    const auto idx = __get_idx();

    const auto p = input[idx];
    const auto p_minus_mean = (long long) p - (long long) mean;
    output[idx] = p_minus_mean * p_minus_mean;
}

__global__ void __clear_image(pixel64_t* inout)
{
    const auto idx = __get_idx();

    const pixel64_t zero = 0;
    inout[idx] = zero;
}

__global__ void __threshold(const pixel64_t* input, pixel64_t* output, pixel64_t threshold)
{
    const auto idx = __get_idx();

    const auto t = threshold;
    const auto in = input[idx];

    if (in >= t)
        output[idx] = (pixel64_t) D_MAX_PIXEL_VAL;
}

__global__ void __sum_history(const pixel64_t* input, pixel64_t* output, std::size_t history_size, std::size_t area)
{
    const auto pixel_idx = __get_idx();

    pixel64_t sum = 0;
    for (std::size_t i = 0; i < history_size; i++)
    {
        const auto history_idx = i * area + pixel_idx;
        sum += input[history_idx];
    }
    output[pixel_idx] = sum;
}

__global__ void __remove_background(const pixel64_t* input, pixel64_t* output, pixel64_t div_val, pixel64_t tolerance)
{
    const auto idx = __get_idx();

    const int64_t mean = (int64_t) (input[idx] / div_val);
    const int64_t p = (int64_t) output[idx];
    if ((pixel64_t) std::abs(mean - p) < tolerance)
        output[idx] = 0;

}

__global__ void __lone_pixel(const pixel64_t* input, pixel64_t* output, std::size_t num_adjacent, std::size_t width)
{
    const auto idx = __get_idx();

    std::size_t idxs[8];
    idxs[0] = idx - 1;
    idxs[1] = idx + 1;
    idxs[2] = idx - width;
    idxs[3] = idx + width;
    idxs[4] = idxs[2] - 1;
    idxs[5] = idxs[2] + 1;
    idxs[6] = idxs[3] - 1;
    idxs[7] = idxs[3] + 1;
    pixel64_t sum = 0;
    for (std::size_t i = 0; i < 8; i++)
        sum += input[idxs[i]];

    if (sum < D_MAX_PIXEL_VAL * num_adjacent)
        output[idx] = 0;
}
