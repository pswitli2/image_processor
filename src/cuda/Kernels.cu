#include "Kernels.hpp"

__global__ void __sum(const pixel64_t* input, pixel64_t* output, std::size_t length)
{
    const auto idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
    const auto offset = idx * length;
    pixel64_t sum = 0;
    for (std::size_t i = offset; i < offset + length; i++)
    {
        const auto p = input[i];
        sum += p;
    }

    output[idx] = sum;
}

__global__ void __copy_image(const pixel64_t* input, pixel64_t* output)
{
    const auto idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
    output[idx] = input[idx];
}

__global__ void __pixel_minus_mean_pow2(const pixel64_t* input, pixel64_t* output, pixel64_t mean)
{
    const auto idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    const auto p = input[idx];
    const auto p_minus_mean = (long long) p - (long long) mean;
    output[idx] = p_minus_mean * p_minus_mean;
}

__global__ void __clear_image(pixel64_t* inout)
{
    const auto idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    const pixel64_t zero = 0;
    inout[idx] = zero;
}

__global__ void __threshold(const pixel64_t* input, pixel64_t* output, pixel64_t threshold)
{
    const auto idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    const auto t = threshold;
    const auto in = input[idx];

    if (in >= t)
        output[idx] = (pixel64_t) D_MAX_PIXEL_VAL;
}
