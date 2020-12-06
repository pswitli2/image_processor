#include "ThresholderKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "CudaUtils.hpp"

__device__ static pixel16_t MAX = std::numeric_limits<pixel16_t>::max() - 1;

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
        output[idx] = (pixel64_t) MAX;
}

void ThresholderKernelWrapper::sum_image(const pixel64_t* d_input, pixel64_t* d_col, pixel64_t* sum)
{
    // sum rows, output sums in d_col
    __sum<<<height(), 1, 1, m_stream>>>(d_input, d_col, width());

    // sum pixels in d_col, output in d_col[0]
    __sum<<<1, 1, 1, m_stream>>>(d_col, d_col, height());
   
    // copy sum in d_col[0] to sum
    cudaMemcpy(sum, d_col, sizeof(pixel64_t), cudaMemcpyDeviceToHost);}

void ThresholderKernelWrapper::execute_impl()
{
    // allocate device column
    pixel64_t* d_col;
    CUDA_MALLOC((void**) &d_col, height() * sizeof(pixel64_t));

    // length 1 array to copy sum into
    pixel64_t sum[1];

    // sum input image pixels
    sum_image(m_d_input, d_col, sum);

    // calculate mean
    const auto mean = sum[0] / (pixel64_t) area();

    // set output to - for p in pixels: (p - mean) * (p - mean)
    __pixel_minus_mean_pow2<<<height(), width(), 1, m_stream>>>(m_d_input, m_d_output, mean);

    // sum output for standard deviation
    sum_image(m_d_output, d_col, sum);

    // calculate standard deviation
    const pixel64_t stddev = sqrt(sum[0] / (pixel64_t) area());

    // calculate threshold
    const auto threshold = mean + (pixel64_t) ((double) stddev * m_tolerance);

    // zero out output image
    __clear_image<<<height(), width(), 1, m_stream>>>(m_d_output);

    __threshold<<<height(), width(), 1, m_stream>>>(m_d_input, m_d_output, threshold);

    CUDA_FREE(&d_col);
}