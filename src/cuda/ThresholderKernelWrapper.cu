#include "ThresholderKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "Kernels.hpp"

ThresholderKernelWrapper::ThresholderKernelWrapper(std::size_t width, std::size_t height, double tolerance)
: KernelWrapper(width, height), m_tolerance(tolerance)
{
    CUDA_MALLOC((void**) &m_d_col, height * sizeof(pixel64_t));
}

ThresholderKernelWrapper::~ThresholderKernelWrapper()
{
    CUDA_FREE(&m_d_col);
}

void ThresholderKernelWrapper::sum_image(const pixel64_t* d_input, pixel64_t* sum)
{
    // sum rows, output sums in m_d_col
    __sum<<<height(), 1, 1, m_stream>>>(d_input, m_d_col, width());

    // sum pixels in m_d_col, output in m_d_col[0]
    __sum<<<1, 1, 1, m_stream>>>(m_d_col, m_d_col, height());
   
    // copy sum in m_d_col[0] to sum
    cudaMemcpy(sum, m_d_col, sizeof(pixel64_t), cudaMemcpyDeviceToHost);}

void ThresholderKernelWrapper::execute_impl()
{
    // length 1 array to copy sum into
    pixel64_t sum[1];

    // sum input image pixels
    sum_image(m_d_input, sum);

    // calculate mean
    const auto mean = sum[0] / (pixel64_t) area();

    // set output to - for p in pixels: (p - mean) * (p - mean)
    __pixel_minus_mean_pow2<<<height(), width(), 1, m_stream>>>(m_d_input, m_d_output, mean);

    // sum output for standard deviation
    sum_image(m_d_output, sum);

    // calculate standard deviation
    const pixel64_t stddev = sqrt(sum[0] / (pixel64_t) area());

    // calculate threshold
    const auto threshold = mean + (pixel64_t) ((double) stddev * m_tolerance);

    // zero out output image
    __clear_image<<<height(), width(), 1, m_stream>>>(m_d_output);

    __threshold<<<height(), width(), 1, m_stream>>>(m_d_input, m_d_output, threshold);
}