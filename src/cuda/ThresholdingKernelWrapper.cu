#include "ThresholdingKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "CudaUtils.hpp"

// __device__ static pixel_t MAX = std::numeric_limits<pixel_t>::max() - 1;

template<typename T>
__global__ void __sum(const T* input, size_t* output, std::size_t length)
{
    const std::size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);
    const std::size_t offset = idx * length;
    std::size_t sum = 0;
    for (std::size_t i = offset; i < offset + length; i++)
    {
        const T p = input[i];
        sum += p;
    }

    output[idx] = sum;
}

__global__ void __pixel_minus_mean_pow2(const pixel_t* input, size_t* output, pixel_t mean)
{
    const std::size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    const size_t p = input[idx];
    const size_t p_minus_mean = p - mean;
    output[idx] = p_minus_mean * p_minus_mean;
}

template<typename T>
void ThresholdingKernelWrapper::sum_image(const T* d_input, std::size_t* d_col, std::size_t* sum)
{
    // sum rows, output sums in d_col
    __sum<<<height(), 1>>>(d_input, d_col, width());

    // sum pixels in d_col, output in d_col[0]
    __sum<<<1, 1>>>(d_col, d_col, height());
   
    // copy sum in d_col[0] to sum
    cudaMemcpy(sum, d_col, sizeof(size_t), cudaMemcpyDeviceToHost);}

void ThresholdingKernelWrapper::execute_impl()
{
    // allocate device column
    std::size_t* d_col;
    CUDA_MALLOC((void**) &d_col, height() * sizeof(std::size_t));
    std::size_t sum[1];

    sum_image(m_d_input, d_col, sum);
    // sum rows, output sums in d_col
    // sum<<<height(), 1>>>(m_d_input, d_col, width());

    // sum pixels in d_col, output in d_col[0]
    // sum<<<1, 1>>>(d_col, d_col, height());
   
    // copy sum in d_col[0] to sum
    // cudaMemcpy(sum, d_col, sizeof(size_t), cudaMemcpyDeviceToHost);

    // calculate mean
    const pixel_t mean = (pixel_t) (sum[0] / area());

    // std::cout << "MEAN CUDA:   " << sum[0] << "  " << mean << std::endl;

    // std::size_t* d_size;
    // CUDA_MALLOC((void**) &d_size, sizeof(std::size_t));


    std::size_t* d_output_sizet;
    CUDA_MALLOC((void**) &d_output_sizet, num_bytes());

    __pixel_minus_mean_pow2<<<height(), width()>>>(m_d_input, d_output_sizet, mean);

    sum_image(d_output_sizet, d_col, sum);

    const pixel_t stddev = sqrt((pixel_t) (sum[0] / area()));

    std::cout << "STDDEV CUDA: " << sum[0] << "  " << stddev << std::endl;



    // pixel_t stddev[1];
    // cudaMemcpy(stddev, d_output, sizeof(pixel_t), cudaMemcpyDeviceToHost);
    // std::cout << "STDDEV CUDA: " << stddev[0] << std::endl;

    CUDA_FREE(&d_col);
    CUDA_FREE(&d_output_sizet);
}