#include "ThresholdingKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "CudaUtils.hpp"

// __device__ static pixel_t MAX = std::numeric_limits<pixel_t>::max() - 1;

template<typename T>
__global__ void sum(const T* input, size_t* output, std::size_t length)
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

__global__ void pixel_minus_mean_pow2(const pixel_t* input, pixel_t* output, pixel_t mean)
{
    const std::size_t idx = ((blockIdx.x * blockDim.x) + threadIdx.x);

    const auto p = input[idx];
    const auto p_minus_mean = p - mean;
    output[idx] = p_minus_mean * p_minus_mean;
}

void sum_image(const pixel_t* input, std::size_t* d_col, std::size_t* sum)
{

}

void ThresholdingKernelWrapper::execute_impl()
{
    // allocate device column
    std::size_t* d_col;
    CUDA_MALLOC((void**) &d_col, height() * sizeof(std::size_t));

    // sum rows, output sums in d_col
    sum<<<height(), 1>>>(d_input, d_col, width());

    // sum pixels in d_col, output in d_col[0]
    sum<<<1, 1>>>(d_col, d_col, height());
   
    // copy sum in d_col[0] to sum
    std::size_t sum[1];
    cudaMemcpy(sum, d_col, sizeof(size_t), cudaMemcpyDeviceToHost);

    // calculate mean
    const pixel_t mean = (pixel_t) (sum[0] / area());

    std::cout << "MEAN CUDA:   " << sum[0] << "  " << mean << std::endl;

    // std::size_t* d_size;
    // CUDA_MALLOC((void**) &d_size, sizeof(std::size_t));




    pixel_minus_mean_pow2<<<height(), width()>>>(d_input, d_output, mean);


    // pixel_t stddev[1];
    // cudaMemcpy(stddev, d_output, sizeof(pixel_t), cudaMemcpyDeviceToHost);
    // std::cout << "STDDEV CUDA: " << stddev[0] << std::endl;

    CUDA_FREE(&d_col);
}