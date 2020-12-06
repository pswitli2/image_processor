#ifndef KERNELS_CU_
#define KERNELS_CU_

#include "CudaUtils.hpp"

__device__ std::size_t __get_idx();

__global__ void __copy_image(const pixel64_t* input, pixel64_t* output);

__global__ void __sum(const pixel64_t* input, pixel64_t* output, std::size_t length);

__global__ void __pixel_minus_mean_pow2(const pixel64_t* input, pixel64_t* output, pixel64_t mean);

__global__ void __clear_image(pixel64_t* inout);

__global__ void __threshold(const pixel64_t* input, pixel64_t* output, pixel64_t threshold);

__global__ void __sum_history(const pixel64_t* input, pixel64_t* output, std::size_t history_size, std::size_t area);

__global__ void __remove_background(const pixel64_t* input, pixel64_t* output, pixel64_t div_val, pixel64_t tolerance);
#endif /** KERNELS_CU_ */
