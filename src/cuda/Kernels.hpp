#ifndef KERNELS_CU_
#define KERNELS_CU_

#include "CudaUtils.hpp"

__global__ void __sum(const pixel64_t* input, pixel64_t* output, std::size_t length);

__global__ void __pixel_minus_mean_pow2(const pixel64_t* input, pixel64_t* output, pixel64_t mean);

__global__ void __clear_image(pixel64_t* inout);

__global__ void __threshold(const pixel64_t* input, pixel64_t* output, pixel64_t threshold);

#endif /** KERNELS_CU_ */
