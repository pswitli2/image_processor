#ifndef CUDAUTILS_HPP_
#define CUDAUTILS_HPP_

#include "Types.hpp"

__device__ static pixel16_t D_MAX_PIXEL_VAL = MAX_PIXEL_VAL;

#define CUDA_MALLOC cudaMalloc
#define CUDA_FREE cudaFree

#endif /** CUDAUTILS_HPP_ */