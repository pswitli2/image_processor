#ifndef CUDAUTILS_HPP_
#define CUDAUTILS_HPP_

#include "Types.hpp"

/**
 * Define some common routines used throughout the CUDA algorithms.
 */

/** Create static device variable for max pixel value */
__device__ static pixel16_t D_MAX_PIXEL_VAL = MAX_PIXEL_VAL;

/** Set functions for creating/destroying Cuda arrays */
#define CUDA_MALLOC cudaMalloc
#define CUDA_FREE cudaFree

#endif /** CUDAUTILS_HPP_ */