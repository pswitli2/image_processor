#ifndef KERNELS_CU_
#define KERNELS_CU_

#include "CudaUtils.hpp"

/**
 * All CUDA kernels should be defined in this file.
 */

/** Get current array index. */
__device__ std::size_t __get_idx();

/** Copy image data from input to output. */
__global__ void __copy_image(const pixel64_t* input, pixel64_t* output);

/** Sum [idx + idx + 1 + ... idx + length] input pixels and place in output[idx]. */
__global__ void __sum(const pixel64_t* input, pixel64_t* output, std::size_t length);

/** For each input pixel p, output = (p - mean) ^ 2. */
__global__ void __pixel_minus_mean_pow2(const pixel64_t* input, pixel64_t* output, pixel64_t mean);

/** Set all pixels in an image to 0. */
__global__ void __clear_image(pixel64_t* inout);

/** Perform thresholding on input and place in output. */
__global__ void __threshold(const pixel64_t* input, pixel64_t* output, pixel64_t threshold);

/** Calculate the sum of history buffer pixels and place in output. */
__global__ void __sum_history(const pixel64_t* input, pixel64_t* output, std::size_t history_size, std::size_t area);

/** Perform background removal on input and place in output. */
__global__ void __remove_background(const pixel64_t* input, pixel64_t* output, pixel64_t div_val, pixel64_t tolerance);

/** Perform lone pixel removal on input and place in output. */
__global__ void __lone_pixel(const pixel64_t* input, pixel64_t* output, std::size_t num_adjacent, std::size_t width);

#endif /** KERNELS_CU_ */
