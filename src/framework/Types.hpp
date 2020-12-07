#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <stdint.h>

/**
 * Set some common typedefs and defines used in C++ and CUDA code.
 */

/** Pixels are read from disk with this type. */
typedef uint16_t pixel16_t;

/** Multiple algorithms require larger types for processing, so uint64_t's are used. */
typedef uint64_t pixel64_t;

/** Maximum value a pixel can be. */
#define MAX_PIXEL_VAL 65535

#endif /** TYPES_HPP_ */