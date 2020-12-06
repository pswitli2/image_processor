#ifndef KERNELS_H_
#define KERNELS_H_

#include <stdint.h>
#include <vector>

typedef uint16_t pixel_t;
typedef std::vector<pixel_t> image_data_t;

void exec_kernel(const image_data_t& input, image_data_t& output);

#endif /** KERNELS_H_ */