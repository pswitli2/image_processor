#include "Kernels.h"

__global__ void kern(const image_data_t& input, image_data_t& output)
{

}

void exec_kernel(const image_data_t& input, image_data_t& output)
{
    kern<<<1, 1>>>(input, output);
}
