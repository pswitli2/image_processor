#include "KernelWrapper.hpp"
#include "CudaUtils.hpp"

__host__ KernelWrapper::KernelWrapper(std::size_t width, std::size_t height)
: m_width(width), m_height(height), m_area(width * height), m_numbytes(m_area * sizeof(pixel_t))
{
    CUDA_MALLOC((void**) &d_input, num_bytes()); // TODO try cudaMallocHost
    CUDA_MALLOC((void**) &d_output, num_bytes());
}

__host__ KernelWrapper::~KernelWrapper()
{
    CUDA_FREE(&d_input); // TODO try cudaFreeHost
    CUDA_FREE(&d_output);
}

__host__ void KernelWrapper::execute(const image_data_t& input, image_data_t& output)
{
    cudaMemcpy(d_input, input.data(), num_bytes(), cudaMemcpyHostToDevice);

    execute_impl();

    cudaMemcpy(output.data(), d_output, num_bytes(), cudaMemcpyDeviceToHost);
}
