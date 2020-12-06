#include "KernelWrapper.hpp"
#include "CudaUtils.hpp"

__host__ KernelWrapper::KernelWrapper(std::size_t width, std::size_t height)
: m_width(width), m_height(height), m_area(width * height), m_numbytes(m_area * sizeof(pixel_t))
{
    CUDA_MALLOC((void**) &m_d_input, num_bytes()); // TODO try cudaMallocHost
    CUDA_MALLOC((void**) &m_d_output, num_bytes());
}

__host__ KernelWrapper::~KernelWrapper()
{
    CUDA_FREE(&m_d_input); // TODO try cudaFreeHost
    CUDA_FREE(&m_d_output);
}

__host__ void KernelWrapper::execute(const image_data_t& input, image_data_t& output)
{
    cudaMemcpy(m_d_input, input.data(), num_bytes(), cudaMemcpyHostToDevice);

    execute_impl();

    cudaMemcpy(output.data(), m_d_output, num_bytes(), cudaMemcpyDeviceToHost);
}
