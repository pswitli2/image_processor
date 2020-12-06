#include "KernelWrapper.hpp"
#include "CudaUtils.hpp"

__host__ KernelWrapper::KernelWrapper(std::size_t width, std::size_t height)
: m_width(width), m_height(height), m_area(width * height), m_numbytes(m_area * sizeof(pixel64_t))
{
    CUDA_MALLOC((void**) &m_d_input, num_bytes());
    CUDA_MALLOC((void**) &m_d_output, num_bytes());
}

__host__ KernelWrapper::~KernelWrapper()
{
    CUDA_FREE(&m_d_input);
    CUDA_FREE(&m_d_output);
}

__host__ void KernelWrapper::execute(const pixel64_t* input, pixel64_t* output)
{
    cudaMemcpy(m_d_input, input, num_bytes(), cudaMemcpyHostToDevice);

    execute_impl();

    cudaMemcpy(output, m_d_output, num_bytes(), cudaMemcpyDeviceToHost);
}
