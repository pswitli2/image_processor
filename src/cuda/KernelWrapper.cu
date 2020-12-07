#include "KernelWrapper.hpp"
#include "CudaUtils.hpp"

__host__ KernelWrapper::KernelWrapper(std::size_t width, std::size_t height)
: m_width(width), m_height(height), m_area(width * height), m_numbytes(m_area * sizeof(pixel64_t))
{
    // init device arrays and stream
    CUDA_MALLOC((void**) &m_d_input, num_bytes());
    CUDA_MALLOC((void**) &m_d_output, num_bytes());
    cudaStreamCreate(&m_stream);
}

__host__ KernelWrapper::~KernelWrapper()
{
    // cleanup device arrays and streams
    CUDA_FREE(&m_d_input);
    CUDA_FREE(&m_d_output);
    cudaStreamDestroy(m_stream);
}

__host__ void KernelWrapper::execute(const pixel64_t* input, pixel64_t* output)
{
    // copy to device array
    cudaMemcpyAsync(m_d_input, input, num_bytes(), cudaMemcpyHostToDevice, m_stream);

    // execute algorithm implementation
    execute_impl();

    // copy output to host array and sync stream
    cudaMemcpyAsync(output, m_d_output, num_bytes(), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
}
