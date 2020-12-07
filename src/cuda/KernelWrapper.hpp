#ifndef KERNEL_WRAPPER_HPP_
#define KERNEL_WRAPPER_HPP_

#include <stdint.h>
#include <vector>

#include "Types.hpp"

class CUstream_st;

/**
 * The KernelWrapper should be subclassed by an algorithm that wants to used CUDA
 * kernels in its implementation.
 */
class KernelWrapper
{
public:

    /**
     * Destructor will cleanup m_d_input, m_d_output, and create m_stream
     */
    virtual ~KernelWrapper();

    /**
     * Execute kernel with new image. This will copy host data to/from
     * m_d_input/m_d_output arrays
     */
    void execute(const pixel64_t* input, pixel64_t* output);

protected:

    /**
     * Create KernelWrapper from width and height. This will allocate
     * m_d_input, m_d_output, and create m_stream
     */
    KernelWrapper(std::size_t width, std::size_t height);

    /**
     * Getters for various sizes.
     */
    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t area() const { return m_area; }
    std::size_t num_bytes() const { return m_numbytes; }

    pixel64_t* m_d_input;  // device buffer with input image data
    pixel64_t* m_d_output; // device buffer with output image data
    CUstream_st* m_stream; // cuda stream used for async copying

private:

    /**
     * This should be implemented by subclasses. Before execution, m_d_input will be
     * set to the new image data. Before exit, the implementation should set m_d_output
     * to the output image.
     */
    virtual void execute_impl() = 0;

    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_area;
    std::size_t m_numbytes;
};

#endif /** KERNEL_WRAPPER_HPP_ */