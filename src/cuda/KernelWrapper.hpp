#ifndef KERNEL_WRAPPER_HPP_
#define KERNEL_WRAPPER_HPP_

#include <stdint.h>
#include <vector>

#include "Types.hpp"

class CUstream_st;

class KernelWrapper
{
public:

    virtual ~KernelWrapper();

    void execute(const pixel64_t* input, pixel64_t* output);

protected:

    KernelWrapper(std::size_t width, std::size_t height);

    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t area() const { return m_area; }
    std::size_t num_bytes() const { return m_numbytes; }

    pixel64_t* m_d_input;
    pixel64_t* m_d_output;

    CUstream_st* m_stream;

private:

    virtual void execute_impl() = 0;

    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_area;
    std::size_t m_numbytes;
};

#endif /** KERNEL_WRAPPER_HPP_ */