#ifndef KERNEL_WRAPPER_HPP_
#define KERNEL_WRAPPER_HPP_

#include <stdint.h>
#include <vector>

typedef uint16_t pixel_t;
typedef std::vector<pixel_t> image_data_t;

class KernelWrapper
{
public:

    ~KernelWrapper();

    void execute(const image_data_t& input, image_data_t& output);

protected:

    KernelWrapper(std::size_t width, std::size_t height);

    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t area() const { return m_area; }
    std::size_t num_bytes() const { return m_numbytes; }

    pixel_t* m_d_input;
    pixel_t* m_d_output;

private:

    virtual void execute_impl() = 0;

    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_area;
    std::size_t m_numbytes;
};

#endif /** KERNEL_WRAPPER_HPP_ */