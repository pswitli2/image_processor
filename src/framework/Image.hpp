#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include <CImg.h>

#include "Utils.hpp"

typedef cimg_library::CImg<pixel16_t> cimg16_t;
typedef cimg_library::CImg<pixel64_t> cimg64_t;

class Image
{
public:

    Image() = default;

    Image(std::size_t width, std::size_t height)
    : m_cimg(width, height) { }

    ~Image() = default;

    void read(const std::string& filename)
    {
        cimg16_t cimg;
        cimg.assign(filename.c_str());

        cimg.move_to(m_cimg);
    }

    void set(const Image& image)
    {
        memcpy(data(), image.data(), bytes());
    }

    void clear()
    {
        memset(data(), 0, bytes());
    }

    pixel64_t* data() { return m_cimg.data(); }
    const pixel64_t* data() const { return m_cimg.data(); }

    std::size_t width() const { return m_cimg.width(); }
    std::size_t height() const { return m_cimg.height(); }
    std::size_t bytes() const { return width() * height() * sizeof(pixel64_t); }

    const cimg64_t& cimg() const { return m_cimg; }

private:

    cimg64_t m_cimg;
};

#endif /** IMAGE_HPP_ */
