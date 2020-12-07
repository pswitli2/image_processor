#ifndef IMAGE_HPP_
#define IMAGE_HPP_

#include <CImg.h>

#include "Utils.hpp"

typedef cimg_library::CImg<pixel16_t> cimg16_t;
typedef cimg_library::CImg<pixel64_t> cimg64_t;

/**
 * This class Abstracts the CImg library from the rest of the framework.
 */
class Image
{
public:

    Image() = default;

    /**
     * Create an image of a specified size.
     */
    Image(std::size_t width, std::size_t height)
    : m_cimg(width, height) { }

    ~Image() = default;

    /**
     * Set image data from a file.
     */
    void read(const std::string& filename)
    {
        cimg16_t cimg;
        cimg.assign(filename.c_str());

        cimg.move_to(m_cimg);
    }

    /**
     * Set image data from another image.
     */
    void set(const Image& image)
    {
        memcpy(data(), image.data(), bytes());
    }

    /**
     * Set image pixel values to 0.
     */
    void clear()
    {
        memset(data(), 0, bytes());
    }

    /**
     * Getters for data and sizes.
     */

    pixel64_t* data() { return m_cimg.data(); }
    const pixel64_t* data() const { return m_cimg.data(); }
    const cimg64_t& cimg() const { return m_cimg; }
    std::size_t width() const { return m_cimg.width(); }
    std::size_t height() const { return m_cimg.height(); }
    std::size_t bytes() const { return width() * height() * sizeof(pixel64_t); }

private:

    cimg64_t m_cimg;
};

#endif /** IMAGE_HPP_ */
