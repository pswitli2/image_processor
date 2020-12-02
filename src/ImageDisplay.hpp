#ifndef IMAGEDISPLAY_HPP_
#define IMAGEDISPLAY_HPP_

#include "CImg.h"

#include "Types.hpp"

class ImageDisplay
{
public:

    explicit ImageDisplay(
        const std::string& title)
    : m_title(title), m_cimgdisplay(), m_cimg()
    {
        m_cimgdisplay.show();
        set_title();
    }

    ~ImageDisplay()
    {
        close();
    }

    void update_image(
        const std::string& image_path)
    {
        m_cimg.assign(image_path.c_str());
        m_cimgdisplay.display(m_cimg);
        set_title();
    }

    void wait()
    {
        while (!m_cimgdisplay.is_closed())
        {
            m_cimgdisplay.wait();
        }
    }

    void close()
    {
        m_cimgdisplay.close();
    }

private:

    void set_title()
    {
        m_cimgdisplay.set_title("%s", m_title.c_str());
    }

    const std::string m_title;

    cimg_library::CImgDisplay m_cimgdisplay;
    cimg_library::CImg<pixel_t> m_cimg;
};

typedef std::shared_ptr<ImageDisplay> ImageDisplayPtr;

#endif /** IMAGEDISPLAY_HPP_ */