#ifndef IMAGEDISPLAY_HPP_
#define IMAGEDISPLAY_HPP_

#include <CImg.h>

#include "Logger.hpp"

class ImageDisplay
{
public:

    explicit ImageDisplay(const std::string& title)
    : m_title(title), m_cimgdisplay(), m_cimg()
    {
        TRACE();

        LOG(LogLevel::DEBUG, " Creating ImageDisplay with title: ", m_title)

        m_cimgdisplay.show();
        set_title();
    }

    ~ImageDisplay()
    {
        TRACE();

        m_cimgdisplay.close();
    }

    void update_image(const path_t& image_path)
    {
        TRACE();

        m_cimg.assign(image_path.c_str());
        m_cimgdisplay.display(m_cimg);
        set_title();
    }

private:

    void set_title()
    {
        TRACE();

        m_cimgdisplay.set_title("%s", m_title.c_str());
    }

    const std::string m_title;

    cimg_library::CImgDisplay m_cimgdisplay;
    cimg_library::CImg<pixel_t> m_cimg;
};

#endif /** IMAGEDISPLAY_HPP_ */
