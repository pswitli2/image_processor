#ifndef IMAGEDISPLAY_HPP_
#define IMAGEDISPLAY_HPP_

#include "Image.hpp"
#include "Logger.hpp"

class ImageDisplay
{
public:

    explicit ImageDisplay(const std::string& title)
    : m_title(title), m_cimgdisplay()
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

    void update_image(const Image& image)
    {
        TRACE();

        cimg16_t cimg = image.cimg();
        m_cimgdisplay.display(cimg);
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
};

typedef std::shared_ptr<ImageDisplay> ImageDisplay_ptr;

#endif /** IMAGEDISPLAY_HPP_ */
