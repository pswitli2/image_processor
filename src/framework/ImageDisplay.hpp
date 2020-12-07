#ifndef IMAGEDISPLAY_HPP_
#define IMAGEDISPLAY_HPP_

#include "Image.hpp"
#include "Logger.hpp"

/**
 * ImageDisplay creates a CImgDisplay window and provides functions to update the dispalyed image.
 */
class ImageDisplay
{
public:

    /**
     * Create ImageDispaly with a title.
     */
    explicit ImageDisplay(const std::string& title)
    : m_title(title), m_cimgdisplay()
    {
        TRACE();

        LOG(LogLevel::DEBUG, " Creating ImageDisplay with title: ", m_title)

        m_cimgdisplay.show();
        set_title();
    }

    /**
     * Close display on exit.
     */
    ~ImageDisplay()
    {
        TRACE();

        m_cimgdisplay.close();
    }

    /**
     * Update the display with a new image.
     */
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
