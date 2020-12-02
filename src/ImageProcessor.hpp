#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include <filesystem>

#include "Types.hpp"

#include "BaseImageAlgorithm.hpp"
#include "ImageDisplay.hpp"

class ImageProcessor
{
public:

    ImageProcessor(
        const BaseImageAlgorithmPtrVec& algorithms,
        const std::string& input_dir,
        const std::string& output_dir,
        bool pre_display_flag = false,
        bool post_display_flag = false)
    : m_inputs(),
      m_algorithms(algorithms),
      m_predisplay(),
      m_postdisplay()
    {
        for (const auto& item: std::filesystem::directory_iterator(input_dir))
        {
            std::string itempath = item.path();
            if (item.is_regular_file() && itempath.find(".png") == itempath.size() - 4)
            {
                m_inputs.push_back(itempath);
            }
        }
        if (m_inputs.empty())
        {
            // TODO handle
        }
        std::sort(m_inputs.begin(), m_inputs.end());

        png::image<pixel_t> first_image(m_inputs.front()); // TODO handle error
        m_width = first_image.get_width();
        m_height = first_image.get_height();
        m_area = m_width * m_height;

        if (m_algorithms.empty())
        {
            // TODO handle
        }
    
        for (auto& algorithm: m_algorithms)
        {
            if (!algorithm)
            {
                // TODO handle
            }

            if (!algorithm->initialize())
            {
                // TODO handle
            }
        }

        if (pre_display_flag)
        {
            m_predisplay = std::make_shared<ImageDisplay>("Pre " + m_algorithms.front()->name());
        }

        if (post_display_flag)
        {
            m_postdisplay = std::make_shared<ImageDisplay>("Post " + m_algorithms.back()->name());
        }
    
        std::filesystem::create_directories(output_dir);
    }

    ~ImageProcessor() = default;

    bool process_inputs()
    {
        png::image<pixel_t> input_image;
        png::image<pixel_t> output_image;
        for (const auto& input: m_inputs)
        {
            input_image.read(input);
            output_image = png::image<pixel_t>(m_width, m_height); // TODO verify this sets zeros

            for (auto& algorithm: m_algorithms)
            {
                if (!algorithm->update(input_image, output_image))
                {
                    // TODO handle
                }
                input_image = output_image;
                output_image = png::image<pixel_t>(m_width, m_height);
            }
        }

        return true;
    }

protected:

    size_t width() { return m_width; }

    size_t height() { return m_height; }

    size_t area() { return m_area; }

private:

    std::vector<std::string> m_inputs;

    BaseImageAlgorithmPtrVec m_algorithms;

    ImageDisplayPtr m_predisplay;
    ImageDisplayPtr m_postdisplay;

    size_t m_width;
    size_t m_height;
    size_t m_area;
};

#endif /** IMAGEPROCESSOR_HPP_ */