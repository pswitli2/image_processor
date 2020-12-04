#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include <filesystem>

#include "BaseImageAlgorithm.hpp"
#include "Types.hpp"

class ImageProcessor
{
public:

    ImageProcessor() = default;
    ~ImageProcessor() = default;

    bool initialize(const BaseImageAlgorithm_vec& algorithms)
    {
        m_algorithms = algorithms;

        for (const auto& item: std::filesystem::directory_iterator(ConfigFile::input_dir()))
        {
            std::string itempath = item.path();
            if (item.is_regular_file() && itempath.find(".png") == itempath.size() - 4)
            {
                m_inputs.push_back(itempath);
            }
        }
        if (m_inputs.empty())
        {
            m_error = "No input images found";
            return false;
        }

        std::sort(m_inputs.begin(), m_inputs.end());

        png::image<pixel_t> first_image(m_inputs.front()); // TODO handle invalid png
        m_width = first_image.get_width();
        m_height = first_image.get_height();
        const size_t area = m_width * m_height;
    
        if (m_algorithms.empty())
        {
            m_error = "No algorithms found";
            return false;
        }
    
        const auto display_type = ConfigFile::display_type();
            if (display_type == DisplayType::ALL || display_type == DisplayType::FIRST_LAST)
            {
                m_display = std::make_shared<ImageDisplay>("Original");
            }

        for (auto& algorithm: m_algorithms)
        {
            if (!algorithm)
            {
                m_error = "Found nullptr algorithm";
                return false;
            }

            bool display_flag = false;
            if (display_type == DisplayType::ALL ||
                (algorithm == m_algorithms.back() && display_type == DisplayType::FIRST_LAST))
            {
                display_flag = true;
            }

            if (!algorithm->initialize(m_width, m_height, area, display_flag))
            {
                m_error = "Unable to initialize algorithm: " + algorithm->name() + ", error: " + algorithm->error();
                return false;
            }
        }
    
        std::filesystem::create_directories(ConfigFile::output_dir());

        return true;
    }

    bool process_images()
    {
        png::image<pixel_t> input;
        png::image<pixel_t> output;
        for (const auto& input_file: m_inputs)
        {
            if (m_display)
            {
                m_display->update_image(input_file);
            }

            input.read(input_file);

            for (auto& algorithm: m_algorithms)
            {
                output = png::image<pixel_t>(m_width, m_height); // TODO verify this sets zeros
                if (!algorithm->update(input_file, input, output))
                {
                    m_error = "Unable to update algorithm: " + algorithm->name() + ", error: " + algorithm->error();
                    return false;
                }

                usleep(ConfigFile::delay() * 1e6);
                input = output;
            }
            m_image_count++;
        }

        return true;
    }

    std::string error() { return m_error; }

    size_t image_count() { return m_image_count; }

protected:
private:

    std::vector<std::filesystem::path> m_inputs;

    BaseImageAlgorithm_vec m_algorithms;
    ImageDisplay_ptr m_display;

    std::string m_error;

    size_t m_width;
    size_t m_height;

    size_t m_image_count;
};

#endif /** IMAGEPROCESSOR_HPP_ */