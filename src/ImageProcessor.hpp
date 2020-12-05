#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include "BaseImageAlgorithm.hpp"

class ImageProcessor
{
public:

    ImageProcessor() = default;
    ~ImageProcessor() = default;

    bool initialize(const BaseImageAlgorithm_vec& algorithms)
    {
        TRACE();

        m_algorithms = algorithms;
        m_image_count = 0;

        // get input image files
        path_t input_dir = "";
        if (!ConfigFile::get_param(INPUT_DIR_PARAM_NAME, input_dir))
            return false;
        for (const auto& item: fs::directory_iterator(input_dir))
        {
            std::string itempath = item.path();
            if (item.is_regular_file() && itempath.find(".png") == itempath.size() - 4)
                m_inputs.push_back(itempath);
        }
        if (m_inputs.empty())
        {
            LOG(LogLevel::ERROR, "No input files found");
            return false;
        }
        std::sort(m_inputs.begin(), m_inputs.end());

        // open first image to get width and height
        png::image<pixel_t> first_image(m_inputs.front()); // TODO handle invalid png
        m_width = first_image.get_width();
        m_height = first_image.get_height();
        const std::size_t area = m_width * m_height;
    
        if (m_algorithms.empty())
        {
            LOG(LogLevel::ERROR, "No algorithms found");
            return false;
        }

        // open display if necessary
        DisplayType display_type = DisplayType::NONE;
        if (!ConfigFile::get_param(DISPLAY_TYPE_PARAM_NAME, display_type))
            return false;
        if (display_type == DisplayType::ALL || display_type == DisplayType::FIRST_LAST)
            m_display = std::make_shared<ImageDisplay>("Original");

        // initialize algorithms
        for (auto& algorithm: m_algorithms)
        {
            if (!algorithm)
            {
                LOG(LogLevel::ERROR, "Found nullptr algorithm");
                return false;
            }
            bool display_flag = false;
            if (display_type == DisplayType::ALL || (algorithm == m_algorithms.back() && display_type == DisplayType::FIRST_LAST))
                display_flag = true;

            if (!algorithm->initialize(m_width, m_height, area, display_flag))
            {
                LOG(LogLevel::ERROR, "Unable to initialize algorithm: " + algorithm->name());

                return false;
            }
        }

        // set delay
        m_delay = 0.0;
        if (!ConfigFile::get_param(DELAY_PARAM_NAME, m_delay))
            return false;

        // create output directory
        path_t output_dir = "";
        if (!ConfigFile::get_param(OUTPUT_DIR_PARAM_NAME, output_dir))
            return false;
        fs::create_directories(output_dir);

        LOG(LogLevel::INFO, "Creating output directory: ", output_dir);
        LOG(LogLevel::INFO, "Initialized ImageProcessor (width: ", m_width, ", height: ", m_height, ", area: ", area, ")");

        return true;
    }

    bool process_images()
    {
        TRACE();

        png::image<pixel_t> input;
        png::image<pixel_t> output;
        for (const auto& input_file: m_inputs)
        {
            if (m_display)
                m_display->update_image(input_file);

            input.read(input_file);

            for (auto& algorithm: m_algorithms)
            {
                output = png::image<pixel_t>(m_width, m_height); // TODO verify this sets zeros
                if (!algorithm->update(input_file, input, output))
                {
                    LOG(LogLevel::ERROR, "Unable to update algorithm: ", algorithm->name());
                    return false;
                }

                usleep(m_delay * 1e6);
                input = output;
            }
            m_image_count++;
            LOG(LogLevel::TRACE, "Processed Image #", m_image_count);
        }

        return true;
    }

    std::size_t image_count() { return m_image_count; }

protected:
private:

    std::vector<path_t> m_inputs;

    BaseImageAlgorithm_vec m_algorithms;
    ImageDisplay_ptr m_display;

    std::size_t m_width;
    std::size_t m_height;

    std::size_t m_image_count;

    double m_delay;
};

#endif /** IMAGEPROCESSOR_HPP_ */