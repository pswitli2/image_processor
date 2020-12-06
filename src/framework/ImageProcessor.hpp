#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include "BaseImageAlgorithm.hpp"

class ImageProcessor
{
public:

    ImageProcessor() = default;
    ~ImageProcessor() = default;

    bool initialize(const BaseImageAlgorithm_vecs& algorithms)
    {
        TRACE();

        m_algorithm_chains = algorithms;
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
        image_t first_image(m_inputs.front()); // TODO handle invalid png
        m_width = first_image.get_width();
        m_height = first_image.get_height();
        const std::size_t area = m_width * m_height;
    
        if (m_algorithm_chains.empty())
        {
            LOG(LogLevel::ERROR, "No algorithm chains found");
            return false;
        }

        size_t chain_idx = 0;
        for (const auto& algorithms: m_algorithm_chains)
        {
            if (algorithms.empty())
            {
                LOG(LogLevel::ERROR, "No algorithms found in chain ", chain_idx);
                return false;
            }
            chain_idx++;
        }

        // open display if necessary
        DisplayType display_type = DisplayType::NONE;
        if (!ConfigFile::get_param(DISPLAY_TYPE_PARAM_NAME, display_type))
            return false;
        if (display_type == DisplayType::ALL || display_type == DisplayType::FIRST_LAST)
            m_display = std::make_shared<ImageDisplay>("Original");

        // create output directory
        path_t base_outputdir = "";
        if (!ConfigFile::get_param(OUTPUT_DIR_PARAM_NAME, base_outputdir))
        {
            return false;
        }
    
        // initialize algorithms
        chain_idx = 0;
        for (auto& algorithms: m_algorithm_chains)
        {
            std::stringstream ss;
            ss << "chain" << chain_idx;
            const path_t chain_outputdir = base_outputdir / path_t(ss.str());
            for (auto& algorithm: algorithms)
            {
                if (!algorithm)
                {
                    LOG(LogLevel::ERROR, "Found nullptr algorithm");
                    return false;
                }

                const path_t output_dir = chain_outputdir / path_t(algorithm->name());
                fs::create_directories(output_dir);
                bool display_flag = false;
                if (display_type == DisplayType::ALL || (algorithm == algorithms.back() && display_type == DisplayType::FIRST_LAST))
                    display_flag = true;

                if (!algorithm->initialize(m_width, m_height, area, display_flag, output_dir))
                {
                    return false;
                }
            }
            chain_idx++;
        }

        // set delay
        m_delay = 0.0;
        if (!ConfigFile::get_param(DELAY_PARAM_NAME, m_delay))
            return false;

        LOG(LogLevel::DEBUG, "ImageProcessor Initialized");

        return true;
    }

    bool process_images()
    {
        TRACE();

        image_t input;
        image_t output;
        for (const auto& input_file: m_inputs)
        {
            if (m_display)
                m_display->update_image(input_file);

            input.read(input_file);
            auto input_data = image_to_vec(input);

            auto input_saved = input_data;

            for (auto& algorithms: m_algorithm_chains)
            {
                input_data = input_saved;
                for (auto& algorithm: algorithms)
                {
                    auto output_data = image_data_t(m_width * m_height, 0);
                    if (!algorithm->update(input_file, input_data, output_data))
                    {
                        return false;
                    }

                    if (m_delay > 0.0)
                        usleep(m_delay * 1e6);
                    input_data = output_data;
                }
            }
            m_image_count++;
            LOG(LogLevel::TRACE, "Processed Image #", m_image_count);
        }

        LOG(LogLevel::DEBUG, "Finished processing images");
        return true;
    }

    std::size_t image_count() const { return m_image_count; }

    void log_info() const
    {
        std::stringstream ss;
        ss << std::endl
           << "ImageProcessor info:"                          << std::endl
           << "  Image info:"                                 << std::endl
           << "    Number of images = " << m_inputs.size()    << std::endl
           << "    Image width      = " << m_width            << std::endl
           << "    Image height     = " << m_height           << std::endl
           << "    Image area       = " << m_width * m_height << std::endl
           << "  Algorithms used:"                            << std::endl;
        size_t chain_idx = 0;
        for (const auto& algorithms: m_algorithm_chains)
        {
            ss << "    Chain " << chain_idx << std::endl;
            for (const auto& algorithm: algorithms)
                ss << "      " << algorithm->name() << ": writing images to: " << algorithm->output_dir() << std::endl;
            chain_idx++;
        }

        LOG(LogLevel::INFO, ss.str());
    }

    void log_results() const
    {
        std::stringstream ss;
        ss << std::endl
           << "ImageProcessor results:" << std::endl
           << "  Algorithm runtime: (Doesnt include disk reading/writing, image displays, etc.)" << std::endl;
        size_t chain_idx = 0;
        double total_duration = 0.0;
        for (const auto& algorithms: m_algorithm_chains)
        {
            double chain_duration = 0.0;
            ss << "    Chain " << chain_idx << std::endl;
            for (const auto& algorithm: algorithms)
            {
                ss << "      " << std::left << std::setw(25) << algorithm->name() << " " << algorithm->duration() << " seconds" << std::endl;
                chain_duration += algorithm->duration();
            }
            ss << "      Chain " << chain_idx << " total duration    " << chain_duration << " seconds" << std::endl;
            chain_idx++;
            total_duration += chain_duration;
        }

        ss << "  Total ImageProcessor duration: " << total_duration << std::endl;
        LOG(LogLevel::INFO, ss.str());
    }

protected:
private:

    std::vector<path_t> m_inputs;

    BaseImageAlgorithm_vecs m_algorithm_chains;
    ImageDisplay_ptr m_display;

    std::size_t m_width;
    std::size_t m_height;

    std::size_t m_image_count;

    double m_delay;
};

#endif /** IMAGEPROCESSOR_HPP_ */