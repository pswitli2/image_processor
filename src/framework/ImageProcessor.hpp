#ifndef IMAGEPROCESSOR_HPP_
#define IMAGEPROCESSOR_HPP_

#include "BaseImageAlgorithm.hpp"

/**
 * The ImageProcessor handles reading images and routing them through algorithms.
 */
class ImageProcessor
{
public:
    ImageProcessor() = default;
    ~ImageProcessor() = default;

    /**
     * Initialize the ImageProcessor with a vector of algorithm vectors
     * Each algorithm vector represents an algorithm chain. Each image is processed
     * subsequently through each chain. Example:
     *     algorithms = { {algo1_cpu, algo2_cpu}, {algo1_cuda, algo2_cuda} }
     *     image1, image2, ..., imagen -> algo1_cpu -> algo2_cpu
     *     image1, image2, ..., imagen -> algo1_cuda -> algo2_cuda
     */
    bool initialize(const BaseImageAlgorithm_vecs &algorithms)
    {
        TRACE();

        m_algorithm_chains = algorithms;
        m_image_count = 0;

        // get input image files
        path_t input_dir = "";
        if (!ConfigFile::get_param(INPUT_DIR_PARAM_NAME, input_dir))
            return false;
        for (const auto &item : fs::directory_iterator(input_dir))
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
        Image first_image;
        first_image.read(m_inputs.front());
        m_width = first_image.width();
        m_height = first_image.height();
        const std::size_t area = m_width * m_height;

        // validate algorithm chains
        if (m_algorithm_chains.empty())
        {
            LOG(LogLevel::ERROR, "No algorithm chains found");
            return false;
        }
        size_t chain_idx = 0;
        for (const auto &algorithms : m_algorithm_chains)
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
            m_display = std::make_shared<ImageDisplay>("Raw");

        // initialize algorithms
        for (auto &algorithms : m_algorithm_chains)
        {
            for (auto &algorithm : algorithms)
            {
                if (!algorithm)
                {
                    LOG(LogLevel::ERROR, "Found nullptr algorithm");
                    return false;
                }

                // determine if this algorithm should have a dispaly
                bool display_flag = false;
                if (display_type == DisplayType::ALL || (algorithm == algorithms.back() && display_type == DisplayType::FIRST_LAST))
                    display_flag = true;

                // initialize algorithm
                if (!algorithm->initialize(m_width, m_height, area, display_flag))
                {
                    return false;
                }
            }
        }

        // set delay if necessary
        m_delay = 0.0;
        if (!ConfigFile::get_param(DELAY_PARAM_NAME, m_delay))
            return false;

        LOG(LogLevel::DEBUG, "ImageProcessor Initialized");

        return true;
    }

    /**
     * Process all images. This is the main loop for the processor.
     */
    bool process_images()
    {
        TRACE();

        // create some images to be used each iteration
        Image input(m_width, m_height);
        Image original_input(m_width, m_height);
        Image output(m_width, m_height);

        // loop over images
        for (const auto &input_file : m_inputs)
        {
            // read image from file
            input.read(input_file);

            // display image if necessary
            if (m_display)
                m_display->update_image(input);

            // save off input for each chain
            original_input.set(input);

            // route image through each chain
            for (auto &algorithms : m_algorithm_chains)
            {
                for (auto &algorithm : algorithms)
                {
                    // update algorithm
                    output.clear();
                    if (!algorithm->update(input, output))
                    {
                        return false;
                    }

                    // delay if necessary
                    if (m_delay > 0.0)
                        usleep(m_delay * 1e6);
    
                    // set output to next input
                    input.set(output);
                }

                // reset input to original image
                input.set(original_input);
            }
            m_image_count++;

            progress_bar();

            LOG(LogLevel::TRACE, "Processed Image #", m_image_count);
        }

        std::cout << std::endl;

        LOG(LogLevel::DEBUG, "Finished processing images");
        return true;
    }

    /**
     * Get number of images processed
     */
    std::size_t image_count() const { return m_image_count; }

    /**
     * Log ImageProcessor information
     */
    void log_info() const
    {
        std::stringstream ss;
        ss << std::endl
           << "ImageProcessor info:" << std::endl
           << "  Image info:" << std::endl
           << "    Number of images = " << m_inputs.size() << std::endl
           << "    Image width      = " << m_width << std::endl
           << "    Image height     = " << m_height << std::endl
           << "    Image area       = " << m_width * m_height << std::endl
           << "  Algorithms used:" << std::endl;
        size_t chain_idx = 0;
        for (const auto &algorithms : m_algorithm_chains)
        {
            ss << "    Chain " << chain_idx << std::endl;
            for (const auto &algorithm : algorithms)
                ss << "      " << algorithm->name() << std::endl;
            chain_idx++;
        }

        LOG(LogLevel::INFO, ss.str());
    }

    /**
     * Log ImageProcessor results
     */
    void log_results() const
    {
        std::stringstream ss;
        ss << std::endl
           << "ImageProcessor results:" << std::endl
           << "  Algorithm runtime: (Doesnt include disk IO, image displays, etc.)" << std::endl;
        size_t chain_idx = 0;
        double total_duration = 0.0;
        for (const auto &algorithms : m_algorithm_chains)
        {
            double chain_duration = 0.0;
            ss << "    Chain " << chain_idx << std::endl;
            for (const auto &algorithm : algorithms)
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

    void progress_bar()
    {
        if (Logger::getLevel() != LogLevel::TRACE) 
        {
            const std::size_t bar_size = 60; 
            const auto progress = (double) m_image_count / (double) m_inputs.size();
            std::cout << "Processing images: [";
            std::size_t pos = (double) bar_size * progress;
            for (std::size_t i = 0; i < bar_size; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout.precision(4);
            std::cout << "] " << std::setw(6) << progress * 100.0 << " %\r";
            std::cout.flush();
        }
    }

    std::vector<path_t> m_inputs;

    BaseImageAlgorithm_vecs m_algorithm_chains;
    ImageDisplay_ptr m_display;

    std::size_t m_width;
    std::size_t m_height;

    std::size_t m_image_count;

    double m_delay;
};

#endif /** IMAGEPROCESSOR_HPP_ */