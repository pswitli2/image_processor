#ifndef BASEIMAGEALGORITHM_HPP_
#define BASEIMAGEALGORITHM_HPP_

#include "ConfigFile.hpp"
#include "ImageDisplay.hpp"

class BaseImageAlgorithm
{
public:

    virtual ~BaseImageAlgorithm() = default;

    virtual std::string name() const = 0;

    bool initialize(const std::size_t width, const std::size_t height, const std::size_t area, bool display_flag, const path_t& output_dir)
    {
        TRACE();

        m_width = width;
        m_height = height;
        m_area = area;
        m_image_count = 0;
        m_duration_ns = 0.0;
        m_outputdir = output_dir;

        // init Display if necessary
        if (display_flag)
        {
            m_display = std::make_shared<ImageDisplay>("Post " + name());
        }

        if (!initialize_impl())
        {
            LOG(LogLevel::ERROR, name(), " - Failed to initialize");
            return false;
        }

        LOG(LogLevel::DEBUG, name(), " - Initialized");

        return true;
    }

    virtual bool update(const path_t& input_file, const image_t& input, image_t& output)
    {
        TRACE();

        m_image_count++;
        const auto start = TIME_NOW();
        if (!update_impl(input, output))
        {
            LOG(LogLevel::ERROR, name(), " - Failed updating (Image #", m_image_count, ")");
            return false;
        }
        const auto end = TIME_NOW();
        m_duration_ns += DURATION_NS(end - start);
        LOG(LogLevel::TRACE, name(), " - Updated (Image #", m_image_count, ", total duration: ", duration(), " sec)");

        const path_t output_file = m_outputdir / input_file.filename();
        output.write(std::string(output_file));
        LOG(LogLevel::TRACE, name(), " - Wrote image to disk: ", output_file);

        if (m_display)
        {
            m_display->update_image(output_file);
        }

        return true;
    }

    double duration() const { return m_duration_ns / 1e9; }

    std::string output_dir() const { return m_outputdir; }

protected:

    BaseImageAlgorithm() = default;

    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t area() const { return m_area; }

private:

    virtual bool initialize_impl() { return true; }

    virtual bool update_impl(const image_t& input, image_t& output) = 0;

    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_area;
    std::size_t m_image_count;

    double m_duration_ns;

    path_t m_outputdir;

    ImageDisplay_ptr m_display;
};

#endif /** BASEIMAGEALGORITHM_HPP_ */
