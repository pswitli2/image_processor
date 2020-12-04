#ifndef BASEIMAGEALGORITHM_HPP_
#define BASEIMAGEALGORITHM_HPP_

#include <chrono>

#include "ImageDisplay.hpp"

class BaseImageAlgorithm
{
public:

    virtual ~BaseImageAlgorithm() = default;

    virtual std::string name() = 0;

    std::string error() { return m_error; }

    bool initialize(const size_t width, const size_t height, const size_t area, bool display_flag)
    {
        m_width = width;
        m_height = height;
        m_area = area;

        m_duration = 0.0;

        if (display_flag)
        {
            m_display = std::make_shared<ImageDisplay>("Post " + name());
        }

        m_output_dir = ConfigFile::output_dir() / std::filesystem::path(name());
        std::filesystem::create_directories(m_output_dir);
        return initialize_impl();
    }

    virtual bool update(const std::filesystem::path& input_file, const Image& input, Image& output)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        if (!update_impl(input, output))
        {
            return false;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        m_duration += (double) std::chrono::duration_cast<std::chrono::microseconds>( end - start ).count();

        const std::string output_file = m_output_dir / input_file.filename();
        output.write(output_file);
        if (m_display)
        {
            m_display->update_image(output_file);
        }

        return true;
    }

    double duration() { return m_duration / 1e6; }

protected:

    BaseImageAlgorithm() = default;

    size_t width() { return m_width; }
    size_t height() { return m_height; }
    size_t area() { return m_area; }

    std::string m_error;

private:

    virtual bool initialize_impl() { return true; }

    virtual bool update_impl(const Image& input, Image& output) = 0;

    size_t m_width;
    size_t m_height;
    size_t m_area;

    double m_duration;

    std::filesystem::path m_output_dir;

    ImageDisplay_ptr m_display;
};

typedef std::shared_ptr<BaseImageAlgorithm> BaseImageAlgorithm_ptr;
typedef std::vector<BaseImageAlgorithm_ptr> BaseImageAlgorithm_vec;

#endif /** BASEIMAGEALGORITHM_HPP_ */