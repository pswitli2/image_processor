#ifndef BASEIMAGEALGORITHM_HPP_
#define BASEIMAGEALGORITHM_HPP_

#include "ConfigFile.hpp"
#include "ImageDisplay.hpp"

/**
 * Algorithms should inherit from this class to run through ImageProcessor.
 *     name(), initialize_impl() (optional), and update_impl() should be implemented
 */
class BaseImageAlgorithm
{
public:

    virtual ~BaseImageAlgorithm() = default;

    /**
     * Return algorithm name
     * */
    virtual std::string name() const = 0;

    /**
     * Initialize algoirthm
     */
    bool initialize(const std::size_t width, const std::size_t height, const std::size_t area, bool display_flag)
    {
        TRACE();

        m_width = width;
        m_height = height;
        m_area = area;
        m_image_count = 0;
        m_duration_ns = 0.0;

        // init Display if necessary
        if (display_flag)
        {
            m_display = std::make_shared<ImageDisplay>("Post " + name());
        }

        // Call implementations initialize
        if (!initialize_impl())
        {
            LOG(LogLevel::ERROR, name(), " - Failed to initialize");
            return false;
        }

        LOG(LogLevel::DEBUG, name(), " - Initialized");

        return true;
    }

    /**
     * Update algorithm with a new image
     */
    virtual bool update(const Image& input, Image& output)
    {
        TRACE();

        // time and call implementations update
        m_image_count++;
        const auto start = TIME_NOW();
        if (!update_impl(input.data(), output.data()))
        {
            LOG(LogLevel::ERROR, name(), " - Failed updating (Image #", m_image_count, ")");
            return false;
        }
        const auto end = TIME_NOW();
        m_duration_ns += DURATION_NS(end - start);
        LOG(LogLevel::TRACE, name(), " - Updated (Image #", m_image_count, ", total duration: ", duration(), " sec)");

        // update display if necessary
        if (m_display)
        {
            m_display->update_image(output);
        }

        return true;
    }

    /**
     * Return total time spent in update_impl() in seconds.
     */
    double duration() const { return m_duration_ns / 1e9; }

protected:

    BaseImageAlgorithm() = default;

    /**
     * Getters for sizes
     */
    std::size_t width() const { return m_width; }
    std::size_t height() const { return m_height; }
    std::size_t area() const { return m_area; }
    std::size_t size_bytes() const { return area() * sizeof(pixel64_t); }

private:

    virtual bool initialize_impl() { return true; }

    virtual bool update_impl(const pixel64_t* input, pixel64_t* output) = 0;

    std::size_t m_width;
    std::size_t m_height;
    std::size_t m_area;
    std::size_t m_image_count;

    double m_duration_ns;

    ImageDisplay_ptr m_display;
};

typedef std::shared_ptr<BaseImageAlgorithm> BaseImageAlgorithm_ptr;
typedef std::vector<BaseImageAlgorithm_ptr> BaseImageAlgorithm_vec;
typedef std::vector<BaseImageAlgorithm_vec> BaseImageAlgorithm_vecs;

#endif /** BASEIMAGEALGORITHM_HPP_ */
