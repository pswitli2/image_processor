#ifndef BACKGROUNDREMOVERCPU_HPP_
#define BACKGROUNDREMOVERCPU_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

/**
 * The background remover attempts to remove thhe background from an image to focus on
 * foreground detections. It does this by saving a history, and checking if a pixel value hasnt
 * changed, if it hasnt, it clears the filter.
 *
 * Required Parameters:
 *     HISTORY_SIZE = length of image history to save
 *     HISTORY_THRESHOLD = how close a pixel has to be from the history mean to be filtered out
 */
class BackgroundRemoverCPU: public BaseImageAlgorithm
{
public:

    BackgroundRemoverCPU()
    : m_initialized(false) { }

    ~BackgroundRemoverCPU() override
    {
        if (m_initialized)
        {
            // delete m_history and m_history_mean
            delete []m_history;
            delete m_history_mean;
        }
    }

    std::string name() const override { return "BackgroundRemoverCPU"; }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(BACKGROUND_HISTORY_LENGTH_PARAM_NAME, m_history_len))
            return false;

        if (!ConfigFile::get_param(BACKGROUND_TOLERANCE_PARAM_NAME, m_tolerance))
            return false;

        // allocate m_history and m_history_mean
        m_idx = 0;
        m_history_full = false;
        m_history = new pixel64_t*[m_history_len];
        for (std::size_t i = 0; i < m_history_len; i++)
        {
            m_history[i] = new pixel64_t[area()];
        }

        m_history_mean = new pixel64_t[area()];
        m_initialized = true;
        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output) override
    {
        TRACE();

        memcpy(output, input, size_bytes());

        // if the history is full, filter pixels
        if (m_history_full)
        {
            // calculate history mean
            memset(m_history_mean, 0, size_bytes());
            for (std::size_t i = 0; i < area(); i++)
            {
                for (std::size_t j = 0; j < m_history_len; j++)
                {
                    m_history_mean[i] += m_history[j][i];
                }
            }

            // filter out background pixels
            for (std::size_t i = 0; i < area(); i++)
            {
                m_history_mean[i] /= m_history_len;
                const auto mean = (int64_t) m_history_mean[i];
                const auto p = (int64_t) output[i];
                if ((pixel64_t) std::abs(mean - p) < m_tolerance)
                    output[i] = 0;
            }
        }

        // save to history
        memcpy(m_history[m_idx], input, size_bytes());

        // set index for next update
        m_idx = (m_idx + 1) % m_history_len;
        if (m_idx == 0)
            m_history_full = true;
        
        return true;
    }

protected:

private:

    bool m_initialized;
    bool m_history_full;

    std::size_t m_idx;
    std::size_t m_history_len;
    std::size_t m_tolerance;

    pixel64_t** m_history;
    pixel64_t* m_history_mean;

};


#endif /** BACKGROUNDREMOVERCPU_HPP_ */
