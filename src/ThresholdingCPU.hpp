#ifndef THRESHOLDINGCPU_HPP_
#define THRESHOLDINGCPU_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

constexpr const char THRESHOLD_TOLERANCE_PARAM_NAME[] = "THRESHOLD_TOLERANCE";

class ThresholdingCPU: public BaseImageAlgorithm
{
public:

    ThresholdingCPU() = default;

    ~ThresholdingCPU() override = default;

    std::string name() const override
    {
        return "ThresholdingCPU";
    }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(THRESHOLD_TOLERANCE_PARAM_NAME, m_tolerance))
            return false;

        return true;
    }

    bool update_impl(const png::image<pixel_t>& input, png::image<pixel_t>& output) override
    {
        TRACE();

        pixel_t max = 0;
        double mean = 0.0;
        for (std::size_t c = 0; c < height(); c++)
        {
            for (std::size_t r = 0; r < width(); r++)
            {
                const auto p = input.get_pixel(r, c);
                if (p > max)
                    max = p;
                mean += (double) p;
            }
        }
        mean = mean / (double) area();

        double stddev = 0.0;
        for (std::size_t c = 0; c < height(); c++)
        {
            for (std::size_t r = 0; r < width(); r++)
            {
                const auto p = input.get_pixel(r, c);
                stddev += std::pow(std::fabs((double) p - mean), 2.0);
            }
        }
        stddev = sqrt(stddev / (double) area());

        const auto threshold = mean + (stddev * m_tolerance);

        for (std::size_t c = 0; c < height(); c++)
        {
            for (std::size_t r = 0; r < width(); r++)
            {
                const pixel_t p = input.get_pixel(r, c);

                pixel_t new_val = 0;
                if ((double) p > threshold)
                    new_val = max;
                output.set_pixel(r, c, new_val);
            }
        }

        return true;
    }

protected:

private:

    double m_tolerance;
};


#endif /** THRESHOLDINGCPU_HPP_ */