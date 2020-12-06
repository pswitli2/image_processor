#ifndef THRESHOLDINGCPU_HPP_
#define THRESHOLDINGCPU_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

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

    bool update_impl(const image_data_t& input, image_data_t& output) override
    {
        TRACE();

        pixel_t max = 0;
        double sum = 0.0;
        for (std::size_t i = 0; i < area(); i++)
        {
            const auto& p = input[i];
            if (p > max)
                max = p;
            sum += (double) p;
        }
        const double mean = sum / (double) area();
        std::cout << "MEAN CPU:    " << sum << "  " << mean << std::endl;

        double stddev = 0.0;
        for (std::size_t i = 0; i < area(); i++)
        {
            const auto p = input[i];
            stddev += std::pow(std::fabs((double) p - mean), 2.0);
        }
        stddev = sqrt(stddev / (double) area());
        // std::cout << "STDDEV CPU:  " << stddev << std::endl;

        const double threshold = mean + (stddev * m_tolerance);

        LOG(LogLevel::TRACE, "ThresholdingCPU update info: mean = ", mean,
            ", max = ", max, ", stddev = ", stddev, ", threshold = ", threshold);

        for (size_t i = 0; i < input.size(); i++)
        {
            if ((double) input[i] >= threshold)
                output[i] = max;
        }

        return true;
    }

protected:

private:

    double m_tolerance;
};


#endif /** THRESHOLDINGCPU_HPP_ */