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

        pixel64_t max = 0;
        pixel64_t sum = 0;
        for (std::size_t i = 0; i < area(); i++)
        {
            const auto& p = input[i];
            if (p > max)
                max = p;
            sum += p;
        }
        const auto mean = sum / (pixel64_t) area();

        sum = 0.0;
        for (std::size_t i = 0; i < area(); i++)
        {
            const auto p = (long long) input[i];
            const auto mean_long = (long long) mean;
            sum += (p - mean_long) * (p - mean_long);
        }
        const auto stddev = sqrt(sum / (pixel64_t) area());

        const auto threshold = mean + (pixel64_t) ((double) stddev * m_tolerance);

        LOG(LogLevel::TRACE, "ThresholdingCPU update info: mean = ", mean,
            ", max = ", max, ", stddev = ", stddev, ", threshold = ", threshold);

        for (size_t i = 0; i < input.size(); i++)
        {
            if (input[i] >= threshold)
                output[i] = max;
        }

        return true;
    }

protected:

private:

    double m_tolerance;
};


#endif /** THRESHOLDINGCPU_HPP_ */