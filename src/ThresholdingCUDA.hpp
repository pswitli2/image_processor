#ifndef THRESHOLDINGCUDA_HPP_
#define THRESHOLDINGCUDA_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

#include "Kernels.h"

class ThresholdingCUDA: public BaseImageAlgorithm
{
public:

    ThresholdingCUDA() = default;

    ~ThresholdingCUDA() override = default;

    std::string name() const override
    {
        return "ThresholdingCUDA";
    }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(THRESHOLD_TOLERANCE_PARAM_NAME, m_tolerance))
            return false;

        return true;
    }

    bool update_impl(const image_data_t& input, image_data_t& output)
    {
        exec_kernel(input, output);

        return true;
    }


protected:

private:

    double m_tolerance;
};


#endif /** THRESHOLDINGCUDA_HPP_ */
