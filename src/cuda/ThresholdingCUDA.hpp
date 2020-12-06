#ifndef THRESHOLDINGCUDA_HPP_
#define THRESHOLDINGCUDA_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

#include "ThresholdingKernelWrapper.hpp"

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

        double tolerance = 1.0;
        if (!ConfigFile::get_param(THRESHOLD_TOLERANCE_PARAM_NAME, tolerance))
            return false;

        m_kernel = std::make_shared<ThresholdingKernelWrapper>(width(), height(), tolerance);

        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output)
    {
        m_kernel->execute(input, output);

        return true;
    }


protected:

private:

    ThresholdingKernelWrapper_ptr m_kernel;
};


#endif /** THRESHOLDINGCUDA_HPP_ */
