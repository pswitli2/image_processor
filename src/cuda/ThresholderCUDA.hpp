#ifndef THRESHOLDERCUDA_HPP_
#define THRESHOLDERCUDA_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

#include "ThresholderKernelWrapper.hpp"

/**
 * See ThresholderCPU docstring for algorithm description. This implemtation
 * performs the same calculations using CUDA.
 */
class ThresholderCUDA: public BaseImageAlgorithm
{
public:

    ThresholderCUDA() = default;

    ~ThresholderCUDA() override = default;

    std::string name() const override { return "ThresholderCUDA"; }

    bool initialize_impl() override
    {
        TRACE();

        double tolerance = 1.0;
        if (!ConfigFile::get_param(THRESHOLD_TOLERANCE_PARAM_NAME, tolerance))
            return false;

        m_kernel = std::make_shared<ThresholderKernelWrapper>(width(), height(), tolerance);

        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output)
    {
        m_kernel->execute(input, output);

        return true;
    }


protected:

private:

    ThresholderKernelWrapper_ptr m_kernel;
};


#endif /** THRESHOLDERCUDA_HPP_ */
