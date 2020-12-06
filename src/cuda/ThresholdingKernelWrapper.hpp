#ifndef THRESHOLDINGKERNELWRAPPER_HPP_
#define THRESHOLDINGKERNELWRAPPER_HPP_

#include <memory>

#include "CudaUtils.hpp"
#include "KernelWrapper.hpp"

class ThresholdingKernelWrapper: public KernelWrapper
{
public:

    ThresholdingKernelWrapper(const std::size_t& width, const std::size_t& height, const double tolerance)
    : KernelWrapper(width, height), m_tolerance(tolerance) { }

    void execute_impl();

private:

    void sum_image(const pixel64_t* d_input, pixel64_t* d_col, pixel64_t* sum);

    double m_tolerance;
};

typedef std::shared_ptr<ThresholdingKernelWrapper> ThresholdingKernelWrapper_ptr;

#endif /** THRESHOLDINGKERNELWRAPPER_HPP_ */