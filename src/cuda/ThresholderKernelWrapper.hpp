#ifndef THRESHOLDERKERNELWRAPPER_HPP_
#define THRESHOLDERKERNELWRAPPER_HPP_

#include <memory>

#include "KernelWrapper.hpp"

class ThresholderKernelWrapper: public KernelWrapper
{
public:

    ThresholderKernelWrapper(const std::size_t& width, const std::size_t& height, const double tolerance)
    : KernelWrapper(width, height), m_tolerance(tolerance) { }

    void execute_impl();

private:

    void sum_image(const pixel64_t* d_input, pixel64_t* d_col, pixel64_t* sum);

    double m_tolerance;
};

typedef std::shared_ptr<ThresholderKernelWrapper> ThresholderKernelWrapper_ptr;

#endif /** THRESHOLDERKERNELWRAPPER_HPP_ */