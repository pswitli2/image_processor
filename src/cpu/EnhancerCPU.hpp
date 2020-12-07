#ifndef ENHANCERCPU_HPP_
#define ENHANCERCPU_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

/**
 * The Enhancer brightens every pixel in the image. This is mainly used for being
 * able to see an almost "raw" input image better.
 *
 * Required Parameters:
 *     ENHANCEMENT_FACTOR = pixel_val = pixel_val ^ ENHANCEMENT_FACTOR
 */
class EnhancerCPU: public BaseImageAlgorithm
{
public:

    EnhancerCPU() = default;

    ~EnhancerCPU() override = default;

    std::string name() const override { return "EnhancerCPU"; }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(ENHANCEMENT_FACTOR_PARAM_NAME, m_factor))
            return false;
        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output) override
    {
        TRACE();

        // enhance each pixel
        for (std::size_t i = 0; i < area(); i++)
        {
            output[i] = (pixel64_t)(std::pow((double) input[i], m_factor));
        }
        return true;
    }

protected:

private:

    double m_factor;
};


#endif /** ENHANCERCPU_HPP_ */
