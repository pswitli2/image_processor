#ifndef ENHANCERCPU_HPP_
#define ENHANCERCPU_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

class EnhancerCPU: public BaseImageAlgorithm
{
public:

    EnhancerCPU() = default;

    ~EnhancerCPU() override = default;

    std::string name() const override
    {
        return "EnhancerCPU";
    }

    bool initialize_impl() override
    {
        TRACE();

        if (!ConfigFile::get_param(ENHANCMENT_FACTOR_PARAM_NAME, m_factor))
            return false;
        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output) override
    {
        TRACE();

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
