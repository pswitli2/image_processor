#ifndef BACKGROUNDREMOVERCUDA_HPP_
#define BACKGROUNDREMOVERCUDA_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

#include "BackgroundRemoverKernelWrapper.hpp"

class BackgroundRemoverCUDA: public BaseImageAlgorithm
{
public:

    BackgroundRemoverCUDA() = default;

    ~BackgroundRemoverCUDA() override = default;

    std::string name() const override
    {
        return "BackgroundRemoverCUDA";
    }

    bool initialize_impl() override
    {
        TRACE();

        std::size_t history_len = 0;
        std::size_t tolerance = 0;
        if (!ConfigFile::get_param(BACKGROUND_HISTORY_LENGTH_PARAM_NAME, history_len))
            return false;

        if (!ConfigFile::get_param(BACKGROUND_TOLERANCE_PARAM_NAME, tolerance))
            return false;

        m_kernel = std::make_shared<BackgroundRemoverKernelWrapper>(width(), height(), history_len, tolerance);

        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output)
    {
        m_kernel->execute(input, output);

        return true;
    }


protected:

private:

    BackgroundRemoverKernelWrapper_ptr m_kernel;
};


#endif /** BACKGROUNDREMOVERCUDA_HPP_ */
