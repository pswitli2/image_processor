#ifndef LONEPIXELREMOVERCUDA_HPP_
#define LONEPIXELREMOVERCUDA_HPP_

#include "BaseImageAlgorithm.hpp"
#include "ConfigFile.hpp"

#include "LonePixelRemoverKernelWrapper.hpp"

/**
 * See LonePixelRemoverCPU docstring for algorithm description. This implemtation
 * performs the same calculations using CUDA.
 */
class LonePixelRemoverCUDA: public BaseImageAlgorithm
{
public:

    LonePixelRemoverCUDA() = default;

    ~LonePixelRemoverCUDA() override = default;

    std::string name() const override { return "LonePixelRemoverCUDA"; }

    bool initialize_impl() override
    {
        TRACE();

        std::size_t num_adjacent = 0;
        if (!ConfigFile::get_param(LONE_PIXEL_NUM_ADJACENT_PARAM_NAME, num_adjacent))
            return false;

        m_kernel = std::make_shared<LonePixelRemoverKernelWrapper>(width(), height(), num_adjacent);

        return true;
    }

    bool update_impl(const pixel64_t* input, pixel64_t* output)
    {
        m_kernel->execute(input, output);

        return true;
    }


protected:

private:

    LonePixelRemoverKernelWrapper_ptr m_kernel;
};


#endif /** LONEPIXELREMOVERCUDA_HPP_ */
