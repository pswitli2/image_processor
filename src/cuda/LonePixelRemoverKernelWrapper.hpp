#ifndef LONEPIXELREMOVERKERNELWRAPPER_HPP_
#define LONEPIXELREMOVERKERNELWRAPPER_HPP_

#include <memory>

#include "KernelWrapper.hpp"

class LonePixelRemoverKernelWrapper: public KernelWrapper
{
public:

    LonePixelRemoverKernelWrapper(std::size_t width, std::size_t height, std::size_t num_adjacent)
    : KernelWrapper(width, height), m_num_adjacent(num_adjacent) { }

    void execute_impl();

private:

    const std::size_t m_num_adjacent;
};

typedef std::shared_ptr<LonePixelRemoverKernelWrapper> LonePixelRemoverKernelWrapper_ptr;

#endif /** LONEPIXELREMOVERKERNELWRAPPER_HPP_ */
