#ifndef BACKGROUNDREMOVERKERNELWRAPPER_HPP_
#define BACKGROUNDREMOVERKERNELWRAPPER_HPP_

#include <memory>

#include "KernelWrapper.hpp"

class BackgroundRemoverKernelWrapper: public KernelWrapper
{
public:

    BackgroundRemoverKernelWrapper(std::size_t width, std::size_t height,
                                   std::size_t history_size, std::size_t tolerance);

    void execute_impl();

private:

    const std::size_t m_history_size;
    const std::size_t m_tolerance;
};

typedef std::shared_ptr<BackgroundRemoverKernelWrapper> BackgroundRemoverKernelWrapper_ptr;

#endif /** BACKGROUNDREMOVERKERNELWRAPPER_HPP_ */
