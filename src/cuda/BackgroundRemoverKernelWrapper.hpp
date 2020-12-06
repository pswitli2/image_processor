#ifndef BACKGROUNDREMOVERKERNELWRAPPER_HPP_
#define BACKGROUNDREMOVERKERNELWRAPPER_HPP_

#include <memory>

#include "KernelWrapper.hpp"

class BackgroundRemoverKernelWrapper: public KernelWrapper
{
public:

    BackgroundRemoverKernelWrapper(std::size_t width, std::size_t height,
                                   std::size_t history_size, std::size_t tolerance);

    ~BackgroundRemoverKernelWrapper() override;

    void execute_impl();

private:

    bool m_history_full;

    std::size_t m_idx;
    const std::size_t m_history_size;
    const std::size_t m_tolerance;

    pixel64_t* m_history;
    pixel64_t* m_history_mean;

};

typedef std::shared_ptr<BackgroundRemoverKernelWrapper> BackgroundRemoverKernelWrapper_ptr;

#endif /** BACKGROUNDREMOVERKERNELWRAPPER_HPP_ */
