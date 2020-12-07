#include "BackgroundRemoverKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "Kernels.hpp"

BackgroundRemoverKernelWrapper::BackgroundRemoverKernelWrapper(std::size_t width, std::size_t height, std::size_t history_size, std::size_t tolerance)
: KernelWrapper(width, height), m_history_size(history_size), m_tolerance(tolerance)
{
    CUDA_MALLOC((void**) &m_history, num_bytes() * m_history_size);
    CUDA_MALLOC((void**) &m_history_mean, num_bytes());

    m_idx = 0;
    m_history_full = false;
}

BackgroundRemoverKernelWrapper::~BackgroundRemoverKernelWrapper()
{
    CUDA_FREE(&m_history);
    CUDA_FREE(&m_history_mean);
}

void BackgroundRemoverKernelWrapper::execute_impl()
{
    // copy input to output
    __copy_image<<<width(), height(), 1, m_stream>>>(m_d_input, m_d_output);

    if (m_history_full)
    {
        // if full sum the history and place in m_history_mean
        __sum_history<<<width(), height(), 1, m_stream>>>(m_history, m_history_mean, m_history_size, area());

        // preform background removal
        __remove_background<<<width(), height(), 1, m_stream>>>(m_history_mean, m_d_output, m_history_size, m_tolerance);
    }

    // copy image to history buffer
    __copy_image<<<width(), height(), 1, m_stream>>>(m_d_input, m_history + (area() * m_idx));

    // update history index for next update
    m_idx = (m_idx + 1) % m_history_size;
    if (m_idx == 0)
        m_history_full = true;
}
