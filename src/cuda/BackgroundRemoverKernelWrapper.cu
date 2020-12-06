#include "BackgroundRemoverKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "CudaUtils.hpp"

BackgroundRemoverKernelWrapper::BackgroundRemoverKernelWrapper(std::size_t width, std::size_t height, std::size_t history_size, std::size_t tolerance)
: KernelWrapper(width, height), m_history_size(history_size), m_tolerance(tolerance)
{

}

void BackgroundRemoverKernelWrapper::execute_impl()
{

}
