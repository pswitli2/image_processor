#include "LonePixelRemoverKernelWrapper.hpp"

#include <iostream>
#include <limits>

#include "Kernels.hpp"

void LonePixelRemoverKernelWrapper::execute_impl()
{
    __copy_image<<<width(), height(), 1, m_stream>>>(m_d_input, m_d_output);

    __lone_pixel<<<width() - 2, height() - 2, 1, m_stream>>>(m_d_input, m_d_output, m_num_adjacent, width());
}
