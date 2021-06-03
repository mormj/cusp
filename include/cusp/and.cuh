#pragma once

#include "kernel.cuh"

namespace cusp
{
template <typename T>
void launch_kernel_and(const T **in, T *out, int grid_size,
                       int block_size, int N, cudaStream_t stream);
}