#pragma once

#include "kernel.cuh"

namespace cusp
{
// for now, try implemnting this such that 'in' is a vector of input arrays.
// if this doesn't work, hardcode an implementation that accepts 2 params
// and let the block handle the rest.
template <typename T>
void launch_kernel_add(const T **in, T *out, int grid_size,
                       int block_size, int num_inputs, int N,
                       cudaStream_t stream = 0);
}