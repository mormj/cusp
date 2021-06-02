#pragma once

namespace cusp
{
template <typename T>
void launch_kernel_multiply_const(float f, const T *in, T *out, int grid_size, int block_size,
                        int N, cudaStream_t stream = 0);
}