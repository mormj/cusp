#pragma once

namespace cusp
{
template <typename T>
void launch_kernel_copy(const T *in, T *out, int grid_size, int block_size,
                        int N, cudaStream_t stream = 0);
}