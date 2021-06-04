#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cusp
{

template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock);

template <typename T>
void launch_kernel_nlog10(const T *in, T *out, float n, float k, int grid_size,
                          int block_size, int N, cudaStream_t stream = 0);
}