#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cusp
{

// is there a way to autogenerate both functions with modtool 
// since every single kernel will have both?
template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock);

template <typename T>
void launch_kernel_abs(const T *in, T *out, int grid_size, int block_size,
                        int N, cudaStream_t stream = 0);
}