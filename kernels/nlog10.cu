#include "../include/cusp/nlog10.cuh"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_nlog10(const T *in, T *out, T n, T k, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = n * (T)log10(float(in[i])) + k;
  }
}

template <typename T>
cudaError_t nlog10<T>::launch(const T *in, T *out, T n, T k, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_nlog10<<<grid_size, block_size, 0, stream>>>(in, out, n, k, N);
  } else {
    kernel_nlog10<<<grid_size, block_size>>>(in, out, n, k, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t nlog10<T>::launch(const std::vector<const void *> inputs,
                                 const std::vector<void *> outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _n, _k, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t nlog10<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_nlog10<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class nlog10<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)

} // namespace cusp