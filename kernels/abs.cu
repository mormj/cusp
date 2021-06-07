#include <cuComplex.h>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include <cusp/abs.cuh>

namespace cusp {

template <typename T> __global__ void kernel_abs(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = (T)(fabsf(float(in[i])));
  }
}

template <> __global__ void kernel_abs<cuFloatComplex>(const cuFloatComplex *in,
                                                       cuFloatComplex *out, int N
                                                      )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i].x = fabsf(in[i].x);
    out[i].y = fabsf(in[i].y);
  }
}

template <typename T>
cudaError_t abs<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_abs<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_abs<<<grid_size, block_size>>>(in, out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t abs<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t abs<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_abs<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class abs<T>;

IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)
IMPLEMENT_KERNEL(int)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(cuFloatComplex)

} // namespace cusp