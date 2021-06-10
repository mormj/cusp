#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cusp/absolute_value.cuh>

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
    out[i].x = sqrtf(powf(in[i].x, 2) + powf(in[i].y, 2));
    out[i].y = 0;
  }
}

template <typename T>
cudaError_t absolute_value<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_abs<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_abs<<<grid_size, block_size>>>(in, out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t absolute_value<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t absolute_value<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_abs<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class absolute_value<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(short)
IMPLEMENT_KERNEL(int)
IMPLEMENT_KERNEL(long)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(cuFloatComplex)

} // namespace cusp
