#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cusp/not.cuh>

namespace cusp {

template <typename T> __global__ void kernel_not(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = ~in[i];
  }
}


template <typename T>
cudaError_t not_bitwise<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_not<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_not<<<grid_size, block_size>>>(in, out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t not_bitwise<T>::launch(const std::vector<const void *>& inputs,
                  const std::vector<void *>& outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t not_bitwise<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_not<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class not_bitwise<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)

} // namespace cusp
