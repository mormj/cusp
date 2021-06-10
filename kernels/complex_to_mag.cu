#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cusp/complex_to_mag.cuh>

namespace cusp {

template <typename T> __global__ void kernel_mag(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i].x = sqrtf(powf(in[i].x, 2) + powf(in[i].y, 2));
    out[i].y = 0;
  }
}


template <typename T>
cudaError_t complex_to_mag<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_mag<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_mag<<<grid_size, block_size>>>(in, out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t complex_to_mag<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t complex_to_mag<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_mag<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class complex_to_mag<T>;

IMPLEMENT_KERNEL(cuFloatComplex)

} // namespace cusp
