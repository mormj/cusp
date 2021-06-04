#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "helper_cuda.h"
#include <cusp/fft_shift.cuh>

namespace cusp {

template <typename T> __global__ void kernel_fft_shift(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int mid = (N-1)/2; // mid index
  if ((N%2) == 0) { // if even number of elements
    if (i < mid + 1) {
        out[i] = in[i + mid + 1];
        out[i + mid + 1] = in[i];
    }
  }
  else { // if odd number of elements
    if (i < mid) {
        out[i] = in[i + mid + 1];
        out[i + mid] = in[i];
    }
    if (i == mid) {
        out[N-1] = in[i];
    }
  }
}

template <typename T>
cudaError_t fft_shift<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_fft_shift<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_fft_shift<<<grid_size, block_size>>>(in, out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t fft_shift<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t fft_shift<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_fft_shift<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class fft_shift<T>;

IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)

} // namespace cusp
