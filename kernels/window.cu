#include <cusp/window.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_window(const T *in, T *out, float * window, int window_length, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i] * window[i%window_length];
  }
}

template <typename T>
cudaError_t window<T>::launch(const T *in, T *out, float * window, int window_length, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_window<<<grid_size, block_size, 0, stream>>>(in, out, window, window_length, N);
  } else {
    kernel_window<<<grid_size, block_size>>>(in, out, window, window_length, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t window<T>::launch(const std::vector<const void *>& inputs,
                                 const std::vector<void *>& outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _window, _window_length, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t window<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_window<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class window<T>;


IMPLEMENT_KERNEL(float)

} // namespace cusp