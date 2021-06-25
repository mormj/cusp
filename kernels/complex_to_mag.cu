#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cusp/complex_to_mag.cuh>

namespace cusp {

template <typename T> __global__ void kernel_mag(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float start = in[i].real() * in[i].real() + in[i].imag() * in[i].imag();
    float guess = sqrtf(start);

    if (guess == 0) {
      out[i] = thrust::complex<float>(guess, 0);
    }
    else {
      for (int t = 0; t < 15; t++) {
        guess = 0.5f * (guess + start / guess);
      }
      out[i] = thrust::complex<float>(guess, 0);
    }
  }
}


template <typename T>
cudaError_t complex_to_mag<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_mag<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                     (thrust::complex<float> *)out, N);
  } else {
    kernel_mag<<<grid_size, block_size>>>((const thrust::complex<float> *)in, 
                                          (thrust::complex<float> *)out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t complex_to_mag<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t complex_to_mag<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_mag<thrust::complex<float>>,
                                            0, 0);
}

#define IMPLEMENT_KERNEL(T) template class complex_to_mag<T>;

IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp
