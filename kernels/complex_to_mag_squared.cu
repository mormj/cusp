#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <cusp/complex_to_mag_squared.cuh>

namespace cusp {

template <typename T> __global__ void kernel_mag_squared(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float mag_squared = in[i].real() * in[i].real() + in[i].imag() * in[i].imag();
    out[i] = thrust::complex<float>(mag_squared, 0);
  }
}


template <typename T>
cudaError_t complex_to_mag_squared<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_mag_squared<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                     (thrust::complex<float> *)out, N);
  } else {
    kernel_mag_squared<<<grid_size, block_size>>>((const thrust::complex<float> *)in, 
                                          (thrust::complex<float> *)out, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t complex_to_mag_squared<T>::launch(const std::vector<const void *> inputs,
                  const std::vector<void *> outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t complex_to_mag_squared<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_mag_squared<thrust::complex<float>>,
                                            0, 0);
}

#define IMPLEMENT_KERNEL(T) template class complex_to_mag_squared<T>;

IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp
