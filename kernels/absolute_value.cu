#include <cuComplex.h>
#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "../include/cusp/absolute_value.cuh"

namespace cusp {

template <typename T> __global__ void kernel_abs(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i] > -1 * in[i] ? (T)in[i] : (T)(-1 * in[i]);
  }
}

template <>
__global__ void kernel_abs<thrust::complex<float>>(const thrust::complex<float> *in,
                                                   thrust::complex<float> *out, int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float start = powf(in[i].real(), 2) + powf(in[i].imag(), 2);
    float guess = sqrtf(start);

    if (guess == 0) {
      out[i] = thrust::complex<float>(guess, 0);
    }
    else {
      for (int t = 0; t < 15; t++) {
        guess = 0.5f * (guess + start / guess);
      }
      out[i] = thrust::complex<float>(guess, 0);
      //out[i] = thrust::complex<float>(sqrtf(powf(in[i].real(), 2) + powf(in[i].imag(), 2)), 0);
    }
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

template <>
cudaError_t absolute_value<std::complex<float>>::launch(const std::complex<float> *in,
                                                        std::complex<float> *out, int N,
                                                        int grid_size, int block_size,
                                                        cudaStream_t stream) {
    if (stream) {
      kernel_abs<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                       (thrust::complex<float> *)out, N);
    } else {
      kernel_abs<<<grid_size, block_size>>>((const thrust::complex<float> *)in,
                                            (thrust::complex<float> *) out, N);
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

template <>
cudaError_t absolute_value<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_abs<thrust::complex<float>>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class absolute_value<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(std::complex<float>);
IMPLEMENT_KERNEL(float)
//IMPLEMENT_KERNEL(cuFloatComplex)

} // namespace cusp
