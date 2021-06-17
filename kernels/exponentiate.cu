#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../include/cusp/exponentiate.cuh"

namespace cusp {

template <typename T>
__global__ void kernel_exponentiate(const T *in, T *out, float e, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = (T)(1);
    for (int j = 0; j < int(e); j++) {
      out[i] *= in[i];
    }
  }
}

template <>
__global__ void kernel_exponentiate<float>(const float *in, float *out, float e, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = pow(in[i], e);
  }
}

template <> __global__ void kernel_exponentiate<thrust::complex<float>>(
                                      const thrust::complex<float> *in,
                                      thrust::complex<float> *out, float e,
                                      int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = thrust::pow(in[i], e);
    // float theta = atan2(in[i].imag(), in[i].real());
    // float mag = sqrtf(powf(in[i].real(), 2) + powf(in[i].imag(), 2));
    // out[i] = thrust::complex<float>(powf(mag, e) * cos(theta * e),
    //                                 powf(mag, e) * sin(theta * e));
    //out[i].x = powf(mag, e) * cos(theta * e); 
    //out[i].y = powf(mag, e) * sin(theta * e);
  }
}

template <typename T>
cudaError_t exponentiate<T>::launch(const T *in, T *out, float e, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_exponentiate<<<grid_size, block_size, 0, stream>>>(in, out, e, N);
  } else {
    kernel_exponentiate<<<grid_size, block_size>>>(in, out, e, N);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t exponentiate<std::complex<float>>::launch(const std::complex<float> *in,
                                                      std::complex<float> *out, float e, 
                                                      int N, int grid_size, int block_size,
                                                      cudaStream_t stream) {
    if (stream) {
      kernel_exponentiate<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                                (thrust::complex<float> *)out, e, N);
    } else {
      kernel_exponentiate<<<grid_size, block_size>>>((const thrust::complex<float> *)in,
                                                     (thrust::complex<float> *) out, e, N);
    }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t exponentiate<T>::launch(const std::vector<const void *> inputs,
                                 const std::vector<void *> outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _e, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t exponentiate<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_exponentiate<T>, 0, 0);
}

template <>
cudaError_t exponentiate<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_exponentiate<thrust::complex<float>>,
                                            0, 0);
}

#define IMPLEMENT_KERNEL(T) template class exponentiate<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp