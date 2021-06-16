#include "../include/cusp/multiply_const.cuh"
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_multiply_const(const T *in, T *out, T k, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i] * k;
  }
}

template <>
__global__ void kernel_multiply_const(const cuFloatComplex* in,
                                      cuFloatComplex* out,
                                      cuFloatComplex k,
                                      int N)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i].x = k.x * in[i].x - k.y * in[i].y;
    out[i].y = k.x * in[i].y + k.y * in[i].x;
  }
}

template <typename T>
cudaError_t multiply_const<T>::launch(const T *in, T *out, T k, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_multiply_const<<<grid_size, block_size, 0, stream>>>(in, out, k, N);
  } else {
    kernel_multiply_const<<<grid_size, block_size>>>(in, out, k, N);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t multiply_const<std::complex<float>>::launch(const std::complex<float> *in, std::complex<float> *out, std::complex<float> k, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_multiply_const<cuFloatComplex><<<grid_size, block_size, 0, stream>>>((const cuFloatComplex *)in, (cuFloatComplex *)out, make_cuFloatComplex(k.real(), k.imag()), N);
  } else {
    kernel_multiply_const<cuFloatComplex><<<grid_size, block_size>>>((const cuFloatComplex *)in, (cuFloatComplex *)out, make_cuFloatComplex(k.real(), k.imag()), N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t multiply_const<T>::launch(const std::vector<const void *> inputs,
                                 const std::vector<void *> outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _k, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t multiply_const<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_multiply_const<T>, 0, 0);
}

template <>
cudaError_t multiply_const<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_multiply_const<cuFloatComplex>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class multiply_const<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
//IMPLEMENT_KERNEL(cuFloatComplex)
IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp