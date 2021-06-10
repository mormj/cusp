#include <cusp/exponentiate.cuh>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_exponentiate(const T *in, T *out, float e, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = (T)(powf(float(in[i]), e));
  }
}

template <> __global__ void kernel_exponentiate<cuFloatComplex>(const cuFloatComplex *in,
                                                       cuFloatComplex *out, float e, int N
                                                      )
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    float theta = atan2(in[i].y, in[i].x);
    float mag = sqrtf(powf(in[i].x, 2) + powf(in[i].y, 2));
    out[i].x = powf(mag, e) * cos(theta * e); 
    out[i].y = powf(mag, e) * sin(theta * e);
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

#define IMPLEMENT_KERNEL(T) template class exponentiate<T>;

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