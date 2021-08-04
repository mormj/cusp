#include <cusp/threshold.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_threshold(const T *in, T *out, T lower, T upper, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    if (in[i] > upper) out[i] = (T)1;
    else if (in[i] < lower) out[i] = (T)0;
    else out[i] = in[i];
  }
}

template <typename T>
cudaError_t threshold<T>::launch(const T *in, T *out, T lower, T upper, int N, int grid_size,
                                 int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_threshold<<<grid_size, block_size, 0, stream>>>(in, out, lower, upper, N);
  } else {
    kernel_threshold<<<grid_size, block_size>>>(in, out, lower, upper, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t threshold<T>::launch(const std::vector<const void *>& inputs,
                                 const std::vector<void *>& outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _lower, _upper, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t threshold<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_threshold<T>, 0, 0);
}


#define IMPLEMENT_KERNEL(T) template class threshold<T>;


IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(double)

} // namespace cusp