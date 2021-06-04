#include <complex>
#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/add.cuh>
#include <iostream>
#include "helper_cuda.h"

namespace cusp {

template <typename T>
__global__ void kernel_add(const T **ins, T *out, int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T *in = (T *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (T*)(*(ins+j));
      out[i] += in[i]; //(*(in + j))[i];
    }
  }
}

template <typename T> add<T>::add(T ninputs) : _ninputs(ninputs) {
  checkCudaErrors(cudaMalloc(&_dev_ptr_array, sizeof(void *) * _ninputs));
}

template <typename T>
cudaError_t add<T>::launch(const std::vector<const void *> inputs, T *output,
                           int ninputs, int grid_size, int block_size,
                           size_t nitems, cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_add<<<grid_size, block_size, 0, stream>>>((const T **)_dev_ptr_array,
                                                     output, ninputs, nitems);
  } else {
    kernel_add<<<grid_size, block_size>>>((const T **)_dev_ptr_array, output,
                                          ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t add<T>::launch(const std::vector<const void *> inputs,
                           const std::vector<void *> outputs, size_t nitems) {

  for (int n = 0; n < _ninputs; n++) {
    std::cout << inputs[n] << ", ";
  }
  std::cout << std::endl;
  return launch(inputs, (T *)outputs[0], _ninputs, _grid_size, _block_size,
                nitems, _stream);
}

template <typename T>
cudaError_t add<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_add<T>, 0,
                                            0);
}

#define IMPLEMENT_KERNEL(T) template class add<T>;

// IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
// IMPLEMENT_KERNEL(int32_t)
// IMPLEMENT_KERNEL(int64_t)
// IMPLEMENT_KERNEL(float)
// IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp