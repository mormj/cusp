#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/subtract.cuh>
#include <cusp/helper_cuda.h>

namespace cusp {

template <typename T>
__global__ void kernel_subtract(const T **ins, T *out, int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T *in = (T *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (T*)(*(ins+j));
      out[i] -= in[i]; //(*(in + j))[i];
    }
  }
}

template <>
__global__ void kernel_subtract<thrust::complex<float>>(const thrust::complex<float> **ins,
                                                   thrust::complex<float> *out,
                                                   int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    thrust::complex<float> *in = (thrust::complex<float> *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (thrust::complex<float>*)(*(ins+j));
      out[i] -= in[i]; //(*(in + j))[i];
    }
  }
}

template <>
__global__ void kernel_subtract<thrust::complex<double>>(const thrust::complex<double> **ins,
                                                   thrust::complex<double> *out,
                                                   int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    thrust::complex<double> *in = (thrust::complex<double> *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (thrust::complex<double>*)(*(ins+j));
      out[i] -= in[i]; //(*(in + j))[i];
    }
  }
}

template <typename T> subtract<T>::subtract(int ninputs) : _ninputs(ninputs) {
  checkCudaErrors(cudaMalloc(&_dev_ptr_array, sizeof(void *) * _ninputs));
}

template <typename T>
cudaError_t subtract<T>::launch(const std::vector<const void *>& inputs, T *output,
                           int ninputs, int grid_size, int block_size,
                           size_t nitems, cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_subtract<<<grid_size, block_size, 0, stream>>>((const T **)_dev_ptr_array,
                                                     output, ninputs, nitems);
  } else {
    kernel_subtract<<<grid_size, block_size>>>((const T **)_dev_ptr_array, output,
                                          ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t subtract<std::complex<float>>::launch(const std::vector<const void *>& inputs,
                           std::complex<float> *output,
                           int ninputs, int grid_size, int block_size,
                           size_t nitems, cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_subtract<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> **)_dev_ptr_array,
                                                     (thrust::complex<float> *)output, ninputs, nitems);
  } else {
    kernel_subtract<<<grid_size, block_size>>>((const thrust::complex<float> **)_dev_ptr_array,
                                          (thrust::complex<float> *) output,
                                          ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t subtract<std::complex<double>>::launch(const std::vector<const void *>& inputs,
                           std::complex<double> *output,
                           int ninputs, int grid_size, int block_size,
                           size_t nitems, cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_subtract<<<grid_size, block_size, 0, stream>>>((const thrust::complex<double> **)_dev_ptr_array,
                                                     (thrust::complex<double> *)output, ninputs, nitems);
  } else {
    kernel_subtract<<<grid_size, block_size>>>((const thrust::complex<double> **)_dev_ptr_array,
                                          (thrust::complex<double> *) output,
                                          ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t subtract<T>::launch(const std::vector<const void *>& inputs,
                           const std::vector<void *>& outputs, size_t nitems) {
  return launch(inputs, (T *)outputs[0], _ninputs, _grid_size, _block_size,
                nitems, _stream);
}

template <typename T>
cudaError_t subtract<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_subtract<T>, 0,
                                            0);
}

template <>
cudaError_t subtract<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_subtract<thrust::complex<float>>,
                                            0, 0);
}

template <>
cudaError_t subtract<std::complex<double>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_subtract<thrust::complex<double>>,
                                            0, 0);
}

#define IMPLEMENT_KERNEL(T) template class subtract<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)
IMPLEMENT_KERNEL(double)
IMPLEMENT_KERNEL(std::complex<double>)

} // namespace cusp