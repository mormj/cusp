#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <cusp/helper_cuda.h>
#include <cusp/dot_product.cuh>
#include <iostream>
#include <stdio.h>

#define default_min_block 256
#define default_min_grid 32

namespace cusp {

// Code is based on "cuda by example: an introduction to general purpose gpu programming."
// I would assume this needs to be licensed / cited but I'm not certain how.
template <typename T>
__global__ void kernel_dot_product(const T *in1, const T *in2, T *out, size_t stride, int N) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ T cache[default_min_block];
  T temp = (T)0;

  while (i < N) {
    temp += in1[stride*i] * in2[i];
    i += blockDim.x * gridDim.x;
  }

  __syncthreads();
  cache[threadIdx.x] = temp;
  int j = blockDim.x / 2;
  
  while (j != 0) {
    if (threadIdx.x < j) cache[threadIdx.x] += cache[j + threadIdx.x];
    __syncthreads();
    j /= 2;
  }

  if (threadIdx.x == 0) out[blockIdx.x] = cache[0];
}

template <>
__global__ void kernel_dot_product<thrust::complex<float>>(
  const thrust::complex<float> *in1, const thrust::complex<float> *in2,
  thrust::complex<float> *out, size_t stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ thrust::complex<float> cache[default_min_block];
    thrust::complex<float> temp(0, 0);

    while (i < N) {
      // temp += in1[i] * in2[i];
      temp += in1[stride*i] *  thrust::complex<float>(in2[i].real(), -1.0 * in2[i].imag());
      i += blockDim.x * gridDim.x;
    }

    __syncthreads();
    cache[threadIdx.x] = temp;
    int j = blockDim.x / 2;
    
    while (j != 0) {
      if (threadIdx.x < j) cache[threadIdx.x] += cache[j + threadIdx.x];
      __syncthreads();
      j /= 2;
    }

    if (threadIdx.x == 0) out[blockIdx.x] = cache[0];
}


template <>
__global__ void kernel_dot_product<thrust::complex<double>>(
  const thrust::complex<double> *in1, const thrust::complex<double> *in2,
  thrust::complex<double> *out, size_t stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ thrust::complex<double> cache[default_min_block];
    thrust::complex<double> temp(0, 0);

    while (i < N) {
      // temp += in1[i] * in2[i];
      temp += in1[stride*i] *  thrust::complex<double>(in2[i].real(), -1.0 * in2[i].imag());
      i += blockDim.x * gridDim.x;
    }

    __syncthreads();
    cache[threadIdx.x] = temp;
    int j = blockDim.x / 2;
    
    while (j != 0) {
      if (threadIdx.x < j) cache[threadIdx.x] += cache[j + threadIdx.x];
      __syncthreads();
      j /= 2;
    }

    if (threadIdx.x == 0) out[blockIdx.x] = cache[0];
}


// This is a really inneficient way of performing decimation,
// but unfortunately I'm not sure of a better way. A kernel
// implementation allows us to calculate the numeric value of
// our dot product before copying memory back to the host,
// whereas a cpu function requires memory to have already 
// been copied back to the host, which can be annoying when
// working with gnuradio blocks.
template <typename T>
__global__ void kernel_decimate(T * outputs, const int gridSize) {
  for (int i = 1; i < gridSize; i++) {
    outputs[0] += outputs[i];
    outputs[i] = 0;
  }
}

template <>
__global__ void kernel_decimate<thrust::complex<float>>(
  thrust::complex<float> * outputs, const int gridSize) {
  for (int i = 1; i < gridSize; i++) {
    outputs[0] += outputs[i];
    outputs[i] = thrust::complex<float>(0.0, 0.0);
  }
}

template <>
__global__ void kernel_decimate<thrust::complex<double>>(
  thrust::complex<double> * outputs, const int gridSize) {
  for (int i = 1; i < gridSize; i++) {
    outputs[0] += outputs[i];
    outputs[i] = thrust::complex<double>(0.0, 0.0);
  }
}

template <typename T> dot_product<T>::dot_product(size_t stride) : _stride(stride) {

}

template <typename T>
cudaError_t dot_product<T>::launch(const std::vector<const void *> &inputs,
                                   T *output, size_t stride, int grid_size, int block_size,
                                   size_t nitems, cudaStream_t stream) {

    if (stream) {
      kernel_dot_product<<<grid_size, block_size, 0, stream>>>(
          (const T *)inputs[0],
          (const T *)inputs[1],
          (T *)output, stride, nitems);
      kernel_decimate<<<1, 1, 0, stream>>>(output, grid_size);
    } else {
      kernel_dot_product<<<grid_size, block_size>>>(
          (const T *)inputs[0],
          (const T *)inputs[1],
          (T *)output, stride, nitems);
      kernel_decimate<<<1, 1>>>(output, grid_size);
    }
    return cudaPeekAtLastError();
}

template <>
cudaError_t dot_product<std::complex<float>>::launch(const std::vector<const void *> &inputs,
                                   std::complex<float> *output, size_t stride, int grid_size, int block_size,
                                   size_t nitems, cudaStream_t stream) {

    if (stream) {
      kernel_dot_product<<<grid_size, block_size, 0, stream>>>(
          (const thrust::complex<float> *)inputs[0],
          (const thrust::complex<float> *)inputs[1],
          (thrust::complex<float> *)output, stride, nitems);
      kernel_decimate<<<1, 1, 0, stream>>>(
          (thrust::complex<float> *)output, grid_size);
    } else {
      kernel_dot_product<<<grid_size, block_size>>>(
          (const thrust::complex<float> *)inputs[0],
          (const thrust::complex<float> *)inputs[1],
          (thrust::complex<float> *)output, stride, nitems);
      kernel_decimate<<<1, 1>>>(
          (thrust::complex<float> *)output, grid_size);
    }
    return cudaPeekAtLastError();
}


template <>
cudaError_t dot_product<std::complex<double>>::launch(const std::vector<const void *> &inputs,
                                   std::complex<double> *output, size_t stride, int grid_size, int block_size,
                                   size_t nitems, cudaStream_t stream) {

    if (stream) {
      kernel_dot_product<<<grid_size, block_size, 0, stream>>>(
          (const thrust::complex<double> *)inputs[0],
          (const thrust::complex<double> *)inputs[1],
          (thrust::complex<double> *)output, stride, nitems);
      kernel_decimate<<<1, 1, 0, stream>>>(
          (thrust::complex<double> *)output, grid_size);
    } else {
      kernel_dot_product<<<grid_size, block_size>>>(
          (const thrust::complex<double> *)inputs[0],
          (const thrust::complex<double> *)inputs[1],
          (thrust::complex<double> *)output, stride, nitems);
      kernel_decimate<<<1, 1>>>(
          (thrust::complex<double> *)output, grid_size);
    }
    return cudaPeekAtLastError();
}


template <typename T>
cudaError_t dot_product<T>::launch(const std::vector<const void *> &inputs,
                                   const std::vector<void *> &outputs,
                                   size_t nitems) {
  return launch(inputs, (T *)outputs[0], _stride, _grid_size, _block_size,
                nitems, _stream);
}

template <typename T> cudaError_t dot_product<T>::occupancy(int *minBlock, int *minGrid) {
  *minBlock = default_min_block;
  *minGrid = default_min_grid;
  return cudaPeekAtLastError();
}

#define IMPLEMENT_KERNEL(T) template class dot_product<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(std::complex<float>);
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<double>);
IMPLEMENT_KERNEL(double)

} // namespace cusp
