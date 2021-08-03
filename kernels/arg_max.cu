#include <cusp/helper_cuda.h>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/arg_max.cuh>
#include <limits>

#define default_min_block 256
#define default_min_grid 32

namespace cusp {

template <typename T>
__global__ void kernel_arg_max(const T* ins, T* out,
                               int numeric_min, int stream_number,
                               int ninputs, int grid_size, int N)
{
    __shared__ thrust::complex<T> cache[default_min_block];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    T temp = numeric_min;
    while (i < N) {
    	if(ins[i] > temp)
    		temp = ins[i];
        i += blockDim.x * gridDim.x;  
    }
   
    // real component is value, imaginary component is index
    cache[cacheIndex] = thrust::complex<T>(temp, blockDim.x * blockIdx.x + threadIdx.x);

    __syncthreads();

    int ib = blockDim.x / 2;
    while (ib != 0) {
      if(cacheIndex < ib && cache[cacheIndex + ib].real() > cache[cacheIndex].real())
        cache[cacheIndex] = cache[cacheIndex + ib]; 

      __syncthreads();

      ib /= 2;
    }
    
    if(cacheIndex == 0) {
        // out[blockIdx.x] = cache[0];
        int index = blockIdx.x + stream_number * grid_size;
        out[index] = cache[0].real();
        out[index + ninputs * grid_size] = cache[0].imag();
    }
}


// first index of output vector is index where max value occurs,
// second index is input stream where max occurs
template <typename T>
__global__ void decimate_arg_max(T *out, int grid_size, int ninputs) {
    int maxIdx = 0;
    int maxStream = 0;
    int offset = ninputs * grid_size;
    for (int stream_number = 0; stream_number < ninputs; stream_number++) {
        for (int block_index = 0; block_index < grid_size; block_index++) {
            int index = stream_number * grid_size + block_index;
            if (out[index] > out[maxIdx]) {
                maxIdx = index;
                maxStream = stream_number;
            }
        }
    }
    out[0] = out[maxIdx + offset];
    out[1] = maxStream;
}


template <typename T> arg_max<T>::arg_max(int ninputs) : _ninputs(ninputs) {}

template <typename T>
cudaError_t arg_max<T>::launch(const std::vector<const void *> &inputs,
                                T *output, int ninputs, int grid_size, 
                                int block_size, size_t nitems,
                                cudaStream_t stream) {

    T numeric_min = std::numeric_limits<T>::min();

    if (stream) {
        for (int i = 0; i < ninputs; i++) {
            kernel_arg_max<<<grid_size, block_size, 0, stream>>>(
                (const T *)inputs[i],
                (T *)output, numeric_min,
                i, ninputs, grid_size, nitems
            );
        }
        cudaDeviceSynchronize();
        decimate_arg_max<<<1, 1, 0, stream>>>(output, grid_size, ninputs);
    }
    else {
        for (int i = 0; i < ninputs; i++) {
            kernel_arg_max<<<grid_size, block_size>>>(
                (const T *)inputs[i],
                (T *)output, numeric_min,
                i, ninputs, grid_size, nitems
            );
        }
        cudaDeviceSynchronize();
        decimate_arg_max<<<1, 1>>>(output, grid_size, ninputs);
    }
    return cudaPeekAtLastError();
}

template <typename T>
cudaError_t arg_max<T>::launch(const std::vector<const void *> &inputs,
                                const std::vector<void *> &outputs,
                                size_t nitems) {
    return launch(inputs, (T *)outputs[0], _ninputs, _grid_size,
        _block_size, nitems, _stream);
}

template <typename T> cudaError_t arg_max<T>::occupancy(int *minBlock, int *minGrid) {
    *minBlock = default_min_block;
    *minGrid = default_min_grid;
    return cudaPeekAtLastError();
}

#define IMPLEMENT_KERNEL(T) template class arg_max<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(double)

} // namespace cusp