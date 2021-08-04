#include <cusp/helper_cuda.h>
#include <complex>
#include <limits>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/max.cuh>

#define default_min_block 256
#define default_min_grid 32

namespace cusp {

template <typename T>
__global__ void kernel_max(const T* ins, T* out, T numeric_min,
    int stream_number, int grid_size, int N)
{
    __shared__ T cache[default_min_block];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    T temp = numeric_min;
    while (i < N) {
    	if(ins[i] > temp)
    		temp = ins[i];
        i += blockDim.x * gridDim.x;  
    }
   
    cache[cacheIndex] = temp;

    __syncthreads();

    int ib = blockDim.x / 2;
    while (ib != 0) {
      if(cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex])
        cache[cacheIndex] = cache[cacheIndex + ib]; 

      __syncthreads();

      ib /= 2;
    }
    
    if(cacheIndex == 0) {
        out[blockIdx.x + stream_number * grid_size] = cache[0];
    }
}

template <typename T>
__global__ void decimate_max_single(T * out, int grid_size, int ninputs) {
    T max = out[0];
    for (int i = 0; i < ninputs * grid_size; i++) {
        if (max < out[i]) max = out[i];
    }
    out[0] = max;
}

template <typename T>
__global__ void decimate_max_multiple(T * out, int grid_size, int ninputs) {
    T max = out[0];
    for (int stream_number = 0; stream_number < ninputs; stream_number++) {
        for (int block_index = 0; block_index < grid_size; block_index++) {
            int index = stream_number * grid_size + block_index;
            if (out[index] > max) {
                max = out[index];
            }
        }
        out[stream_number] = max;
    }
}

template <typename T>
cudaError_t max<T>::launch(const std::vector<const void *> &inputs,
                                T *output, int ninputs, bool multi_output,
                                int grid_size, int block_size, size_t nitems,
                                cudaStream_t stream) {

    T numeric_min = std::numeric_limits<T>::min();

    if (stream) {
        for (int i = 0; i < ninputs; i++) {
            kernel_max<<<grid_size, block_size, 0, stream>>>(
                (const T *)inputs[i],
                (T *)output, numeric_min, i, grid_size, nitems
            );
        }
        cudaDeviceSynchronize();
        if (multi_output) {
            decimate_max_multiple<<<1, 1, 0, stream>>>(
                output, grid_size, ninputs
            );
        } else {
            decimate_max_single<<<1, 1, 0, stream>>>(
                output, grid_size, ninputs
            );
        }
    }
    else {
        for (int i = 0; i < ninputs; i++) {
            kernel_max<<<grid_size, block_size>>>(
                (const T *)inputs[i],
                (T *)output, numeric_min, i, grid_size, nitems
            );
        }
        cudaDeviceSynchronize();
        if (multi_output) {
            decimate_max_multiple<<<1, 1>>>(
                output, grid_size, ninputs
            );
        } else {
            decimate_max_single<<<1, 1>>>(
                output, grid_size, ninputs
            );
        }
    }
    return cudaPeekAtLastError();
}

template <typename T>
cudaError_t max<T>::launch(const std::vector<const void *> &inputs,
                                const std::vector<void *> &outputs,
                                size_t nitems) {
    return launch(inputs, (T *)outputs[0], _ninputs, _multi_output,
        _grid_size, _block_size, nitems, _stream);
}

template <typename T> cudaError_t max<T>::occupancy(int *minBlock, int *minGrid) {
    *minBlock = default_min_block;
    *minGrid = default_min_grid;
    return cudaPeekAtLastError();
}

#define IMPLEMENT_KERNEL(T) template class max<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(double)

} // namespace cusp