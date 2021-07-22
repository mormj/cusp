#include <cusp/helper_cuda.h>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/max.cuh>

#define default_min_block 256
#define default_min_grid 32

namespace cusp {

// Kernel identifies the maximum value within each block. In a seperate kernel,
// we go through output and either record the absolute max value or identify
// the maximum value from each input stream

template <typename T>
__global__ void kernel_max(const T* ins, T* out, int stream_number, int grid_size, int N)
{
    __shared__ T cache[default_min_block];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    T temp = ins[i];
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
        // out[blockIdx.x] = cache[0];
        out[blockIdx.x + stream_number * grid_size] = cache[0];
    }
}


// Internal functions that cleans output stream of extraneous data.
// These kernels might need cleaning up eventually to make better use
// of parallelism. I'm thinking you launch the kernels with the normal
// bounds of grid_size and block_size, but you only look at kernels if 
// blockDim.x * blockIdx.x + threadIdx.x < grid_size * ninputs. However, 
// grid_size * ninputs is usually really small, so I'm not certain if
// there's really any notable benefits in trying to parallelize these
// kernels?

template <typename T>
__global__ void kernel_get_max_single(T * out, int grid_size, int ninputs) {
    T max = out[0];
    for (int i = 0; i < ninputs * grid_size; i++) {
        if (max < out[i]) max = out[i];
        out[i] = (T)0;
    }
    out[0] = max;
}

template <typename T>
__global__ void kernel_get_max_multiple(T * out, int grid_size, int ninputs) {
    for (int stream_number = 0; stream_number < ninputs; stream_number++) {
        T max = (T)0;
        for (int block_index = 0; block_index < grid_size; block_index++) {
            if (out[stream_number * grid_size + block_index] > max) {
                max = out[stream_number * grid_size + block_index];
            }
            out[stream_number * grid_size + block_index] = (T)0;
        }
        out[stream_number] = max;
    }
}


// design two kernels, one with vlen = 1
// and one with vlen = len

template <typename T> max<T>::max(int ninputs, bool multi_output) : _ninputs(ninputs),
    _multi_output(multi_output) {}

template <typename T>
cudaError_t max<T>::launch(const std::vector<const void *> &inputs,
                                T *output, int ninputs, bool multi_output,
                                int grid_size, int block_size, size_t nitems,
                                cudaStream_t stream) {

    if (stream) {
        for (int i = 0; i < ninputs; i++) {
            kernel_max<<<grid_size, block_size, 0, stream>>>(
                (const T *)inputs[i],
                (T *)output, i, grid_size, nitems
            );
        }
        if (multi_output) {
            kernel_get_max_multiple<<<1, 1, 0, stream>>>(
                (T *) output, grid_size, ninputs
            );
        } else {
            kernel_get_max_single<<<1, 1, 0, stream>>>(
                (T *)output, grid_size, ninputs
            );
        }
    }
    else {
        for (int i = 0; i < ninputs; i++) {
            kernel_max<<<grid_size, block_size>>>(
                (const T *)inputs[i],
                (T *)output, i, grid_size, nitems
            );
        }
        if (multi_output) {
            kernel_get_max_multiple<<<1, 1>>>(
                (T *) output, grid_size, ninputs
            );
        } else {
            kernel_get_max_single<<<1, 1>>>(
                (T *)output, grid_size, ninputs
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