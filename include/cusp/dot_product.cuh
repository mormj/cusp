#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class dot_product : public kernel
    {
    private:
        size_t _stride = 1;

    public:
        // Stride of 2 on the input allows for interleaved complex data to be 
        // filtered with real taps
        dot_product(size_t stride = 1);

        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, size_t stride, int grid_size, int block_size, size_t nitems,
            cudaStream_t stream = 0);

        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;

        virtual cudaError_t occupancy(int *minBlock, int *minGrid);

        void decimate(std::vector<T>& outputs, const int gridSize);
    };

}