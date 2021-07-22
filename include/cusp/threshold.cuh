#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class threshold : public kernel
    {
    private:
        T _upper;
        T _lower;
    public:
        threshold(T lower, T upper) : _lower(lower), _upper(upper) {};
        cudaError_t launch(const T *in, T *out, T lower, T upper, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}