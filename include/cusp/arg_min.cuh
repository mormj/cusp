#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class arg_min : public kernel
    {
    private:
        int _ninputs;
    public:
        arg_min(int ninputs = 1);

        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, int ninputs, int grid_size, int block_size,
            size_t nitems, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}