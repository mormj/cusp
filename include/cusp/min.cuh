#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class min : public kernel
    {
    private:
        int _ninputs;
        bool _multi_output;
    public:
        min(int ninputs = 1, bool multi_output = false);

        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, int ninputs, bool multi_output, int grid_size,
            int block_size, size_t nitems, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}