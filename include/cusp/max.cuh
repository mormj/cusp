#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class max : public kernel
    {
    private:
        int _ninputs;
        bool _multi_output;
    public:
        max(int ninputs = 1, bool multi_output = false) : _ninputs(ninputs),
            _multi_output(multi_output) {};

        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, int ninputs, bool multi_output, int grid_size,
            int block_size, size_t nitems, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}