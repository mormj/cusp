#pragma once

#include <cusp/kernel.cuh>

namespace cusp
{
    template <typename T>
    class or_bitwise : public kernel
    {
    private:
        int _ninputs;
        void **_dev_ptr_array;
    public:
        or_bitwise(int ninputs);
        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, int ninputs, int grid_size, int block_size, size_t nitems, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}