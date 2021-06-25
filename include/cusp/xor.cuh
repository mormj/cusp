#pragma once

#include <cusp/kernel.cuh>

namespace cusp
{
    template <typename T>
    class xor_bitwise : public kernel
    {
    private:
        T _ninputs;
        void **_dev_ptr_array;
    public:
        xor_bitwise(T ninputs);
        cudaError_t launch(const std::vector<const void *> inputs,
            T* output, int ninputs, int grid_size, int block_size, size_t nitems, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *> inputs,
            const std::vector<void *> outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}