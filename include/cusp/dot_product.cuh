#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class dot_product : public kernel
    {
    private:
        void **_dev_ptr_array;
    public:
        dot_product();

        cudaError_t launch(const std::vector<const void *>& inputs,
            T* output, int grid_size, int block_size, size_t nitems,
            cudaStream_t stream = 0);

        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;

        virtual cudaError_t occupancy(int *minBlock, int *minGrid);

        void decimate(std::vector<T>& outputs, const int gridSize);
    };

}