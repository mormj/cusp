#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T, typename T_TAPS>
    class convolve : public kernel
    {
    private:
        std::vector<T_TAPS> _taps;
        T *_dev_taps;
    public:
        convolve(const std::vector<T_TAPS>& taps );
        cudaError_t launch(const T *in, T *out, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *> inputs,
            const std::vector<void *> outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };
}