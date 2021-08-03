#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

// In current gnuradio, keep 1 in n saves last element in each window,
// but there are suggestions that keeping first element per window is
// more intuitive, so that will be my implementation.


namespace cusp
{
    template <typename T>
    class keep_one_in_n : public kernel
    {
    private:
        int _window;
    public:
        keep_one_in_n(int window) : _window(window) {};
        cudaError_t launch(const T *in, T *out, int window, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}