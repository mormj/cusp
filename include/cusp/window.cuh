#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{
    template <typename T>
    class window : public kernel
    {
    private:
        float * _window;
        int _window_length;
    public:
        window(float * window, int window_length) : _window(window), _window_length(window_length) {};
        cudaError_t launch(const T *in, T *out, float * window, int window_length, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };

}