#pragma once

#include <cusp/kernel.cuh>

namespace cusp
{

template <typename T>
class nlog10 : public kernel
{
private:
    T _n;
    T _k;
public:
    nlog10(T n, T k) : _n(n), _k(k) {};
    cudaError_t launch(const T *in, T *out, T n, T k, int grid_size, int block_size,
        int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *>& inputs,
        const std::vector<void *>& outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}