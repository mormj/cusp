#pragma once

#include <cusp/kernel.cuh>

namespace cusp
{

template <typename T>
class nlog10 : public kernel
{
private:
    float _n;
    float _k;
public:
    nlog10(float n, float k) : _n(n), _k(k) {};
    cudaError_t launch(const T *in, T *out, float n, float k, int grid_size, int block_size,
        int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *> inputs,
        const std::vector<void *> outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}