#pragma once

#include <cusp/kernel.cuh>
<<<<<<< HEAD
#include <thrust/complex.h>
=======
#include <complex>
>>>>>>> 6cbad9bf6dbba7e689acf6b3d9b2692261ad9080

namespace cusp
{

class conjugate : public kernel
{
public:
    conjugate() = default;
    cudaError_t launch(const std::complex<float> *in, std::complex<float> *out, int grid_size, int block_size,
        int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *> inputs,
        const std::vector<void *> outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}