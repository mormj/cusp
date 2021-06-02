#include <cusp/add_const.cuh>
#include <gtest/gtest.h>
#include <complex>
#include <cusp/cusp.cuh>

using namespace cusp;

TEST(AddConstKernel, Basic)
{
    int N = 1024*100;
    std::vector<std::complex<float>> input_data(N);
    std::vector<std::complex<float>> output_data(N);
    launch_kernel_copy<uint64_t>((uint64_t *)input_data.data(), (uint64_t *)output_data.data(),
        1024, N / 1024, N);

    EXPECT_EQ(input_data, output_data);
}