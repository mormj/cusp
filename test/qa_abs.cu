#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "../include/cusp/abs.cuh"

using namespace cusp;

TEST(AbsKernel, Basic)
{
    int N = 1024*100;
    std::vector<int> input_data(N);
    std::vector<int> expectedResult(N);
    std::vector<int> output_data(N);

    for (int i = 0; i < N; i++) {
        input_data[i] = -1 * rand() % 100;
        expectedResult[i] = -1 * input_data[i];
    }

    launch_kernel_abs<int>((int *)input_data.data(), (int *)output_data.data(),
        1024, N / 1024, N);

    EXPECT_EQ(expectedResult, output_data);
}