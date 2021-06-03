#include <gtest/gtest.h>
#include <complex>
#include <cusp/add_const.cuh>

using namespace cusp;

TEST(AddConstKernel, Basic)
{

    std::vector<int> input_data = {0,1,5,8,12};
    std::vector<int> expectedResult = {2,3,7,10,14};
    std::vector<int> output_data(5);
    int x = 2;

    int N = 1024*100;
    //std::vector<std::complex<float>> input_data(N);
    //std::vector<std::complex<float>> output_data(N);
    launch_kernel_add_const<int>(x, (int *)input_data.data(), (int *)output_data.data(),
            1024, N / 1024, N);

    EXPECT_EQ(expectedResult, output_data);

    
}