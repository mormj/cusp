#include <gtest/gtest.h>
#include <complex>
#include <random>
#include "../include/cusp/nlog10.cuh"

using namespace cusp;

TEST(Nlog10, Basic)
{
    int N = 1024*100;
    float n = 2;
    float k = 3;
    std::vector<float> input_data(N);
    std::vector<float> output_data(N);
    std::vector<float> expected_data(N);
    
    for (int i=0; i<N; i++){
        input_data[i] = (float)(rand() % 100);
        expected_data[i] = n * log10(input_data[i]) + k;
    }

    launch_kernel_nlog10<float>((float *)input_data.data(), (float *)output_data.data(),
        n, k, 1024, N / 1024, N);

    EXPECT_EQ(expected_data, output_data);
}