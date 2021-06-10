#include <gtest/gtest.h>
#include <cusp/nlog10.cuh>
#include <cmath>

using namespace cusp;


template <typename T> 
void run_test(int N, float n, float k)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = i;
      expected_output_data[i] = (T)(n * log10(float(i) + k)));
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::nlog10<T> op(n, k);
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(ComplexToMagKernel, Basic) {
  int N = 1024 * 100;
  float n = 2.0f;
  float k = 3.0f;

  run_test<float>(N, n, k);
  run_test<int>(N, n, k);
}
