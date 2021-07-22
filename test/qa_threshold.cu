#include <gtest/gtest.h>
#include <complex>
#include <cusp/threshold.cuh>
#include <cmath>

using namespace cusp;


template <typename T> 
void run_test(int N, T lower, T upper)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;

      if (host_input_data[i] > upper) {
          expected_output_data[i] = (T)1;
      } else if (host_input_data[i] < lower) {
          expected_output_data[i] = (T)0;
      } else expected_output_data[i] = host_input_data[i];
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::threshold<T> op(lower, upper);
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(ThresholdKernel, Basic) {
  int N = 1024 * 100;

  run_test<int32_t>(N, 1024 * 10, 1024 * 90);
  run_test<int64_t>(N, 1024 * 20, 1024 * 80);
  run_test<float>(N, 1024 * 10.0, 1024 * 90.0);
  run_test<double>(N, 1024 * 20.0, 1024 * 80.0);
}