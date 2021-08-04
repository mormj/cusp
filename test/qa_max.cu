#include <gtest/gtest.h>
#include <complex>
#include <cusp/max.cuh>
#include <algorithm>
#include <random>

using namespace cusp;

template <typename T> 
void run_test(int N, int num_inputs, bool multiple_outputs)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)(rand() % N);
    }

    expected_output_data[0] = *std::max_element(
        host_input_data.begin(), host_input_data.end());
    
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::max<T> op(num_inputs, multiple_outputs);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);

    if (multiple_outputs) {
      for (int i = 0; i < num_inputs; i++) {
        EXPECT_EQ(expected_output_data[0], host_output_data[i]);
      }
    }
    else {
      EXPECT_EQ(expected_output_data[0], host_output_data[0]);
    }
}

TEST(MaxKernel, Basic) {
  int N = 1024 * 100;

  run_test<int16_t>(N, 1, true);
  run_test<int32_t>(N, 2, false);
  run_test<float>(N, 3, true);
  run_test<double>(N, 4, false);
}