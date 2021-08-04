#include <gtest/gtest.h>
#include <complex>
#include <cusp/keep_one_in_n.cuh>
#include <cmath>

using namespace cusp;

template <typename T> 
void run_test(int N, int window)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      if (i % window == 0) {
        expected_output_data[ i / window ] = host_input_data[i];
      }
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::keep_one_in_n<T> op(window);
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<float>>(int N, int window)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i), float(i));
      if (i % window == 0) {
        expected_output_data[ i / window ] = host_input_data[i];
      }
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::keep_one_in_n<std::complex<float>> op(window);
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(KeepOneInNKernel, Basic) {
  int N = 1024 * 100;
  run_test<int32_t>(N, 5);
  run_test<float>(N, 15);
  run_test<std::complex<float>>(N, 30);
}