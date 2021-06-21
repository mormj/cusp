#include <gtest/gtest.h>
#include <complex>
#include "../include/cusp/add_const.cuh"

using namespace cusp;


template <typename T> 
void run_test(int N, T k)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      expected_output_data[i] = i + k;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::add_const<T> op(k);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<float>>(int N, std::complex<float> k)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (std::complex<float>)(float(i), float(i));
      float real = host_input_data[i].real() + k.real();
      float imag = host_input_data[i].imag() + k.imag();
      std::complex<float> temp(real, imag);
      expected_output_data[i] = temp;
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::add_const<std::complex<float>> op(k);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    for (int i = 0; i < (int)expected_output_data.size(); i++) {

      // Also add a test case to check for imaginary component

      EXPECT_NEAR(expected_output_data[i].real(),
                  host_output_data[i].real(),
                  abs(expected_output_data[i].real() / 10000));

      EXPECT_NEAR(expected_output_data[i].imag(),
                  host_output_data[i].imag(),
                  abs(expected_output_data[i].imag() / 10000));
    }
}


TEST(AddConstKernel, Basic) {
  int N = 1024 * 100;

  run_test<int16_t>(N, 123);
  run_test<float>(N, 456.0001);
  std::complex<float> param(2.0, 2.0);
  run_test<std::complex<float>>(N, param);
}