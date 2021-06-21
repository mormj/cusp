#include <gtest/gtest.h>
#include <complex>
#include "../include/cusp/exponentiate.cuh"
#include <cmath>
#include <cuComplex.h>

using namespace cusp;

template <typename T> 
void run_test(int N, float e)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = T(i);
      expected_output_data[i] = pow(host_input_data[i], (T)e);
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::exponentiate<T> op(e);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < (int)expected_output_data.size(); i++) {
      if (expected_output_data[i] != host_output_data[i]) {
        std::cout << "Expected: " << expected_output_data[i] << std::endl;
        std::cout << "Actual: " << host_output_data[i] << std::endl;
      }
    }
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<float>(int N, float e)
{
    std::vector<float> host_input_data(N);
    std::vector<float> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = float(i);
      expected_output_data[i] = pow(host_input_data[i], e);
    }
    std::vector<float> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(float));
    cudaMalloc(&dev_output_data, N * sizeof(float));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(float), cudaMemcpyHostToDevice);
  
    cusp::exponentiate<float> op(e);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < (int)expected_output_data.size(); i++) {
      EXPECT_NEAR(expected_output_data[i],
                  host_output_data[i],
                  expected_output_data[i] / 10000);
    }
  
    //EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<float>>(int N, float e)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i), float(i));
      expected_output_data[i] = pow(host_input_data[i], e);
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::exponentiate<std::complex<float>> op(e);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    //EXPECT_EQ(expected_output_data, host_output_data);
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


TEST(ExponentiateKernel, Basic) {
  int N = 1024 * 100;
  float e = 1.5;

  run_test<int>(N, e);
  run_test<float>(N, e);
  run_test<std::complex<float>>(N, e);
}