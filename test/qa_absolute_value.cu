#include <gtest/gtest.h>
#include <complex>
#include <cusp/absolute_value.cuh>
#include <cmath>
#include <cuComplex.h>

using namespace cusp;


template <typename T> 
void run_test(int N)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = T(-1 * i);
      expected_output_data[i] = (T)(i));
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::absolute_value<T> op();
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<cuFloatComplex>(int N)
{
    std::vector<cuFloatComplex> host_input_data(N);
    std::vector<cuFloatComplex> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = make_cuFloatComplex(float(i), float(i * 2));
      float mag = sqrtf(powf(in[i].x, 2) + powf(in[i].y, 2));
      expected_output_data[i] = make_cuFloatComplex(mag, 0.0f);
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::absolute_value<T> op();
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(AbsKernel, Basic) {
  int N = 1024 * 100;

  run_test<int>(N);
  run_test<float>(N);
  run_test<cuFloatComplex>(N);
}