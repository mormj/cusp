#include <gtest/gtest.h>
#include <complex>
#include <cusp/multiply_const.cuh>
#include <cuComplex.h>

using namespace cusp;

template <> 
void run_test<cuFloatComplex>(int N, float k)
{
    std::vector<cuFloatComplex> host_input_data(N);
    std::vector<cuFloatComplex> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = cuFloatComplex(float(i), float(i));
      expected_output_data[i] = cuFloatComplex(float(i) * k, float(i) * k);
    }
    std::vector<cuFloatComplex> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(cuFloatComplex));
    cudaMalloc(&dev_output_data, N * sizeof(cuFloatComplex));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  
    cusp::multiply_const<cuFloatComplex> op(k);
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


template <typename T> 
void run_test(int N, T k)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      expected_output_data[i] = i * k;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::multiply_const<T> op(k);
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(MultiplyConstKernel, Basic) {
  int N = 1024 * 100;
  float e = 2.0f;

  run_test<int16_t>(N, 2);
  run_test<float>(N, 3.0);
  run_test<cuFloatComplex>(N, e);
}