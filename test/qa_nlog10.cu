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
      host_input_data[i] = i + 1;
      expected_output_data[i] = (T)n * (T)log10(float(i)) + (T)k;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::nlog10<T> op(n, k);
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
void run_test<float>(int N, float n, float k)
{
    std::vector<float> host_input_data(N);
    std::vector<float> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = i + 1;
      expected_output_data[i] = n * (float)log10(host_input_data[i]) + k;
    }
    std::vector<float> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(float));
    cudaMalloc(&dev_output_data, N * sizeof(float));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(float), cudaMemcpyHostToDevice);
  
    cusp::nlog10<float> op(n, k);
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
}


TEST(Nlog10Kernel, Basic) {
  int N = 100;
  float n = 2.0;
  float k = 2.0;

  run_test<float>(N, n, k);
  //run_test<int>(N, n, k);
}