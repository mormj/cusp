#include <gtest/gtest.h>
#include <cusp/window.cuh>
#include <cmath>

using namespace cusp;


template <typename T> 
void run_test(int N, float * window, int window_length)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    //int nbatches = N/window_length;
    for (int i = 0; i < N; i++) {
      host_input_data[i] = i;
      expected_output_data[i] = host_input_data[i] * window[i%window_length];
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::window<T> op(window, window_length);
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
void run_test<float>(int N, float * window, int window_length)
{
    std::vector<float> host_input_data(N);
    std::vector<float> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = i;
      expected_output_data[i] = host_input_data[i] * window[i%window_length];
    }
    std::vector<float> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
    float * dev_window;
  
    cudaMalloc(&dev_input_data, N * sizeof(float));
    cudaMalloc(&dev_output_data, N * sizeof(float));
    cudaMalloc(&dev_window, N * sizeof(float));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_window, window, N * sizeof(float), cudaMemcpyHostToDevice);
  
    cusp::window<float> op(dev_window, window_length);
    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (N + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(float), cudaMemcpyDeviceToHost);

    

    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(WindowKernel, Basic) {
  int N = 8;
  float window[2] = {1,2};
  int window_length = sizeof(window)/sizeof(window[0]);

  run_test<float>(N, window, window_length);
}