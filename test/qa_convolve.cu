#include <cmath>
#include <complex>
#include <cusp/convolve.cuh>
#include <gtest/gtest.h>

#include "helper_cuda.h"

using namespace cusp;

template <typename T> void run_test(int N, const std::vector<T> &taps) {
  std::vector<T> host_input_data(N);
  std::vector<T> expected_output_data(N);
  for (int i = 0; i < N; i++) {
    host_input_data[i] = T(i);
  }
  std::vector<T> host_output_data(N);

  void *dev_input_data;
  // void *dev_taps;
  void *dev_output_data;

  checkCudaErrors(cudaMalloc(&dev_input_data, N * sizeof(T)));
  // checkCudaErrors(cudaMalloc(&dev_taps, taps.size() * sizeof(T)));
  checkCudaErrors(cudaMalloc(&dev_output_data, N * sizeof(T)));

  checkCudaErrors(cudaMemcpy(dev_input_data, host_input_data.data(),
                             N * sizeof(T), cudaMemcpyHostToDevice));

  // checkCudaErrors(cudaMemcpy(dev_taps, taps.data(), taps.size() * sizeof(T),
  //                            cudaMemcpyHostToDevice));

  cusp::convolve<T, T> op(taps);
  int minGrid, blockSize, gridSize;
  op.occupancy(&blockSize, &minGrid);
  gridSize = (N + blockSize - 1) / blockSize;
  op.set_block_and_grid(blockSize, gridSize);
  checkCudaErrors(op.launch({dev_input_data}, {dev_output_data}, N));

  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(host_output_data.data(), dev_output_data,
                             N * sizeof(T), cudaMemcpyDeviceToHost));

  // EXPECT_EQ(expected_output_data, host_output_data);

  for (auto &x : host_output_data) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
}

TEST(ConvolveKernel, Basic) {
  int N = 1024 * 1;

  std::vector<float> ftaps{1,1,1};
  run_test<float>(N, ftaps);
}