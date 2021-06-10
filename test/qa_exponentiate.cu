#include <gtest/gtest.h>
#include <complex>
#include <cusp/exponentiate.cuh>
#include <cmath>
#include <cuComplex.h>

using namespace cusp;


template <> 
void run_test<cuFloatComplex>(int N, float e)
{
    std::vector<cuFloatComplex> host_input_data(N);
    std::vector<cuFloatComplex> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = make_cuFloatComplex(float(i), float(2 * i));

      float theta = atan2(h_a[i].y, h_a[i].x);
      float mag = sqrtf(powf(h_a[i].x, 2) + powf(h_a[i].y, 2));

      float x = powf(mag, e) * cos(theta * e); 
      float y = powf(mag, e) * sin(theta * e);

      expected_output_data[i] = make_cuFloatComplex(x, y);
    }
    std::vector<cuFloatComplex> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(cuFloatComplex));
    cudaMalloc(&dev_output_data, N * sizeof(cuFloatComplex));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
  
    cusp::exponentiate<cuFloatComplex> op(e);
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
void run_test(int N, float e)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = i;
      expected_output_data[i] = (T)powf(float(i), e);
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::exponentiate<T> op(e);
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(ComplexToMagKernel, Basic) {
  int N = 1024 * 100;
  float e = 2.0f;

  run_test<cuFloatComplex>(N, e);
  run_test<float>(N, e);
}
