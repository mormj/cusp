#include <gtest/gtest.h>
#include <complex>
#include <cusp/fft_shift.cuh>

using namespace cusp;


template <typename T> 
void run_test(int N)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      expected_output_data[i] = (T)i;
    }
    int mid = (N-1)/2; // mid index
    for (int i = 0; i < N; i++) {
        if ((N%2) == 0) { // if even number of elements
            if (i < mid + 1) {
                expected_output_data[i] = host_input_data[i + mid + 1];
                expected_output_data[i + mid + 1] = host_input_data[i];
            }
        }
        else { // if odd number of elements
            if (i < mid) {
                expected_output_data[i] = host_input_data[i + mid + 1];
                expected_output_data[i + mid] = host_input_data[i];
            }
            if (i == mid) {
                expected_output_data[N-1] = host_input_data[i];
            }
        }
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::fft_shift<T> op;
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);
    op.launch({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(FFTShiftKernel, Basic) {
  int N = 1024 * 100;

  run_test<uint64_t>(N);
  //run_test<float>(N, 456.0);
}