#include <gtest/gtest.h>
#include <complex>
#include <cusp/add.cuh>

using namespace cusp;

template <typename T> 
void run_add_test(int N, T num_inputs)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      expected_output_data[i] = num_inputs * i;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    std::cout << "ptr: " <<  dev_input_data << std::endl;

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    int ncopies = N * sizeof(std::complex<float>) / sizeof(T);
    cusp::add<T> op(num_inputs);

    int minGrid, blockSize, gridSize;
    op.occupancy(&blockSize, &minGrid);
    gridSize = (ncopies + blockSize - 1) / blockSize;
    op.set_block_and_grid(blockSize, gridSize);
    op.launch({dev_input_data}, {dev_output_data}, ncopies);
    /*
    int minGrid, minBlock;
    op.occupancy(&minBlock, &minGrid);
    op.set_block_and_grid(minGrid, N / minGrid);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    op.launch(input_data_pointer_vec, {dev_output_data}, N);
    */
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(Add, Basic) {
  int N = 1024 * 100;

  run_add_test<int16_t>(N, 3);
  // run_add_test<float>(N, 4);
}