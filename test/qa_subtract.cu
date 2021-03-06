#include <gtest/gtest.h>
#include <complex>
#include <cusp/subtract.cuh>
#include <cusp/helper_cuda.h>

using namespace cusp;

template <typename T> 
void run_test(int N, int num_inputs)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)i;
      expected_output_data[i] = i * (2 - num_inputs);
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::subtract<T> op(num_inputs);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    checkCudaErrors(op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N));
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<float>>(int N, int num_inputs)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i), float(i));
      float real = (2 - num_inputs) * host_input_data[i].real();
      float imag = (2 - num_inputs) * host_input_data[i].imag();
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
  
    cusp::subtract<std::complex<float>> op(num_inputs);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    checkCudaErrors(op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N));
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<double>>(int N, int num_inputs)
{
    std::vector<std::complex<double>> host_input_data(N);
    std::vector<std::complex<double>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = (std::complex<double>)(double(i), double(i));
      double real = (2 - num_inputs) * host_input_data[i].real();
      double imag = (2 - num_inputs) * host_input_data[i].imag();
      std::complex<double> temp(real, imag);
      expected_output_data[i] = temp;
    }
    std::vector<std::complex<double>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<double>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<double>));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  
    cusp::subtract<std::complex<double>> op(num_inputs);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    checkCudaErrors(op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N));
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

TEST(SubtractKernel, Basic) {
  int N = 1024 * 100;

  run_test<int16_t>(N, 3);
  run_test<float>(N, 4);
  run_test<double>(N, 3);
  run_test<std::complex<float>>(N, 3);
  run_test<std::complex<double>>(N, 3);
}