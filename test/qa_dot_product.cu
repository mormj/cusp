#include <gtest/gtest.h>
#include <complex>
#include <cusp/dot_product.cuh>

using namespace cusp;

template <typename T> 
void run_test(int N)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)(i);
      T out = host_input_data[i];
 
      expected_output_data[0] += out * out;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::dot_product<T> op;

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_data;

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);

    EXPECT_EQ(expected_output_data[0], host_output_data[0]);
}

template <> 
void run_test<float>(int N)
{
    std::vector<float> host_input_data(N);
    std::vector<float> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = (float)(i);
      float out = host_input_data[i];
 
      expected_output_data[0] += out * out;
    }
    std::vector<float> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(float));
    cudaMalloc(&dev_output_data, N * sizeof(float));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(float), cudaMemcpyHostToDevice);
  
    cusp::dot_product<float> op;

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_data;

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(float), cudaMemcpyDeviceToHost);

    float error = abs(expected_output_data[0]) / 100000;
    EXPECT_NEAR(expected_output_data[0], host_output_data[0],
                error);
}

template <> 
void run_test<double>(int N)
{
    std::vector<double> host_input_data(N);
    std::vector<double> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = (double)(i);
      double out = host_input_data[i];
 
      expected_output_data[0] += out * out;
    }
    std::vector<double> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(double));
    cudaMalloc(&dev_output_data, N * sizeof(double));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(double), cudaMemcpyHostToDevice);
  
    cusp::dot_product<double> op;

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_data;

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(double), cudaMemcpyDeviceToHost);

    double error = abs(expected_output_data[0]) / 100000;
    EXPECT_NEAR(expected_output_data[0], host_output_data[0],
                error);
}


template <> 
void run_test<std::complex<float>>(int N)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(i, i);
      std::complex<float> out = host_input_data[i];
 
      expected_output_data[0] += out * std::complex<float>(host_input_data[i].real(),
                                                           -1 * host_input_data[i].imag());
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::dot_product<std::complex<float>> op;

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_data;


    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    float real_error = abs(expected_output_data[0].real()) / 100000;
    float imag_error = abs(expected_output_data[0].imag()) / 100000;

    EXPECT_NEAR(expected_output_data[0].real(), host_output_data[0].real(),
                real_error);
    EXPECT_NEAR(expected_output_data[0].imag(), host_output_data[0].imag(),
                imag_error);
}

template <> 
void run_test<std::complex<double>>(int N)
{
    std::vector<std::complex<double>> host_input_data(N);
    std::vector<std::complex<double>> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<double>(i, i);
      std::complex<double> out = host_input_data[i];
 
      expected_output_data[0] += out * std::complex<double>(host_input_data[i].real(),
                                                           -1 * host_input_data[i].imag());
    }
    std::vector<std::complex<double>> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<double>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<double>));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);
  
    cusp::dot_product<std::complex<double>> op;

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_data;

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    double real_error = abs(expected_output_data[0].real()) / 100000;
    double imag_error = abs(expected_output_data[0].imag()) / 100000;

    EXPECT_NEAR(expected_output_data[0].real(), host_output_data[0].real(),
                real_error);
    EXPECT_NEAR(expected_output_data[0].imag(), host_output_data[0].imag(),
                imag_error);
}


void run_test_strided(int N)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<float> host_input_taps(N);
    std::vector<std::complex<float>> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(i, -i);
      host_input_taps[i] = 1.0;
      std::complex<float> out = std::complex<float>(real(host_input_data[i])*host_input_taps[i],imag(host_input_data[i])*host_input_taps[i]) ;
 
      expected_output_data[0] += out;
    }

    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_input_taps;
    void *dev_output_data_re;  
    void *dev_output_data_im;

    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_input_taps, N * sizeof(float));
    cudaMalloc(&dev_output_data_re, N * sizeof(float));
    cudaMalloc(&dev_output_data_im, N * sizeof(float));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_input_taps, host_input_taps.data(),
               N * sizeof(float), cudaMemcpyHostToDevice);

    cusp::dot_product<float> op(2);

    std::vector<const void *> input_data_pointer_vec(2);
    input_data_pointer_vec[0] = dev_input_data;
    input_data_pointer_vec[1] = dev_input_taps;

    op.launch_default_occupancy(input_data_pointer_vec, {dev_output_data_re}, N);
    input_data_pointer_vec[0] = (float *)dev_input_data + 1;
    op.launch_default_occupancy(input_data_pointer_vec, {dev_output_data_im}, N);
 
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data_re,
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((float *)(host_output_data.data())+1, dev_output_data_im,
               sizeof(float), cudaMemcpyDeviceToHost);

    float error = abs(expected_output_data[0]) / 100000;
    EXPECT_NEAR(expected_output_data[0].real(), host_output_data[0].real(),
                error);
    EXPECT_NEAR(expected_output_data[0].imag(), host_output_data[0].imag(),
                error);
}

TEST(DotProductKernel, Basic) {
  int N = 10;

  run_test<int32_t>(N);
  run_test<int64_t>(N);
  run_test<float>(N);
  run_test<double>(N);
  run_test<std::complex<float>>(N);
  run_test<std::complex<double>>(N);

  run_test_strided(N);
} 