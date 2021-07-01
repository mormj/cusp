#include "helper_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/convolve.cuh>

namespace cusp {

extern "C" __global__ void __launch_bounds__(512)
    _cupy_convolve_float32(const float *__restrict__ inp, const int inpW,
                           const float *__restrict__ kernel, const int kerW,
                           const int mode, const bool swapped_inputs,
                           float *__restrict__ out, const int outW);

template <typename T, typename T_TAPS>
convolve<T, T_TAPS>::convolve(const std::vector<T_TAPS> &taps,
                              const convolve_mode_t mode)
    : _taps(taps), _mode(mode) {
  checkCudaErrors(cudaMalloc(&_dev_taps, taps.size() * sizeof(T)));
  checkCudaErrors(cudaMemcpy(_dev_taps, taps.data(), taps.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
};

template <typename T, typename T_TAPS>
cudaError_t convolve<T, T_TAPS>::launch(const T *in, T *out, int N,
                                        int grid_size, int block_size,
                                        cudaStream_t stream) {

  auto N_out = output_length(N);

  if (stream) {
    _cupy_convolve_float32<<<grid_size, block_size, 0, stream>>>(
        in, N, _dev_taps, _taps.size(), (int)_mode, false, out, N_out);
  } else {
    _cupy_convolve_float32<<<grid_size, block_size>>>(
        in, N, _dev_taps, _taps.size(), (int)_mode, false, out, N_out);
  }
  return cudaPeekAtLastError();
}

template <typename T, typename T_TAPS>
cudaError_t convolve<T, T_TAPS>::launch(const std::vector<const void *> inputs,
                                        const std::vector<void *> outputs,
                                        size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], nitems, _grid_size,
                _block_size, _stream);
}

template <typename T, typename T_TAPS>
cudaError_t convolve<T, T_TAPS>::occupancy(int *minBlock, int *minGrid) {
  auto rc = cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                               _cupy_convolve_float32, 0, 0);

  *minBlock =
      std::min(*minBlock, 512); // Convolve kernels are limited to 512 threads

  return rc;
}

template <typename T, typename T_TAPS>
int convolve<T, T_TAPS>::output_length(int input_length) {
  int N_out = input_length;
  if (_mode == convolve_mode_t::VALID) {
    N_out = input_length - _taps.size() + 1;
  } else if (_mode == convolve_mode_t::FULL) {
    N_out = input_length + _taps.size() - 1;
  }
  return N_out;
}

#define IMPLEMENT_KERNEL(T, T_TAPS) template class convolve<T, T_TAPS>;

// IMPLEMENT_KERNEL(int8_t)
// IMPLEMENT_KERNEL(int16_t)
// IMPLEMENT_KERNEL(int32_t)
// IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float, float)
// IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp