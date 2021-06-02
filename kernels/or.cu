#include "../include/cusp/or.cuh"

namespace cusp {

template <typename T> __global__ void kernel_or(const T *in, T *out,
                                                int num_inputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
    for (int i = id; i < num_inputs * N; i += N) {
      out[id] |= in[i];
    }
  }
}

template <typename T>
void launch_kernel_or(const T *in, T *out, int grid_size, int block_size,
                      int num_inputs, int N, cudaStream_t stream = 0) {

  if (stream) {
    kernel_or<<<grid_size, block_size, 0, stream>>>(in, out, N);
  } else {
    kernel_or<<<grid_size, block_size>>>(in, out, N);
  }
}

#define IMPLEMENT_KERNEL(T)                                                \
  template void launch_kernel_or(const T *in, T *out, int grid_size,        \
                                 int block_size, int num_inputs, int N,      \
                                 cudaStream_t stream);

IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)

} // namespace cusp
