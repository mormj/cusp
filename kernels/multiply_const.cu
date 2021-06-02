#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cusp {

template <typename T> __global__ void kernel_multiply_const(float f, const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i] * f;
  }
}

template <typename T>
void launch_kernel_multiply_const(float f, const T *in, T *out, int grid_size, int block_size,
                        int N, cudaStream_t stream = 0) {

  if (stream) {
    kernel_add_const<<<grid_size, block_size, 0, stream>>>(f, in, out, N);
  } else {
    kernel_add_const<<<grid_size, block_size>>>(f, in, out, N);
  }
}

#define IMPLEMENT_KERNEL(T)                                                    \
  template void launch_kernel_multiply_const(float f, const T *in, T *out, int grid_size,         \
                                   int block_size, int N,                      \
                                   cudaStream_t stream);

IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)

} // namespace cusp

