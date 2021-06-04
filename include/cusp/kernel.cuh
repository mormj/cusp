
#pragma once

#include <vector>

namespace cusp {

class kernel {
protected:
  int _block_size;
  int _grid_size;
  cudaStream_t _stream = 0;

public:
  kernel() = default;
  virtual ~kernel() = default;
  void set_stream(cudaStream_t stream) { _stream = stream; }
  void set_block_and_grid(int block_size, int grid_size) {
    _block_size = block_size;
    _grid_size = grid_size;
  }
  virtual cudaError_t launch(const std::vector<const void *> inputs,
                      const std::vector<void *> outputs, size_t nitems) = 0;
  virtual cudaError_t occupancy(int *minBlock, int *minGrid) = 0;
};

} // namespace cusp