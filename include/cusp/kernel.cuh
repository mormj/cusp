#pragma once

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cusp
{

// see if theres any way this can be genericized -> ie, autofill function
// name or something along those lines
    // actually, this can almost certainly be done by the modtool (assuming
    // a consistent naming convention is followed)
template <typename T>
void get_block_and_grid(int* minGrid, int* minBlock);
}