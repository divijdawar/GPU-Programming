#include <cuda_runtime.h> 
#include <torch/extension.h>

template <typename T, typnename Accum=float>
__global__ void layer_norm_kernel(
    const T *__restrict__ input, 
    T *__restrict__ output, 
    const T *__restrict__ gamma, 
    const T *__restrict__ beta, 
    int N, 
    int D, 

)
