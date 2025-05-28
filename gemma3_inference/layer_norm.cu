#include <cuda_runtime.h> 
#include <torch/extension.h>

template <typename T>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) layer_norm_kernel(
    const T* __restrict__ input,
          T* __restrict__ output, 
    const T* __restrict__ gamma, // scale parameter
    const T* __restrict__ beta, // shift parameter
    int N, 
    int D, 
    float epsilon
) { 
    int thread_idx = 
}