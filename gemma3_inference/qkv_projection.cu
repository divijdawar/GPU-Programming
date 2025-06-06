#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda_fp16.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_SM 1

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) qkv_projection_kernel(
    const half* __restrict__ input, // [B,T, D]
    const half* __restrict__ W_q, // [B, 3D]
          half* __restrict__ output, // [B, T, 3D]
    int B, // batch size
    int T, // sequence length
    int D, // hidden dimension
    int D_q, // query dimension
    int D_k, // key dimension
    int D_v // value dimension
) {
    
}