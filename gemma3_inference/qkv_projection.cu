#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda_fp16.h>

#define MAX_THREADS_PER_BLOCK 1024
#define MIN_BLOCKS_PER_SM 1
#define TILE_SIZE 16

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) qkv_projection_kernel(
    const half* __restrict__ X, // [B,T, D]
          half* __restrict__ output, // [B, T, 3H]
    const half* __restrict__ W_q, // [3H, D]
    int B, // batch size
    int T, // sequence length 
    int D, // hidden dimension
    int H // number of heads
) { 
    int b = blockIdx.x; // batch index
    int t = blockIdx.y; // sequence index
    int h = blockIdx.z; // hidden dimension index
    
    int idx = 
}
