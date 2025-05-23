// The following kernel is the embedding kernel for gemma3 inference
// All the kernels are designed to run on an RTX 4090 GPU
#include <cuda_fp16.h>

__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) embedding_kernel(
    const int __restrict__ *token_ids, // [B, T]
    const half __restrict__*lookup_table, // [V, D]
    half* output, // [B, T, D]
    const int B, // batch_size 
    const int T, // sequence length
    const int D, // embedding dimension 
    const int V, // vocabulary size
) {
    int a = blockIdx.x; // batch index
    int b = blockIdx.y; // sequence index
    int x = threadIdx.x; 

    // bounds check 
    if ( a >= B || b >= T || x >= D) return;

    int token_id = token_ids[a * T + b]
}