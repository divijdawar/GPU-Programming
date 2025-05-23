// The following kernel is the embedding kernel for gemma3 inference
// All the kernels are designed to run on an RTX 4090 GPU
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void embedding_kernel(
    const int __restrict__ *token_ids, // [B, T]
    const half __restrict__*lookup_table, // [V, D]
    half* output, // [B, T, D]
    const int B, // batch_size 
    const int T, // sequence length
    const int D, // embedding dimension 
    const int V // vocabulary size
) {
    int a = blockIdx.x; // batch index
    int b = blockIdx.y; // sequence index
    int x = threadIdx.x; 

    __shared__ int shared_token_ids[T];

    // Load token IDs to shared memory
    if (x < T) {
        shared_token_ids[x] = token_ids[a * T + x];
    }
    __syncthreads();
    
    for(int i = x; i < D; i += blockDim.x) { 
        half val = lookup_table[shared_token_ids[b] * D + i];
        output[a * T * D + b * D + i] = val;
    }
}