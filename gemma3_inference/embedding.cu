// The following kernel is the embedding kernel for gemma3 inference
// All the kernels are designed to run on an RTX 4090 GPU
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void embedding_kernel(
    const int __restrict__ *token_ids, // [B, T]
    const half __restrict__*lookup_table, // [V, D]
    half* output, // [B, T, D]
    const int B, // batch_size 
    const int T, // sequence length
    const int D, // embedding dimension 
    const int V // vocabulary size
) {
    // Use dynamic shared memory instead of static allocation
    extern __shared__ int shared_token_ids[];
    
    int batch_idx = blockIdx.x; // batch index
    int seq_idx = blockIdx.y;   // sequence index
    int thread_idx = threadIdx.x; 
    
    // Bounds checking
    if (batch_idx >= B || seq_idx >= T) return;

    // Cooperatively load token IDs to shared memory
    // Each thread loads one token ID
    if (thread_idx < T) {
        shared_token_ids[thread_idx] = token_ids[batch_idx * T + thread_idx];
    }
    __syncthreads();
    
    int token_id = shared_token_ids[seq_idx];
    
    // Bounds check for token_id
    if (token_id >= V || token_id < 0) return;
    
    for(int dim_idx = thread_idx; dim_idx < D; dim_idx += blockDim.x) { 
        half val = lookup_table[token_id * D + dim_idx];
        output[batch_idx * T * D + seq_idx * D + dim_idx] = val;
    }
}