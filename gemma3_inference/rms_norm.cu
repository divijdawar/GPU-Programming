#include <cuda_runtime.h> 
#include <torch/extension.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32

template <typename T>
__global__ void __launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_BLOCKS_PER_SM) rms_norm_kernel(
    const T* __restrict__ input, // [N,D]
          T* __restrict__ output, // [N,D]
    const T* __restrict__ gamma, // scale parameter
    int N, // batch size
    int D, // hidden dimension
    float epsilon
) { 
    int thread_idx = threadIdx.x;
    int block_idx = blockIdx.x;
    int block_dim = blockDim.x;

    if (block_idx >= N) return; 

    // shared memory for final reduction across warps
    __shared__ float warp_sums[MAX_THREADS_PER_BLOCK / WARP_SIZE];
    float thread_sum_sq = 0.0f;

    // computing local sums
    for (int i = thread_idx; i < D; i += block_dim) { 
        int idx = block_idx * D + i;
        float a = input[idx];
        thread_sum_sq += a * a; 
    }

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    // warp-level reduction using shuffle down
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    // store results in shared memory
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum_sq;
    }

    __syncthreads();

    // final reduction across warps
    if (warp_id == 0) { 
        float total_sum_sq = 0.0f; 

        if (lane_id < blockDim.x / WARP_SIZE) {
            total_sum_sq = warp_sums[lane_id];
        }

        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            total_sum_sq += __shfl_down_sync(0xffffffff, total_sum_sq, offset);
        }
        
        // compute RMS normalization factor
        if (threadIdx.x == 0) {
            float rms = sqrtf(total_sum_sq / D + epsilon);
            warp_sums[0] = 1.0f / rms; 
        }
    }
    
    __syncthreads();
    
    float inv_rms = warp_sums[0];
    
    for (int i = thread_idx; i < D; i += block_dim) {
        int idx = block_idx * D + i;
        float normalized = input[idx] * inv_rms;
        output[idx] = gamma[i] * normalized; 
    }
}