#include <cuda_runtime.h> 

__constant__ float kernel[3]; 

__global__ void convolution1D(
    const float* input, 
    const float* kernel, 
    const float* output, 
    int input_size, 
    int kernel_size) { 

        __shared__ float shared_input[]; 

        int output_size = input_size - kernel_size + 1; 
        int t = blockIdx.x * blockDim.x + threadIdx.x; 
        int tid = threadIdx.x; 
        int offset = blockIdx.x * blockDim.x; 
        int tile_size = blockDim.x + kernel_size - 1; 

        if (tid < tile_size && (offset + tid) < input_size) { 
            shared_input[tid] = input[offset + tid]; 
        }
        __syncthreads();

        if (t >= output_size) return; 

        float accum = 0.0f; 

        for (int i = 0; i < kernel_size; i++) { 
            accum += shared_input[t + i] * kernel[i]; 
        }

        output[t] = accum; 
        __syncthreads(); 
    }