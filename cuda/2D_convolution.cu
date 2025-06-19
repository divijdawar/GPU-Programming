#include <cuda_runtime.h>

__global__ void convolution2D(
    const float* __restrict__ input, 
    const float* __restrict__ kernel, 
    float* __restrict__ output, 
    int input_rows, 
    int input_cols, 
    int kernel_rows, 
    int kernel_cols
) { 
    int output_rows = input_rows - kernel_rows + 1; 
    int output_cols = input_cols - kernel_cols + 1; 

    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row >= output_rows || col >= output_cols) return; 

    float accum = 0.0f; 

    for(int i = 0; i < kernel_rows; i++) { 
        for (int j = 0; j < kernel_cols; j++) { 
            accum += input[(row + i) * input_cols + (col + j)] * kernel[i * kernel_cols + j]; 
        }
    }
    output[row * output_cols + col] = accum; 
}