#include <cuda_runtime.h>

void convolution2D(
    const float* input, 
    const float* kernel, 
    float* output, 
    int input_rows, 
    int input_cols, 
    int kernel_rows, 
    int kernel_cols
) { 
    int output_rows = input_rows - kernel_rows + 1; 
    int output_cols = input_cols - kernel_cols + 1; 

    int row = blockIdx.y + block_dim.y + threadIdx.y 
    int col = blockIdx.x + block_dim.x + threadIdx.x 

    if (row >= output_rows || col >= output_cols) return; 

    float accum = 0.0f; 

    for(int i = 0; i < kernel_rows; i++) { 
        for (int j = 0; j < kernel_cols; j++) { 
        }
    }
}