#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void atomicAdd(const float *input, float *result , int N);

int main() {
    int N = 1<< 24; 
    size_t size = N * sizeof(float); 
    
    // allocate host memory 
    float *h_input = (float*)malloc(size);
    float h_result = 0.0f;

    // Initialize host array with random values
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // Random float between 0 and 1
    }

    // Allocate device memory 
    float *d_input, *d_result; 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch (example configuration, adjust as needed)
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    atomicAdd<<<gridSize, blockSize>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    
    // Kernel execution
    atomicAdd<<<gridSize, blockSize>>>(d_input, d_result, N);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Calculate GFLOPs: total number of floating point operations divided by execution time (in seconds)
    float gflops = (float)N / (milliseconds * 1e6f); // 1e6 converts ms to seconds and adds GFLOP scaling
    printf("Time: %f ms, GFLOPs: %f\n", milliseconds, gflops);

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum result: %f\n", h_result);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_input);
    cudaFree(d_input);
    cudaFree(d_result);
    return 0;
}