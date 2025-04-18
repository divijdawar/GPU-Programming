#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>
#include <stdlib.h>

// implementing custom atomic max for floats using atomicCAS
__device__ float atomicMax(float *address, float val) { 
    // convert float to int since atomicCAS works only on integers
    int *address_as_i = (int*)address; 
    int old = *address_as_i, assumed; 

    do {
        assumed = old; 
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old); 

    return __int_as_float(old); 
}

__global__ void atomicMaxKernel(const float *array, float *result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        atomicMax(result, array[idx]); 
    }
}

// Using atomicExch to swap a value and then perform a simple calculation 
__global__ void atomicExchKernel(float *data, float newValue, int N) { 
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        float oldValue = atomicExch(&data[idx], newValue); 
        data[idx] = oldValue + newValue; 
    }   
}

int main() {
    int N = 1 << 20; 
    float *h_data = (float*)malloc(N * sizeof(float)); 
    float *h_result = (float*)malloc(sizeof(float)); 
    float *d_data, *d_result; 

    cudaMalloc(&d_data, N * sizeof(float)); 
    cudaMalloc(&d_result, sizeof(float)); 

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100); 
    }
    
    // Initialize result to minimum float value
    *h_result = -FLT_MAX;
    
    // Copy data to device
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block; 

    cudaEvent_t start, stop; 
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 

    // First kernel: atomicMax
    printf("Running atomicMax kernel...\n");
    cudaEventRecord(start, 0);

    atomicMaxKernel<<<blocks_per_grid, threads_per_block>>>(d_data, d_result, N); 

    cudaEventRecord(stop, 0); 
    cudaEventSynchronize(stop); 

    float milliseconds = 0; 
    cudaEventElapsedTime(&milliseconds, start, stop); 
    
    // Copy result back
    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate GFLOPS for atomicMax kernel
    // Each thread performs approximately 1 operation (comparison and potential update)
    double operations = (double)N;
    double gigaFlops = (operations * 1e-9f) / (milliseconds * 1e-3f);
    
    printf("atomicMax kernel result: %f\n", *h_result);
    printf("atomicMax kernel time: %f ms\n", milliseconds);
    printf("atomicMax kernel performance: %f GFLOPS\n\n", gigaFlops);
    
    // Second kernel: atomicExch
    printf("Running atomicExch kernel...\n");
    float newValue = 10.0f;
    
    cudaEventRecord(start, 0);
    
    atomicExchKernel<<<blocks_per_grid, threads_per_block>>>(d_data, newValue, N);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate GFLOPS for atomicExch kernel
    // Each thread performs approximately 2 operations (exchange and addition)
    operations = (double)N * 2;
    gigaFlops = (operations * 1e-9f) / (milliseconds * 1e-3f);
    
    printf("atomicExch kernel time: %f ms\n", milliseconds);
    printf("atomicExch kernel performance: %f GFLOPS\n", gigaFlops);
    
    // Copy some results back to verify
    cudaMemcpy(h_data, d_data, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("First 10 elements after atomicExch:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_data[i]);
    }
    printf("\n");
    
    // Clean up
    free(h_data);
    free(h_result);
    cudaFree(d_data);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}


