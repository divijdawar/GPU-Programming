#include <cuda.h>
#include <curand_kernel.h> 
#include <stdio.h>

#define threads_per_block 256

__global__ void monte_carlo_kernel(
    float *results, 
    int numPaths, 
    unsigned long seed,
    float S0, 
    float K, 
    float r, 
    float sigma, 
    float T) 
    {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    curandState state; 
    curand_init(seed, tid, 0, &state); 

    float sumPayoff = 0.0f; 
    int pathsPerThread = (numPaths + gridDim.x * blockDim.x - 1) / (gridDim.x * blockDim.x); 

    for (int i = 0; i < pathsPerThread; i++) { 
        float Z = curand_normal(&state); 
        float ST = S0 * exp((r - 0.5f * sigma * sigma) * T + sigma * sqrt(T) * Z); 
        float payoff = fmaxf(ST - K, 0.0f); 
        sumPayoff += payoff; 
    }

    atomicAdd(results, sumPayoff);
}

int main (){ 
    int numPaths = 1 << 20; 
    float S0 = 100.0f; 
    float K = 100.0f; 
    float r = 0.05f; 
    float sigma = 0.2f; 
    float T = 1.0f; 
    float *d_results, h_results = 0.0f;

    cudaMalloc(&d_results, sizeof(float)); 
    cudaMemset(d_results, 0, sizeof(float)); 

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int blocks = 128; 
    monte_carlo_kernel<<<blocks, threads_per_block>>>(d_results, numPaths, 12345, S0, K, r, sigma, T); 
    cudaDeviceSynchronize(); 
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Assuming each path does approximately 4 FLOPs (1 exp, 1 sqrt, 1 mul, 1 sub/fmax)
    float totalFlops = 4.0f * numPaths;
    float gflops = (totalFlops / (milliseconds / 1000.0f)) / 1e9;

    cudaMemcpy(&h_results, d_results, sizeof(float), cudaMemcpyDeviceToHost); 
    printf("Estimated price: %f\n", h_results / numPaths);
    printf("Time elapsed: %f ms\n", milliseconds);
    printf("Performance: %f GFLOPS\n", gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_results); 
    return 0; 
}