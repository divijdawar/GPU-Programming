#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define tile_width 16

__global__ void shared_matmul(const float *A, const float *B, float *C, int width) {

    // Shared memory allocation 
    __shared__ float tileA[tile_width][tile_width]; 
    __shared__ float tileB[tile_width][tile_width];

    int row = blockIdx.y * tile_width + threadIdx.y; 
    int col = blockIdx.x * tile_width + threadIdx.x; 
    float value = 0.0f; 

    //Looping over tiles
    for (int m = 0; m < (width + tile_width - 1)/ tile_width; m ++) {
        // loading matrix A into shared memory 
        if (row < width && m * tile_width + threadIdx.x < width)
            tileA[threadIdx.y][threadIdx.x] = A[row * width + m * tile_width + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        
        // loading matrix B into shared memory 
        if (col < width && m * tile_width + threadIdx.y < width)
            tileB[threadIdx.y][threadIdx.x] = B[(m * tile_width + threadIdx.y) * width + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // this is to ensure  complete tile loading 
        __syncthreads(); 

        for (int k = 0; k < tile_width; k++ ){
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < width && col < width ){
        C[row * width + col ] = value; 
    }
}

void intitilailzeMatrix (float *mat, int width ){ 
    for (int i = 0; i < width * width; i++){
        mat[i] = (float)(rand() %100)/100.0f;
    }
}
 
int main() {
    int width = 1024;
    size_t size = width * width * sizeof(float);
 
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
 
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
 
    srand(time(NULL));
    intitilailzeMatrix(h_A, width);
    intitilailzeMatrix(h_B, width);
 
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
 
    dim3 dimBlock(tile_width, tile_width);
    dim3 dimGrid((width + tile_width - 1) / tile_width, (width + tile_width - 1) / tile_width);
 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEventRecord(start);
    shared_matmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
 
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
 
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
 
    double flops = 2.0 * width * width * width; // 2 * N^3
    double gflops = flops / (milliseconds / 1000.0) / 1e9;
 
    printf("Execution time: %.3f ms\n", milliseconds); // achieving 9.38 gflops
    printf("GFLOPS: %.2f\n", gflops);
 
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
 
    return 0;
}