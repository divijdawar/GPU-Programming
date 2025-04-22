#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// comparing transfer times in pinned vs pageable memory
__global__ void vectorAdd(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        c[idx] = a[idx] + b[idx]; 
    }
}

int main() {
    int N = 1 << 20; 
    size_t bytes = N * sizeof(float); 
    float elapsed_time;
    cudaEvent_t start, stop;
    
    // Initialize CUDA timing events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("Vector size: %d elements (%lu MB)\n\n", N, bytes / (1024 * 1024));
    
    // pagable memory 
    float *a, *b, *c; 
    a = (float*)malloc(bytes);
    b = (float*)malloc(bytes);
    c = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = (float)(rand() %100) / 10.0f; 
        b[i] = (float)(rand() %100) / 10.0f; 
    }

    float *d_a, *d_b, *d_c; 
    cudaMalloc(&d_a, bytes); 
    cudaMalloc(&d_b, bytes); 
    cudaMalloc(&d_c, bytes); 

    cudaEventRecord(start, 0);
    
    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice); 

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock; 

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N); 

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Pageable Memory - first 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.2f", c[i]);
        if (i < 4) printf(", ");
    }
    printf("\n\n");
    // this takes about 4.91ms on an average on a 3070
    printf("Pageable Memory Transfer Time: %.2f ms\n\n", elapsed_time);
    
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c); 
    free(a); 
    free(b); 
    free(c);

    // pinned memory 
    float *pinned_a, *pinned_b, *pinned_c; 
    cudaMallocHost(&pinned_a, bytes); 
    cudaMallocHost(&pinned_b, bytes); 
    cudaMallocHost(&pinned_c, bytes); 
    
    for (int i = 0; i < N; i++) {
        pinned_a[i] = (float)(rand() %100) / 10.0f; 
        pinned_b[i] = (float)(rand() %100) / 10.0f; 
    }

    float *d_pinned_a, *d_pinned_b, *d_pinned_c;
    
    cudaMalloc((void**)&d_pinned_a, bytes); 
    cudaMalloc((void**)&d_pinned_b, bytes); 
    cudaMalloc((void**)&d_pinned_c, bytes); 

    cudaEventRecord(start, 0);
    
    cudaMemcpyAsync(d_pinned_a, pinned_a, bytes, cudaMemcpyHostToDevice); 
    cudaMemcpyAsync(d_pinned_b, pinned_b, bytes, cudaMemcpyHostToDevice); 

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_pinned_a, d_pinned_b, d_pinned_c, N); 

    cudaMemcpyAsync(pinned_c, d_pinned_c, bytes, cudaMemcpyDeviceToHost); 
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Pinned Memory - first 5 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("%.2f", pinned_c[i]);
        if (i < 4) printf(", ");
    }
    printf("\n\n");
    // this takes about 1.07ms on an average on a 3070, which is 4.4x faster than pageable memory
    printf("Pinned Memory Transfer Time: %.2f ms\n", elapsed_time);

    cudaFreeHost(pinned_a); 
    cudaFreeHost(pinned_b); 
    cudaFreeHost(pinned_c); 
    cudaFree(d_pinned_a); 
    cudaFree(d_pinned_b); 
    cudaFree(d_pinned_c); 

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
