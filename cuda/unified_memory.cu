#include <cuda_runtime.h>
#include <stdio.h>

// using unified memory, the cpu and the gpu can access the same memory space
// using cudaMallocManaged, the memory is allocated on the cpu and the gpu

__global__ void vectorAdd(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    if (idx < N) {
        c[idx] = a[idx] + b[idx]; 
    }
}

int main() {
    int N = 1 << 20; 
    size_t bytes = N * sizeof(float); 

    float *a, *b, *c; 
    //memory is allocated once
    // cpu initializes the data while the gpu can access it
    //cudaMallocManaged(&a, bytes); 
    //cudaMallocManaged(&b, bytes); 
    //cudaMallocManaged(&c, bytes); 

    cudage

    for (int i = 0; i < N; i++) {
        a[i] = 1.0f; 
        b[i] = 2.0f; 
    }

    vectorAdd<<<(N + 256) / 256, 256>>>(a,b,c,N);
    cudaDeviceSynchronize(); 

    cudaFree(a); 
    cudaFree(b); 
    cudaFree(c); 

    return 0; 
    
}