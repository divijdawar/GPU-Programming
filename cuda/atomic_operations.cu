#include <cuda_runtime.h>

__global__ void atomicAdd(const float *input, float *result, int N) {
    int idx = threadIdx.x + blockIdx.x *blockDim.x;

    // boundary check to ensure that threads do not access out-of-bound memory 
    if (idx < N) { 
        atomicAdd(result, input[idx]);
    }
}