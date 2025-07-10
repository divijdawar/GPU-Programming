__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;                 

    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : 0.0f;  
}