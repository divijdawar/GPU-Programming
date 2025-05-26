#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel declaration
__global__ void embedding_kernel(
    const int* token_ids,
    const half* lookup_table,
    half* output,
    const int B,
    const int T,
    const int D,
    const int V
);

// CUDA kernel launcher function
torch::Tensor embedding_cuda_forward(
    torch::Tensor token_ids,    // [B, T] - int32
    torch::Tensor lookup_table, // [V, D] - half
    int B, int T, int D, int V
) {
    // Minimal validation for speed
    TORCH_CHECK(token_ids.is_cuda() && lookup_table.is_cuda(), "Tensors must be on CUDA");
    
    // Create output tensor
    torch::Tensor output = torch::empty({B, T, D}, 
        torch::TensorOptions().dtype(torch::kFloat16).device(token_ids.device()));
    
    // Launch kernel with optimal configuration for cloud GPUs
    dim3 grid(B, T);
    dim3 block(min(D, 1024)); // Adaptive block size
    
    embedding_kernel<<<grid, block>>>(
        token_ids.data_ptr<int>(),
        reinterpret_cast<const half*>(lookup_table.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        B, T, D, V
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    // Synchronize to catch any kernel execution errors
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution failed: ", cudaGetErrorString(err));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_forward", &embedding_cuda_forward, "Embedding forward pass (CUDA)");
} 