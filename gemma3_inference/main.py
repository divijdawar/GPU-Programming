import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
import embedding_cuda  # Import the compiled CUDA extension

class Gemma3InferenceEngine: 
    def __init__(self):
        self.model_path = "/Users/divij/Desktop/Programming/GPU-Programming/gemma3_4b_it"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load the model to get embedding weights
        self.model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float16)
        self.embedding_weights = self.model.embed_tokens.weight.cuda()
        
        # Model dimensions
        self.vocab_size = self.embedding_weights.shape[0]
        self.embed_dim = self.embedding_weights.shape[1]
    
    def tokenize(self, text): 
        return self.tokenizer.encode(text, return_tensors="pt")
    
    def embedding(self, tokens): 
        # Move tokens to GPU and ensure correct dtype
        tokens = tokens.cuda().int()
        
        B, T = tokens.shape
        
        # Call the CUDA kernel
        embeddings = embedding_cuda.embedding_forward(
            tokens, 
            self.embedding_weights,
            B, T, self.embed_dim, self.vocab_size
        )
        
        return embeddings

def main(): 
    # Initialize the inference engine
    engine = Gemma3InferenceEngine()
    
    text = input("How can I help you today? ")
    tokens = engine.tokenize(text)
    print(f"Tokens shape: {tokens.shape}")
    
    # Get embeddings using CUDA kernel
    embeddings = engine.embedding(tokens)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings dtype: {embeddings.dtype}")
    
    return embeddings

if __name__ == "__main__":
    main() 