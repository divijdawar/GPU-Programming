import os
import torch
from transformers import AutoTokenizer, AutoModel
import embedding_cuda

class Gemma3InferenceEngine:
    def __init__(self, model_path=None):
        # Use environment variable or default path for cloud deployment
        self.model_path = model_path or os.getenv('MODEL_PATH', './gemma3_4b_it')
        
        # Load tokenizer and model efficiently
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModel.from_pretrained(self.model_path, torch_dtype=torch.float16)
        
        # Extract and move embedding weights to GPU
        self.embedding_weights = model.embed_tokens.weight.cuda().contiguous()
        del model  # Free memory immediately
        
        # Model dimensions
        self.vocab_size = self.embedding_weights.shape[0]
        self.embed_dim = self.embedding_weights.shape[1]
    
    def tokenize(self, text):
        return self.tokenizer.encode(text, return_tensors="pt")
    
    def embedding(self, tokens):
        tokens = tokens.cuda().int().contiguous()
        B, T = tokens.shape
        
        return embedding_cuda.embedding_forward(
            tokens, self.embedding_weights, B, T, self.embed_dim, self.vocab_size
        )

def main():
    engine = Gemma3InferenceEngine()
    
    text = input("How can I help you today? ")
    tokens = engine.tokenize(text)
    return 

if __name__ == "__main__":
    main() 