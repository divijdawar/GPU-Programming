import torch 
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def tokenize(text):
    # Path to the directory containing the tokenizer files
    model_path = "/Users/divij/Desktop/Programming/GPU-Programming/gemma3_4b_it" 
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # You would typically use the tokenizer to encode the input text
    # inputs = tokenizer(text, return_tensors="pt")
    # return inputs
    print(f"Tokenizer loaded successfully from {model_path}")
    return tokenizer

    
if __name__ == "__main__":
    main() 