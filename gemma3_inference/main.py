import json
import os
import torch 
import torch.nn as nn
from transformers import AutoTokenizer

def tokenize(text): 
    model_path = "/Users/divij/Desktop/Programming/GPU-Programming/gemma3_4b_it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer.encode(text, return_tensors="pt")

def main(): 
    text = input("How can I help you today? ")
    tokens = tokenize(text)
    print(tokens)

if __name__ == "__main__":
    main() 