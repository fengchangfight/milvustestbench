"""
Shared embeddings module for Milvus testbench.

This module provides common embedding functions and LLM utilities that can be used across
multiple files in the project.
"""

import os
import requests
from typing import List
from sentence_transformers import SentenceTransformer

# Initialize the model once at module level for efficiency
model = SentenceTransformer('all-MiniLM-L6-v2')

# Optional: OpenAI client for future use
# from openai import OpenAI
# openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# model_name = "text-embedding-3-small"


def emb_text(text: str) -> List[float]:
    """
    Generate embedding for a single text using SentenceTransformer.
    
    Args:
        text (str): The text to embed
        
    Returns:
        List[float]: The embedding vector
    """
    return model.encode(text).tolist()


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.
    
    Args:
        texts (List[str]): List of texts to embed
        
    Returns:
        List[List[float]]: List of embedding vectors
        
    Note:
        Returns empty list if input is empty.
    """
    if not texts:
        return []
    return [emb_text(text) for text in texts]


# Alternative OpenAI implementation (commented out)
# def get_embeddings_openai(texts: List[str]) -> List[List[float]]:
#     """
#     Generate embeddings using OpenAI API.
#     
#     Args:
#         texts (List[str]): List of texts to embed
#         
#     Returns:
#         List[List[float]]: List of embedding vectors
#     """
#     if not texts:
#         return []
#     response = openai_client.embeddings.create(input=texts, model=model_name)
#     return [embedding.embedding for embedding in response.data]


def get_llm_response(system_prompt: str, user_prompt: str, model: str = "llama2") -> str:
    return get_llm_response_single(f"{system_prompt}\n\n{user_prompt}", model);


def get_llm_response_single(prompt: str, mode: str = "llama2") -> str:
    """
    Get response from local Ollama model.
    
    Args:
        system_prompt (str): The system prompt/instructions
        user_prompt (str): The user's question or request
        model (str): The Ollama model to use (default: "llama2")
        
    Returns:
        str: The LLM response or error message
    """
    try:
        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': model,
                                   'prompt': f"{prompt}",
                                   'stream': False
                               })
        return response.json()['response']
    except Exception as e:
        return f"Error: Could not connect to Ollama. Make sure it's running with: ollama serve\nError: {e}"