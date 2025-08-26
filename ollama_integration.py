"""
ollama_integration.py - Script to integrate Ollama as a fallback LLM
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

class OllamaLLMWrapper:
    """
    Wrapper class for Ollama LLM integration
    """
    
    def __init__(self, model_name: str = "llama2"):
        """
        Initialize the Ollama LLM wrapper
        """
        self.model_name = model_name
        self.ollama_llm = Ollama(model=model_name)
    
    def get_llm(self):
        """
        Get the Ollama LLM instance
        """
        return self.ollama_llm

class HuggingFaceEmbeddingsWrapper:
    """
    Wrapper class for HuggingFace embeddings as a fallback for OpenAI embeddings
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        """
        Initialize the HuggingFace embeddings wrapper
        """
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
    
    def get_embeddings(self):
        """
        Get the HuggingFace embeddings instance
        """
        return self.embeddings

def get_llm(use_ollama: bool = False, ollama_model: str = "llama2"):
    """
    Get an LLM instance, using Ollama if specified or if OpenAI API key is not available
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Use Ollama if explicitly requested or if OpenAI API key is not available
    if use_ollama or not openai_api_key:
        print("Using Ollama LLM")
        ollama_wrapper = OllamaLLMWrapper(model_name=ollama_model)
        return ollama_wrapper.get_llm()
    else:
        # Import here to avoid loading OpenAI if not needed
        from langchain_openai import ChatOpenAI
        print("Using OpenAI LLM")
        return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

def get_embeddings(use_huggingface: bool = False, hf_model: str = "sentence-transformers/all-mpnet-base-v2"):
    """
    Get embeddings, using HuggingFace if specified or if OpenAI API key is not available
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Use HuggingFace if explicitly requested or if OpenAI API key is not available
    if use_huggingface or not openai_api_key:
        print("Using HuggingFace Embeddings")
        hf_wrapper = HuggingFaceEmbeddingsWrapper(model_name=hf_model)
        return hf_wrapper.get_embeddings()
    else:
        # Import here to avoid loading OpenAI if not needed
        from langchain_openai import OpenAIEmbeddings
        print("Using OpenAI Embeddings")
        return OpenAIEmbeddings()

if __name__ == "__main__":
    # Example usage
    llm = get_llm(use_ollama=True)
    response = llm.invoke("What is the meaning of life?")
    print(response)
