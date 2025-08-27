from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import json


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text based on prompt."""
        pass




class OllamaProvider(LLMProvider):
    """Ollama LLM provider for local Llama models."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url
        self.model = model
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, **kwargs) -> str:
        """Generate text using Ollama API."""
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get('top_p', 1),
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()['response'].strip()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")


def get_llm_provider(provider_name: str, **config) -> LLMProvider:
    """Factory function to get the appropriate LLM provider."""
    providers = {
        'ollama': OllamaProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
    
    return providers[provider_name](**config)