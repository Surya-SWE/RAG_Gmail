import os
import sys
from typing import List
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import OLLAMA_BASE_URL, OLLAMA_MODEL

def get_embeddings(texts: List[str], model: str = None) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using Ollama.
    Args:
        texts (List[str]): List of email bodies or snippets.
        model (str): Ollama model name for embeddings.
    Returns:
        List[List[float]]: List of embedding vectors.
    """
    if model is None:
        model = OLLAMA_MODEL
    
    embeddings = []
    url = f"{OLLAMA_BASE_URL}/api/embeddings"
    
    for i, text in enumerate(texts):
        payload = {
            "model": model,
            "prompt": text
        }
        
        try:
            print(f"Processing text {i+1}/{len(texts)}...")
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            embedding = response.json()["embedding"]
            embeddings.append(embedding)
            print(f"✓ Completed {i+1}/{len(texts)}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama embedding API error: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected response format from Ollama: {e}")
    
    return embeddings

if __name__ == "__main__":
    sample_texts = [
        "Your flight itinerary is attached.",
        "Reminder: project meeting at 2 PM."
    ]
    try:
        vectors = get_embeddings(sample_texts)
        print(f"Generated {len(vectors)} embeddings")
        print(f"Embedding dimension: {len(vectors[0])}")
        print("Vector for first text (truncated):", vectors[0][:5])
    except Exception as e:
        print(f"Error generating embeddings: {e}")
