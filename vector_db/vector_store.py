import os
import sys
from typing import List, Dict, Optional
from pinecone import Pinecone, ServerlessSpec

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION

def init_pinecone():
    """Initialize Pinecone connection and create index if needed."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Create index if not exists
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
        pc.create_index(
            name=PINECONE_INDEX_NAME, 
            dimension=EMBEDDING_DIMENSION,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=PINECONE_ENV
            )
        )
    
    index = pc.Index(PINECONE_INDEX_NAME)
    return index

def upsert_embeddings(index, vectors: List[Dict]):
    """
    Upsert embeddings into Pinecone.
    vectors: List of dicts with keys: id (str), values (List[float]), metadata (optional dict)
    Example: [{"id": "email1", "values": [...], "metadata": {...}}, ...]
    """
    index.upsert(vectors)

def query_similar(index, vector: List[float], top_k: int = 5, filter_dict: Optional[Dict] = None):
    """
    Query index for top_k similar vectors.
    Returns matches with id, score, and metadata.
    
    Args:
        index: Pinecone index
        vector: Query embedding vector
        top_k: Number of results to return
        filter_dict: Optional metadata filters
    """
    query_params = {
        "vector": vector,
        "top_k": top_k,
        "include_metadata": True
    }
    
    if filter_dict:
        query_params["filter"] = filter_dict
    
    result = index.query(**query_params)
    return result.matches

def prepare_email_vectors(emails: List[Dict], embeddings: List[List[float]]) -> List[Dict]:
    """
    Prepare email data for Pinecone upsert with enhanced metadata.
    
    Args:
        emails: List of email dictionaries
        embeddings: List of embedding vectors
        
    Returns:
        List of vectors ready for Pinecone upsert
    """
    vectors = []
    for email, embedding in zip(emails, embeddings):
        # Create comprehensive metadata
        metadata = {
            "subject": email.get('subject', ''),
            "snippet": email.get('snippet', '')[:500],  # Limit snippet length
            "from": email.get('from', ''),
            "date": email.get('date', ''),
            "threadId": email.get('threadId', ''),
            "body_preview": email.get('body', '')[:1000]  # Store first 1000 chars
        }
        
        # Remove empty metadata fields
        metadata = {k: v for k, v in metadata.items() if v}
        
        vectors.append({
            "id": email['id'],
            "values": embedding,
            "metadata": metadata
        })
    
    return vectors

def delete_all_vectors(index):
    """Delete all vectors from the index."""
    index.delete(delete_all=True)
    print(f"All vectors deleted from index: {PINECONE_INDEX_NAME}")

if __name__ == "__main__":
    index = init_pinecone()
    
    # Example dummy embedding vector (using EMBEDDING_DIMENSION)
    dummy_vector = [0.1] * EMBEDDING_DIMENSION
    
    upsert_embeddings(index, [{"id": "test1", "values": dummy_vector, "metadata": {"subject": "Test Email"}}])
    
    matches = query_similar(index, dummy_vector, top_k=1)
    print("Query results:", matches)
