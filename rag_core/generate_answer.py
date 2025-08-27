import os
import sys
from typing import List, Tuple, Dict

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    LLM_PROVIDER, OLLAMA_BASE_URL, LLAMA_MODEL,
    TOP_K_RESULTS, MAX_CONTEXT_LENGTH
)
from embedding.embed_texts import get_embeddings
from vector_db.vector_store import init_pinecone, query_similar
from llm.providers import get_llm_provider

# Initialize LLM provider based on configuration
def get_llm():
    """Get the configured LLM provider."""
    if LLM_PROVIDER == 'ollama':
        return get_llm_provider('ollama', base_url=OLLAMA_BASE_URL, model=LLAMA_MODEL)
    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}. Available: 'ollama'")

def build_prompt(question: str, email_snippets: List[str]) -> str:
    """
    Construct an augmented prompt by combining user question and retrieved email contexts.
    """
    context = "\n\n".join(email_snippets)
    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Using only the CONTEXT above, answer the QUESTION as accurately and completely as possible."
    )
    return prompt

def generate_llm_answer(prompt: str, max_tokens: int = 512) -> str:
    """
    Send the prompt to the LLM and get the response.
    """
    llm = get_llm()
    return llm.generate(
        prompt,
        max_tokens=max_tokens,
        temperature=0.2,    # Lower temperature for more factual answers
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

def generate_answer(question: str) -> Tuple[str, List[Dict]]:
    """
    Complete RAG pipeline: embed question, search emails, generate answer.
    
    Args:
        question: User's question about emails
        
    Returns:
        Tuple of (answer, source_emails)
    """
    # Initialize Pinecone
    index = init_pinecone()
    
    # Generate embedding for the question
    question_embedding = get_embeddings([question])[0]
    
    # Search for relevant emails
    matches = query_similar(index, question_embedding, top_k=TOP_K_RESULTS)
    
    if not matches:
        return "No relevant emails found for your question.", []
    
    # Extract email snippets and metadata
    email_contexts = []
    source_emails = []
    
    for match in matches:
        metadata = match.metadata
        
        # Build context from email
        context = f"Subject: {metadata.get('subject', 'N/A')}\n"
        context += f"From: {metadata.get('from', 'N/A')}\n"
        context += f"Date: {metadata.get('date', 'N/A')}\n"
        
        # Use body preview if available, otherwise snippet
        content = metadata.get('body_preview') or metadata.get('snippet', '')
        context += f"Content: {content}\n"
        
        email_contexts.append(context)
        
        # Store source information
        source_emails.append({
            'id': match.id,
            'score': match.score,
            'subject': metadata.get('subject', 'N/A'),
            'from': metadata.get('from', 'N/A'),
            'date': metadata.get('date', 'N/A'),
            'snippet': metadata.get('snippet', '')
        })
    
    # Build augmented prompt
    prompt = build_prompt(question, email_contexts)
    
    # Truncate prompt if too long
    if len(prompt) > MAX_CONTEXT_LENGTH:
        # Keep question and truncate context
        available_context_length = MAX_CONTEXT_LENGTH - len(question) - 100
        truncated_contexts = []
        current_length = 0
        
        for context in email_contexts:
            if current_length + len(context) < available_context_length:
                truncated_contexts.append(context)
                current_length += len(context)
            else:
                break
        
        prompt = build_prompt(question, truncated_contexts)
    
    # Generate answer
    answer = generate_llm_answer(prompt)
    
    return answer, source_emails

# --- Example usage ---
if __name__ == "__main__":
    sample_question = "What is my next meeting about?"
    sample_email_snippets = [
        "Subject: Project meeting tomorrow\nHi team, Our next project meeting is scheduled for tomorrow at 2 PM in conference room B.",
        "Subject: Meeting agenda\nWe'll discuss the project timeline and deliverables tomorrow's meeting."
    ]

    prompt = build_prompt(sample_question, sample_email_snippets)
    answer = generate_answer(prompt)
    print("Generated Answer:\n", answer)