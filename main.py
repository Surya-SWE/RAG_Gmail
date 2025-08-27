#!/usr/bin/env python3
"""
Main script to run the complete Gmail RAG pipeline.
1. Fetches emails from the last week
2. Generates embeddings
3. Stores in Pinecone
4. Allows querying emails with natural language questions
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.settings import validate_config
from auth.gmail_auth import get_gmail_service
from email_ingest.fetch_email import fetch_last_week_emails
from embedding.embed_texts import get_embeddings
from vector_db.vector_store import init_pinecone, prepare_email_vectors, upsert_embeddings
from rag_core.generate_answer import generate_answer

def ingest_emails():
    """Fetch emails from last week and store in vector database."""
    print("=" * 50)
    print("GMAIL RAG - EMAIL INGESTION")
    print("=" * 50)
    
    # Validate configuration
    try:
        validate_config()
        print("✓ Configuration validated successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False
    
    # Initialize Gmail service
    print("\n1. Authenticating with Gmail...")
    try:
        service = get_gmail_service()
        print("✓ Gmail authentication successful")
    except Exception as e:
        print(f"❌ Gmail authentication failed: {e}")
        return False
    
    # Fetch emails from last week
    print("\n2. Fetching emails from the last 7 days...")
    try:
        emails = fetch_last_week_emails(service, max_results=10)
        print(f"✓ Fetched {len(emails)} emails")
        
        if not emails:
            print("No emails found in the last week.")
            return False
            
        # Display sample of fetched emails
        print("\nSample of fetched emails:")
        for i, email in enumerate(emails[:10]):
            print(f"  {i+1}. {email['subject'][:60]}... (from: {email['from'][:30]}...)")
            
    except Exception as e:
        print(f"❌ Failed to fetch emails: {e}")
        return False
    
    # Extract text content for embedding
    print("\n3. Preparing email content for embedding...")
    texts = []
    valid_emails = []
    
    for email in emails:
        # Use body if available, otherwise use snippet
        content = email.get('body') or email.get('snippet', '')
        if content.strip():
            texts.append(f"Subject: {email['subject']}\n\nContent: {content}")
            valid_emails.append(email)
    
    print(f"✓ Prepared {len(texts)} emails with content")
    
    # Generate embeddings
    print("\n4. Generating embeddings...")
    try:
        embeddings = get_embeddings(texts)
        print(f"✓ Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"❌ Failed to generate embeddings: {e}")
        return False
    
    # Initialize Pinecone
    print("\n5. Initializing vector database...")
    try:
        index = init_pinecone()
        print("✓ Vector database initialized")
    except Exception as e:
        print(f"❌ Failed to initialize vector database: {e}")
        return False
    
    # Prepare and upsert vectors
    print("\n6. Storing embeddings in vector database...")
    try:
        vectors = prepare_email_vectors(valid_emails, embeddings)
        upsert_embeddings(index, vectors)
        print(f"✓ Successfully stored {len(vectors)} email embeddings")
    except Exception as e:
        print(f"❌ Failed to store embeddings: {e}")
        return False
    
    print("\n✅ Email ingestion completed successfully!")
    print(f"Total emails processed: {len(valid_emails)}")
    return True

def query_emails():
    """Interactive query interface for asking questions about emails."""
    print("\n" + "=" * 50)
    print("GMAIL RAG - QUERY INTERFACE")
    print("=" * 50)
    
    # Validate configuration
    try:
        validate_config()
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Initialize Pinecone
    try:
        index = init_pinecone()
        stats = index.describe_index_stats()
        print(f"✓ Connected to vector database")
        print(f"  Total vectors: {stats['total_vector_count']}")
    except Exception as e:
        print(f"❌ Failed to connect to vector database: {e}")
        return
    
    print("\nYou can now ask questions about your emails.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        # Get user query
        query = input("Your question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting query interface...")
            break
            
        if not query:
            continue
        
        try:
            # Generate answer
            print("\n🤔 Searching and generating answer...")
            answer, sources = generate_answer(query)
            
            # Display answer
            print("\n" + "=" * 50)
            print("ANSWER:")
            print("=" * 50)
            print(answer)
            
            # Display sources
            # if sources:
            #     print("\n" + "-" * 50)
            #     print("SOURCES:")
            #     print("-" * 50)
            #     for i, source in enumerate(sources[:3], 1):
            #         print(f"\n{i}. Subject: {source.get('subject', 'N/A')}")
            #         print(f"   From: {source.get('from', 'N/A')}")
            #         print(f"   Date: {source.get('date', 'N/A')}")
            #         print(f"   Relevance Score: {source.get('score', 0):.3f}")
            #         if source.get('snippet'):
            #             print(f"   Preview: {source['snippet'][:100]}...")
            
            print("\n" + "=" * 50 + "\n")
            
        except Exception as e:
            print(f"\n❌ Error generating answer: {e}\n")

def main():
    """Main entry point for the Gmail RAG system."""
    print("\n🚀 GMAIL RAG SYSTEM")
    print("==================\n")
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'ingest':
            ingest_emails()
        elif command == 'query':
            query_emails()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python main.py ingest  - Fetch and store emails from last week")
            print("  python main.py query   - Query stored emails")
            print("  python main.py        - Run full pipeline (ingest then query)")
    else:
        # Run full pipeline
        print("Running full pipeline...\n")
        if ingest_emails():
            query_emails()

if __name__ == "__main__":
    main()