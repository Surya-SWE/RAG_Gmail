# Gmail RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying your Gmail inbox using LLMs and vector search.

## Features

- Fetches emails from Gmail
- Generates embeddings using Ollama
- Stores vectors in Pinecone
- Natural language Q&A over your emails

## Setup

1. **Clone the repo**
   ```bash
   git clone <your-repo-url>
   cd RAG_GMail
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   - Copy `.env.example` to `.env` and fill in your API keys and config.
   - Place Gmail credentials in `auth/credentials/credentials.json`.

4. **Start Ollama**
   ```bash
   ollama serve
   ```

5. **Run ingestion**
   ```bash
   python main.py ingest
   ```

6. **Query your emails**
   ```bash
   python main.py query
   ```

## Project Structure

```
config/         # Settings and config validation
auth/           # Gmail authentication
email_ingest/   # Email fetching logic
embedding/      # Embedding generation (Ollama)
vector_db/      # Pinecone vector DB logic
rag_core/       # RAG pipeline and answer generation
main.py         # Entry point
requirements.txt
README.md
```

## Notes

- Do **not** commit your Gmail tokens or API keys.
- Make sure Ollama and Pinecone are running before using the pipeline.