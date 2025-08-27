import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'ollama')  # Options: 'ollama', 'replicate'

# Embedding Configuration (Ollama models typically use different dimensions)
EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', '768'))  # nomic-embed-text dimension

# Llama Configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'nomic-embed-text')
LLAMA_MODEL = os.getenv('LLAMA_3_2_MODEL', 'llama3.2')
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')
REPLICATE_MODEL = os.getenv('REPLICATE_MODEL', 'meta/llama-3.1-70b-instruct')

# Pinecone Configuration
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', 'gmail-rag-index')

# Gmail Configuration
GMAIL_SCOPES = [os.getenv('GMAIL_SCOPES', 'https://www.googleapis.com/auth/gmail.readonly')]
GMAIL_CREDENTIALS_PATH = 'auth/credentials/credentials.json'
GMAIL_TOKEN_PATH = 'auth/credentials/token.json'

# RAG Configuration
MAX_CONTEXT_LENGTH = 3000
TOP_K_RESULTS = 5

# Debug Configuration
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FILE = os.getenv('LOG_FILE', 'rag_gmail.log')

def validate_config():
    """Validate that all required environment variables are set."""
    required_vars = {
        'PINECONE_API_KEY': PINECONE_API_KEY,
        'PINECONE_ENV': PINECONE_ENV
    }
    
    # Add provider-specific validation
    if LLM_PROVIDER == 'replicate':
        required_vars['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    if not os.path.exists(GMAIL_CREDENTIALS_PATH):
        raise FileNotFoundError(f"Gmail credentials file not found at {GMAIL_CREDENTIALS_PATH}")
    
    return True

if __name__ == "__main__":
    try:

        if validate_config():
            print("Validated Successfully")

    except Exception as ex:
        print(f"{ex}")
    
