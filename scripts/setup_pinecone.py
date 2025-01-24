# scripts/setup_pinecone.py

import pinecone
import os

def setup_pinecone_index(api_key, environment, index_name='job-postings', dimension=1536, metric='cosine'):
    """
    Initialize Pinecone and create an index if it doesn't exist.

    :param api_key: Pinecone API key.
    :param environment: Pinecone environment.
    :param index_name: Name of the Pinecone index.
    :param dimension: Dimension of the embeddings.
    :param metric: Metric for similarity search.
    """
    pinecone.init(api_key=api_key, environment=environment)
    
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
        print(f"Pinecone index '{index_name}' created successfully.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")

if __name__ == "__main__":
    SERPAPI_API_KEY = os.getenv('SERPAPI_API_KEY', 'YOUR_SERPAPI_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY', 'YOUR_PINECONE_API_KEY')
    PINECONE_ENVIRONMENT = os.getenv('PINECONE_ENVIRONMENT', 'YOUR_PINECONE_ENVIRONMENT')

    setup_pinecone_index(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
