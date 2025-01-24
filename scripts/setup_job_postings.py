# scripts/setup_job_postings.py

import os
from serpapi import GoogleSearch
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
import pinecone
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_job_postings(serpapi_api_key, query, location='United States', num_results=50):
    """
    Fetch job postings from Google using SerpApi.

    :param serpapi_api_key: SerpApi API key.
    :param query: Job search query.
    :param location: Location for job search.
    :param num_results: Number of job postings to fetch.
    :return: DataFrame containing job postings.
    """
    params = {
        "engine": "google_jobs",
        "q": query,
        "location": location,
        "hl": "en",
        "gl": "us",
        "api_key": serpapi_api_key
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    jobs = results.get('jobs_results', [])

    # Pagination handling if necessary
    while 'serpapi_pagination' in results and 'next' in results['serpapi_pagination'] and len(jobs) < num_results:
        params['start'] = results['serpapi_pagination']['next']
        search = GoogleSearch(params)
        results = search.get_dict()
        jobs.extend(results.get('jobs_results', []))
        logging.info(f"Fetched {len(jobs)} job postings so far...")

    # Limit to the desired number of results
    jobs = jobs[:num_results]

    # Convert to DataFrame
    df = pd.DataFrame(jobs)

    # Ensure unique job IDs
    if 'job_id' not in df.columns:
        df['job_id'] = df.index.astype(str)

    logging.info(f"Total job postings fetched: {len(df)}")
    return df

def generate_embeddings(openai_api_key, df, text_column='description'):
    """
    Generate embeddings for a specified text column in the DataFrame.

    :param openai_api_key: OpenAI API key.
    :param df: DataFrame containing job postings.
    :param text_column: Column containing text to embed.
    :return: List of embeddings.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    texts = df[text_column].tolist()
    logging.info("Generating embeddings for job descriptions...")
    job_embeddings = embeddings.embed_documents(texts)
    logging.info("Embeddings generated successfully.")
    return job_embeddings

def initialize_pinecone(pinecone_api_key, pinecone_environment, index_name='job-postings', dimension=1536, metric='cosine'):
    """
    Initialize Pinecone and create an index if it doesn't exist.

    :param pinecone_api_key: Pinecone API key.
    :param pinecone_environment: Pinecone environment.
    :param index_name: Name of the Pinecone index.
    :param dimension: Dimension of the embeddings.
    :param metric: Metric for similarity search.
    :return: Pinecone Index object.
    """
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=dimension, metric=metric)
        logging.info(f"Pinecone index '{index_name}' created successfully.")
    else:
        logging.info(f"Pinecone index '{index_name}' already exists.")
    index = pinecone.Index(index_name)
    return index

def upsert_to_pinecone(index, df, batch_size=100):
    """
    Upsert job postings into Pinecone index in batches.

    :param index: Pinecone Index object.
    :param df: DataFrame containing job postings with embeddings.
    :param batch_size: Number of records to upsert per batch.
    """
    to_upsert = []
    for idx, row in df.iterrows():
        unique_id = str(row.get('job_id', idx))
        vector = row['embedding']
        metadata = row.drop(['embedding']).to_dict()
        to_upsert.append((unique_id, vector, metadata))

    total = len(to_upsert)
    logging.info(f"Starting upsert of {total} job postings to Pinecone...")
    for i in range(0, total, batch_size):
        batch = to_upsert[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            logging.info(f"Upserted batch {i//batch_size + 1} ({len(batch)} records).")
        except Exception as e:
            logging.error(f"Error upserting batch {i//batch_size + 1}: {e}")
    logging.info("All job postings upserted successfully.")

def main():
    # Retrieve API keys from environment variables
    serpapi_api_key = os.getenv('SERPAPI_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

    # Validate API keys
    missing_keys = []
    if not serpapi_api_key:
        missing_keys.append('SERPAPI_API_KEY')
    if not openai_api_key:
        missing_keys.append('OPENAI_API_KEY')
    if not pinecone_api_key:
        missing_keys.append('PINECONE_API_KEY')
    if not pinecone_environment:
        missing_keys.append('PINECONE_ENVIRONMENT')

    if missing_keys:
        logging.error(f"Missing the following environment variables: {', '.join(missing_keys)}")
        sys.exit(1)

    # Define job search parameters
    query = "Software Engineer"
    location = "San Francisco, CA"
    num_results = 50
    index_name = 'job-postings'

    # Fetch job postings
    job_df = fetch_job_postings(serpapi_api_key, query, location, num_results)

    # Generate embeddings
    job_embeddings = generate_embeddings(openai_api_key, job_df, text_column='description')
    job_df['embedding'] = job_embeddings

    # Initialize Pinecone and connect to index
    index = initialize_pinecone(pinecone_api_key, pinecone_environment, index_name=index_name)

    # Upsert data into Pinecone
    upsert_to_pinecone(index, job_df, batch_size=100)

if __name__ == "__main__":
    main()
