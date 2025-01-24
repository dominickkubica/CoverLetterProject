# enhanced_app.py

import streamlit as st
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pandas as pd
import os
import io
from pdfminer.high_level import extract_text
import docx2txt
import tempfile
import openai

# Function to extract text from uploaded resume
def extract_resume_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            text = extract_text(tmp_file.name)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
            tmp_file.write(uploaded_file.read())
            text = docx2txt.process(tmp_file.name)
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.getvalue().decode("utf-8")
    else:
        text = ""
    return text

# Initialize Session State for API Keys
if 'serpapi_api_key' not in st.session_state:
    st.session_state['serpapi_api_key'] = ''
if 'openai_api_key' not in st.session_state:
    st.session_state['openai_api_key'] = ''
if 'pinecone_api_key' not in st.session_state:
    st.session_state['pinecone_api_key'] = ''
if 'pinecone_environment' not in st.session_state:
    st.session_state['pinecone_environment'] = ''

# Streamlit App
st.title("Job Postings Search with Cover Letter Generator")

# Sidebar for API Key Input and Navigation
st.sidebar.header("Configuration")

# API Key Inputs
with st.sidebar.expander("🔑 Enter Your API Keys"):
    st.session_state['serpapi_api_key'] = st.text_input(
        "SerpApi API Key", type="password", value=st.session_state['serpapi_api_key']
    )
    st.session_state['openai_api_key'] = st.text_input(
        "OpenAI API Key", type="password", value=st.session_state['openai_api_key']
    )
    st.session_state['pinecone_api_key'] = st.text_input(
        "Pinecone API Key", type="password", value=st.session_state['pinecone_api_key']
    )
    st.session_state['pinecone_environment'] = st.text_input(
        "Pinecone Environment", value=st.session_state['pinecone_environment']
    )
    if st.button("Save API Keys"):
        st.success("API Keys saved successfully!")

# Navigation Sidebar
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Configuration", "Search Jobs", "Generate Cover Letter"]
)

# Check if API Keys are provided
api_keys_provided = all([
    st.session_state['serpapi_api_key'],
    st.session_state['openai_api_key'],
    st.session_state['pinecone_api_key'],
    st.session_state['pinecone_environment']
])

if not api_keys_provided and app_mode != "Configuration":
    st.warning("Please enter your API keys in the sidebar before proceeding.")
    st.stop()

# Initialize OpenAI
openai.api_key = st.session_state['openai_api_key']

# Initialize Pinecone
pinecone.init(api_key=st.session_state['pinecone_api_key'], environment=st.session_state['pinecone_environment'])
index_name = 'job-postings'

# Connect to Pinecone Index
if index_name in pinecone.list_indexes():
    index = pinecone.Index(index_name)
else:
    st.error(f"Pinecone index '{index_name}' does not exist. Please ensure it's created and populated.")
    st.stop()

# Initialize LangChain Embeddings
embeddings = OpenAIEmbeddings(openai_api_key=st.session_state['openai_api_key'])

# Application Modes
if app_mode == "Search Jobs":
    st.header("Search and Explore Job Postings")

    # Input query
    query = st.text_input("Enter your job search query:", "Software Engineer in San Francisco")

    # Number of results
    num_results = st.slider("Number of job postings to fetch:", min_value=10, max_value=100, value=50, step=10)

    if st.button("Fetch Job Postings"):
        if query:
            with st.spinner("Fetching job postings..."):
                # Generate embedding for the query
                query_embedding = embeddings.embed_query(query)

                # Query Pinecone
                try:
                    results = index.query(query_embedding, top_k=num_results, include_metadata=True)
                except Exception as e:
                    st.error(f"Error querying Pinecone: {e}")
                    st.stop()

                # Convert results to DataFrame
                jobs = []
                for match in results.matches:
                    metadata = match.metadata
                    jobs.append({
                        "Title": metadata.get('title', 'No Title'),
                        "Company": metadata.get('company', 'N/A'),
                        "Location": metadata.get('location', 'N/A'),
                        "Description": metadata.get('description', 'N/A'),
                        "URL": metadata.get('url', '#')
                    })
                jobs_df = pd.DataFrame(jobs)

                if jobs_df.empty:
                    st.info("No job postings found for the given query.")
                else:
                    st.success("Job postings fetched successfully!")
                    st.dataframe(jobs_df)

elif app_mode == "Generate Cover Letter":
    st.header("Generate a Personalized Cover Letter")

    # Step 1: Select Job Posting
    st.subheader("Select a Job Posting")

    # Fetch top 100 job postings for selection
    with st.spinner("Loading job postings..."):
        try:
            top_results = index.query(embeddings.embed_query("Job"), top_k=100, include_metadata=True)
        except Exception as e:
            st.error(f"Error querying Pinecone: {e}")
            st.stop()

        job_options = []
        for match in top_results.matches:
            metadata = match.metadata
            title = metadata.get('title', 'No Title')
            company = metadata.get('company', 'N/A')
            location = metadata.get('location', 'N/A')
            job_id = match.id
            job_options.append(f"{title} at {company} ({location})")

    if not job_options:
        st.warning("No job postings available to select.")
    else:
        selected_job = st.selectbox("Choose a job posting:", job_options)

        # Find the selected job's metadata
        selected_index = job_options.index(selected_job)
        selected_metadata = top_results.matches[selected_index].metadata

        # Display selected job details
        st.markdown("**Selected Job Details:**")
        st.write(f"**Title:** {selected_metadata.get('title', 'No Title')}")
        st.write(f"**Company:** {selected_metadata.get('company', 'N/A')}")
        st.write(f"**Location:** {selected_metadata.get('location', 'N/A')}")
        st.write(f"**Description:** {selected_metadata.get('description', 'N/A')}")
        st.markdown(f"[Apply Here]({selected_metadata.get('url', '#')})")

        st.markdown("---")

        # Step 2: Upload Resume
        st.subheader("Upload Your Resume")
        uploaded_file = st.file_uploader("Choose a resume file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

        if uploaded_file is not None:
            resume_text = extract_resume_text(uploaded_file)
            if resume_text:
                st.success("Resume uploaded and parsed successfully!")
                # Optionally, display a preview of the resume text
                # st.text_area("Resume Preview:", value=resume_text[:1000] + "...", height=200)
            else:
                st.error("Unsupported file type or failed to extract text.")
        else:
            resume_text = ""

        st.markdown("---")

        # Step 3: Generate Cover Letter
        if uploaded_file is not None and resume_text:
            if st.button("Generate Cover Letter"):
                with st.spinner("Generating cover letter..."):
                    # Prepare prompt for OpenAI
                    prompt = f"""
                    Write a professional cover letter for the following job posting:

                    Job Title: {selected_metadata.get('title', 'No Title')}
                    Company: {selected_metadata.get('company', 'N/A')}
                    Location: {selected_metadata.get('location', 'N/A')}
                    Job Description: {selected_metadata.get('description', 'N/A')}

                    Based on the following resume:

                    {resume_text}

                    The cover letter should highlight relevant skills and experiences, express enthusiasm for the role, and maintain a professional tone.
                    """

                    # Generate cover letter using OpenAI
                    try:
                        response = openai.Completion.create(
                            engine="text-davinci-003",
                            prompt=prompt,
                            max_tokens=500,
                            temperature=0.7,
                            n=1,
                            stop=None,
                        )

                        cover_letter = response.choices[0].text.strip()
                        st.success("Cover letter generated successfully!")
                        st.text_area("Generated Cover Letter:", value=cover_letter, height=300)

                        # Provide a download button
                        def convert_text_to_downloadable(text):
                            return text.encode('utf-8')

                        st.download_button(
                            label="Download Cover Letter",
                            data=convert_text_to_downloadable(cover_letter),
                            file_name='cover_letter.txt',
                            mime='text/plain',
                        )
                    except Exception as e:
                        st.error(f"An error occurred while generating the cover letter: {e}")
