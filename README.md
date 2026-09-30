<p align="center">
  <img src="assets/banner.svg" alt="Cover Letter Generator" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/users-300%2B-0d0b24?style=for-the-badge&logoColor=ffb86b">
  <img src="https://img.shields.io/badge/LangChain-0d0b24?style=for-the-badge&logo=langchain&logoColor=ffb86b">
  <img src="https://img.shields.io/badge/Pinecone-0d0b24?style=for-the-badge&logoColor=ff5ca8">
  <img src="https://img.shields.io/badge/OpenAI-0d0b24?style=for-the-badge&logo=openai&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-0d0b24?style=for-the-badge&logo=streamlit&logoColor=ff4b4b">
</p>

Writing a good cover letter for every application takes an hour you don't have. This Streamlit app
finds real job postings, reads your resume, and drafts a cover letter written for the posting you
pick, not a generic template. It has been used by 300+ people.

Built by Team Symphony for a generative AI course at Santa Clara University.

## How it works

```mermaid
flowchart LR
    A[Job search query] -->|SerpApi| B[Live postings]
    B -->|OpenAI embeddings| C[(Pinecone index)]
    D[Your resume<br/>PDF · DOCX · TXT] --> E[Resume text]
    C --> F{Pick a posting}
    E --> G[GPT writes the letter]
    F --> G
    G --> H[Tailored cover letter]
```

1. **Search:** pulls live postings for any query ("Data Analyst in San Jose") through SerpApi.
2. **Embed:** stores each posting as an OpenAI embedding in a Pinecone index so they can be searched and matched.
3. **Read:** extracts the text of your resume, whether it's a PDF, a Word file, or plain text.
4. **Write:** GPT combines the posting and your resume into a letter that points to your actual experience.

## Run it

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Enter your own SerpApi, OpenAI, and Pinecone keys in the sidebar. They're kept only for your session
and never written to disk. To create the Pinecone index ahead of time, run
`python scripts/setup_pinecone.py` with `PINECONE_API_KEY` and `PINECONE_ENVIRONMENT` set.

## Repository layout

```
app/app.py                     Streamlit app
scripts/setup_pinecone.py      Creates the Pinecone index
scripts/setup_job_postings.py  Fetches and loads postings in bulk
requirements.txt
```
