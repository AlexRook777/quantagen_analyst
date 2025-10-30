QuantaGen Analyst

Lightweight scaffold for the QuantaGen Analyst project as described in the project spec.

Structure:
- api/: FastAPI application, agents, services, tests and scripts.
- docker-compose.yml: starts api, postgres and chromadb services.

Quick start (development):
1. Create a virtualenv and install dev deps:
   python -m venv .venv
   venv\Scripts\activate
   pip install -r api/requirements-dev.txt
2. Run tests:
   pytest
3. Run the app:
   uvicorn api.app.main:app --reload --port 8000

Docker (one-liner):
   docker-compose up --build

This repository is a scaffold. Many AI integrations are stubbed/dummy to avoid external API calls.


----------------------------------------------------------------------------------------------------------
## Ingesting 10-K Filings

The `api/scripts/ingest.py` script is used to download and process SEC 10-K filings for a given ticker, chunk the text, and store the results in ChromaDB for retrieval by the RAG agent.

**Purpose:**  
- Downloads 10-K filings using `sec-edgar-downloader`.
- Extracts and splits text from each filing's `primary-document.html`.
- Embeds and stores the chunks in ChromaDB for semantic search.

**Usage:**
1. Activate your virtual environment and install dependencies.
2. Run the script:

By default, it will download the 5 latest 10-K filings for AAPL and ingest them.

**Customization:**  
- To ingest filings for a different ticker, modify the ticker argument in the script.
- The ingested data will be stored in the directory specified by the `CHROMA_PATH` environment variable.

This step is required before using the `/ask_filing` endpoint, so the RAG agent has access to the company's filings.
----------------------------------------------------------------------------------------------------------