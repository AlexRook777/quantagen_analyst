QuantaGen Analyst

Lightweight scaffold for the QuantaGen Analyst project as described in the project spec.

Structure:
- api/: FastAPI application, agents, services, tests and scripts.
- docker-compose.yml: starts api, postgres and chromadb services.

Quick start (development):
1. Create a virtualenv and install dev deps:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r api/requirements-dev.txt
2. Run tests:
   pytest
3. Run the app:
   uvicorn api.app.main:app --reload --port 8000

Docker (one-liner):
   docker-compose up --build

This repository is a scaffold. Many AI integrations are stubbed/dummy to avoid external API calls.
