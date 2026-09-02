# AI Agent / RAG Platform

FastAPI backend service for the AI Agent and RAG platform.

## Getting Started

### Prerequisites
- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)

### Installation & Environment
```bash
cp .env.example .env
uv sync
```

### Running the Application
```bash
uv run uvicorn app.main:app --reload
```

### Running Tests
```bash
uv run pytest
``

