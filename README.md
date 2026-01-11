# CCBA RAG System

[![CI](https://github.com/vvChu/RAG_QCTCVN/actions/workflows/ci.yml/badge.svg)](https://github.com/vvChu/RAG_QCTCVN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Vietnamese Construction Standards Expert System - RAG for QCVN/TCVN documents.

## Features

- **Hybrid Retrieval**: Dense + Sparse (BGE-M3) with RRF fusion
- **Vietnamese Optimized**: Structural chunking by Chương/Điều/Khoản
- **Multi-LLM**: Gemini (primary), Groq (fallback)
- **Production Ready**: Docker, CI/CD, comprehensive tests

## Quick Start

### Installation

```bash
# Clone
git clone https://github.com/your-username/RAG_QCTCVN.git
cd RAG_QCTCVN

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
# Start API + UI
.\start.bat

# Or manually:
python server.py          # API: http://localhost:8000
python -m streamlit run app.py  # UI: http://localhost:8501
```

### Docker

```bash
docker-compose up -d
```

## Project Structure

```
RAG_QCTCVN/
├── src/ccba_rag/     # Main package
│   ├── core/         # Settings, models, prompts
│   ├── ingestion/    # Document processing
│   ├── retrieval/    # Embeddings, search
│   └── generation/   # LLM generation
├── config/           # YAML configuration
├── entrypoint/       # CLI entry point
├── tests/            # Unit tests
└── docs/             # Documentation
```

## Configuration

| File | Purpose |
|------|---------|
| `config/default.yaml` | Model parameters |
| `config/prompts.yaml` | System prompts |
| `.env` | API keys |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | System status |
| `/query` | POST | RAG query |
| `/index` | POST | Index documents |
| `/documents` | GET | List documents |
| `/upload` | POST | Upload file |

## Testing

```bash
pytest tests/ -v
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md) before submitting PRs.

## License

MIT
