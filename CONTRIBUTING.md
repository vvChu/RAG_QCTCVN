# Contributing to CCBA RAG System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/vvChu/RAG_QCTCVN.git
   cd RAG_QCTCVN
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Code Style

- **Linter**: [Ruff](https://github.com/astral-sh/ruff) with 100 character line limit
- **Type Hints**: Use Python type annotations
- **Docstrings**: Google style

Run linting:

```bash
ruff check src/
ruff format src/
```

## Testing

Run tests before submitting:

```bash
pytest tests/ -v
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes with clear, atomic commits
3. Ensure all tests pass and linting is clean
4. Update documentation if needed
5. Submit PR with a clear description

### Commit Message Convention

```
type(scope): brief description

- feat: New feature
- fix: Bug fix
- docs: Documentation
- refactor: Code refactoring
- test: Adding tests
- chore: Maintenance
```

Example: `feat(retrieval): add hybrid search with RRF fusion`

## Reporting Bugs

Use the bug report template and include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (Python version, OS)

## Feature Requests

Use the feature request template with:

- Problem statement
- Proposed solution
- Alternatives considered

## Questions?

Open a discussion or issue for any questions.
