# Contributing to GlassEye

Thank you for your interest in contributing to GlassEye!

## How to Contribute

1. **Fork the Repository**: Create your own feature branch from `main`.
2. **Local Setup**:
   ```bash
   make setup
   ```
3. **Run Backend & Frontend**:
   ```bash
   make backend
   make frontend
   ```
4. **Run Tests & Linting**:
   ```bash
   .venv/bin/ruff check .
   PYTHONPATH=backend .venv/bin/pytest backend/tests -q
   ```
5. **Submit a Pull Request**: Push your branch to your fork and submit a PR with a clear summary of your changes.

## Code Style & Guidelines
- Keep Python code formatted according to `ruff`.
- Maintain test coverage for new endpoints or features.
- Keep documentation up-to-date.
