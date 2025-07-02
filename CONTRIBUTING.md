# Contributing to Family Contact Agent

Thank you for your interest in contributing to Family Contact Agent! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Code Style](#code-style)
- [Commit Messages](#commit-messages)
- [Review Process](#review-process)

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and considerate in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Set up the development environment** (see below)
4. **Create a feature branch** for your changes
5. **Make your changes** following the guidelines below
6. **Test your changes** thoroughly
7. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Go (for WhatsApp bridge components)

### Local Setup

1. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/family-contact-agent.git
   cd family-contact-agent
   ```

2. **Set up Python environment:**
   ```bash
   # Using uv (recommended) - One command setup
   make setup

   # Or manually:
   # Create virtual environment and sync dependencies
   uv sync
   # Install package in editable mode with dev dependencies
   uv pip install -e ".[dev]"
   # Install pre-commit hooks
   pre-commit install

   # Activate the virtual environment
   source .venv/bin/activate  # On Unix/macOS
   # .venv\Scripts\activate   # On Windows
   ```

3. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

### Dependency Management

This project uses `uv` for dependency management. Here are the key commands for development:

**Initial Setup:**
```bash
# Complete setup (recommended)
make setup

# Or step by step:
uv sync                    # Sync dependencies from lock file
uv pip install -e ".[dev]" # Install package in editable mode
```

**Adding/Removing Dependencies:**
```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name

# Update all dependencies
uv sync
```

**Common Development Commands:**
```bash
make sync           # Sync dependencies from lock file
make install-dev    # Install dev dependencies in editable mode
make test           # Run tests
make format         # Format code
```

## Making Changes

### Before You Start

1. **Check existing issues** to see if your change is already being worked on
2. **Create an issue** if you're planning a significant change
3. **Discuss the approach** with maintainers for major features

### Code Style

This project follows strict code style guidelines:

- **Black** for code formatting
- **Ruff** for linting and import sorting
- **Type hints** for all function parameters and return values
- **Docstrings** for all public functions and classes

### File Organization

- Keep related functionality together
- Use descriptive file and function names
- Follow the existing project structure
- Add tests for new functionality

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/test_specific_file.py

# Run tests with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Follow the existing test patterns
- Aim for good test coverage

### Manual Testing

Before submitting a PR, test your changes manually:

1. **Test the specific functionality** you changed
2. **Test related functionality** to ensure no regressions
3. **Test in different environments** if applicable
4. **Test edge cases** and error conditions

## Submitting Changes

### Pull Request Process

1. **Use the PR template** - it will be automatically loaded when you create a PR
2. **Fill out all sections** of the template
3. **Link related issues** using keywords like "Fixes #123"
4. **Provide clear descriptions** of your changes
5. **Include testing instructions** if needed

### PR Checklist

Before submitting, ensure:

- [ ] All pre-commit hooks pass
- [ ] All tests pass
- [ ] Code is properly formatted
- [ ] Documentation is updated
- [ ] No new warnings are introduced
- [ ] Changes are tested locally

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Test additions or changes
- `chore`: Maintenance tasks

**Examples:**
```
feat(translation): add support for new LLM provider

fix(whatsapp): resolve audio message download issue

docs(readme): update usage examples for new CLI modes
```

## Code Style

### Python Guidelines

- Use type hints for all function parameters and return values
- Follow PEP 8 style guidelines
- Use descriptive variable and function names
- Add docstrings for all public functions
- Keep functions focused and small
- Use meaningful comments for complex logic

### Import Organization

Imports should be organized as follows:

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import List, Optional

# Third-party imports
import numpy as np
import soundfile as sf

# Local imports
from translation_agent.libs.translation import Translation
from canto_nlp.tts.infer import OnnxInferenceSession
```

### Error Handling

- Use appropriate exception types
- Provide meaningful error messages
- Log errors with appropriate levels
- Handle edge cases gracefully

## Review Process

### What Reviewers Look For

- **Functionality**: Does the code work as intended?
- **Code quality**: Is the code clean, readable, and maintainable?
- **Testing**: Are there adequate tests?
- **Documentation**: Is the documentation updated?
- **Performance**: Are there any performance implications?
- **Security**: Are there any security concerns?

### Responding to Feedback

- **Be open to feedback** and suggestions
- **Address all comments** before requesting re-review
- **Ask questions** if feedback is unclear
- **Make incremental changes** if requested

## Getting Help

If you need help with your contribution:

1. **Check existing documentation** and issues
2. **Ask questions** in your PR or issue
3. **Join discussions** in existing issues
4. **Reach out to maintainers** if needed

## Recognition

All contributors will be recognized in the project's README and release notes. Significant contributions may be highlighted in the project documentation.

Thank you for contributing to Family Contact Agent! 🎉
