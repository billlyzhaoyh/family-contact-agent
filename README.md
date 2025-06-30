# Family Contact Agent

A passion project to foster love and communication within families, especially when members communicate in different languages via voice notes. This tool helps bridge language gaps so everyone can stay connected. 💙

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Setup Instructions](#setup-instructions)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
Some family members often communicate in voice notes, but not everyone speaks the same language. This project brings together open source tools to translate, transcribe, and deliver messages so everyone can participate in the conversation.

## Features
- Translate voice notes between languages
- Text-to-speech (TTS) and automatic speech recognition (ASR) for Cantonese
- WhatsApp integration for seamless message delivery

## Architecture
This project integrates several open source components:
- [Bedrock Translation Agent](https://github.com/aws-samples/bedrock-translation-agent/tree/main)
- [WhatsApp MCP Server](https://github.com/lharries/whatsapp-mcp)
- Canto TTS components from [hon9kon9ize](https://huggingface.co/hon9kon9ize) and [Hugging Face Spaces](https://huggingface.co/spaces/hon9kon9ize/tts/tree/main)

## Roadmap
- [ ] Add Cantonese ASR so voice notes can be transcribed and translated back to English
- [ ] Integrate WhatsApp MCP server into the main agent for prompt-based message sending/receiving
- [ ] Improve error handling and user experience
- [ ] Add web UI for easier interaction

## Setup Instructions

### Prerequisites
- Python 3.11 (see `.python-version`)
- [uv](https://github.com/astral-sh/uv) (a fast Python package manager)
- Go (for WhatsApp bridge)

### 1. Set up Python environment with uv
```sh
# Install uv if you don't have it
pip install uv

# Create a virtual environment (optional but recommended)
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -e ".[dev]"  # Install with dev dependencies
```

### 2. Set up WhatsApp MCP Bridge
```sh
cd whatsapp_mcp/whatsapp-bridge
# Run the Go server (first time will show a QR code to scan in WhatsApp)
go run main.go
```
You may need to re-authenticate every month.

### 3. Download Model Files
Model files will be downloaded automatically on first run. Make sure you have internet access.

### 4. Run the Main Application
```sh
python main.py
```

## Development

### Pre-commit Hooks
This project uses pre-commit hooks to ensure code quality. After installing dev dependencies, set up pre-commit:

```sh
# Install pre-commit hooks
pre-commit install

# Run pre-commit on all files (optional)
pre-commit run --all-files
```

The pre-commit hooks will automatically:
- Format code with Black
- Sort imports with isort
- Check code style with flake8

### Code Formatting
The project uses Black for code formatting and isort for import sorting. These are configured in `pyproject.toml` and will run automatically via pre-commit hooks.

## Contributing
Contributions are welcome! Please open issues or submit pull requests to help improve the project.

Before submitting a pull request, please ensure:
1. All pre-commit hooks pass
2. Code is properly formatted
3. Tests pass (if applicable)

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- AWS Bedrock Translation Agent
- WhatsApp MCP
- Hugging Face and hon9kon9ize for Cantonese TTS/ASR resources
