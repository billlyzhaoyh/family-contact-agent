# Family Contact Agent

A passion project to foster love and communication within families, especially when members communicate in different languages via voice notes. This tool helps bridge language gaps so everyone can stay connected. 💙

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [Setup Instructions](#setup-instructions)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview
Some family members often communicate in voice notes, but not everyone speaks the same language. This project brings together open source tools to translate, transcribe, and deliver messages so everyone can participate in the conversation.

## Features
- Translate voice notes between languages using LiteLLM (supports multiple LLM providers)
- Text-to-speech (TTS) and automatic speech recognition (ASR) for Cantonese
- WhatsApp integration for seamless message delivery
- Multi-provider LLM support (OpenAI, Anthropic, Azure, AWS Bedrock, Cohere, Google, Mistral)
- Command-line interface with multiple operation modes

## Architecture
This project integrates several open source components:
- **LiteLLM Translation Agent**: Multi-provider LLM integration for translation tasks
- **WhatsApp MCP Server**: Message delivery and management
- **Canto TTS components**: From [hon9kon9ize](https://huggingface.co/hon9kon9ize) and [Hugging Face Spaces](https://huggingface.co/spaces/hon9kon9ize/tts/tree/main)
- **Canto ASR components**: From [alvanlii](https://huggingface.co/alvanlii/whisper-small-cantonese)

### LLM Provider Support
The system supports multiple LLM providers through LiteLLM:
- **OpenAI** (default): GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo
- **Anthropic**: Claude 3.5 Sonnet, Claude 3.5 Haiku, Claude 3 Opus/Sonnet/Haiku
- **Azure OpenAI**: Azure-hosted GPT models
- **AWS Bedrock**: Claude models via Bedrock
- **Cohere**: Command and Command Light models
- **Google**: Gemini Pro and Gemini Flash
- **Mistral**: Mistral Large, Medium, and Small models

## Usage

The Family Contact Agent supports three operation modes:

### 1. Send Mode
Send text messages as audio to a specific contact.

```bash
python main.py --mode send --text "Hello, how are you?" --phone "+1234567890" --name "John Doe"
```

**Required arguments:**
- `--text`: The text message to send (will be translated to Cantonese and converted to audio)
- `--phone`: The contact's phone number
- `--name`: The contact's name (used to search for the contact)

### 2. Receive Mode
Receive and process the latest audio message from a contact.

```bash
python main.py --mode receive --phone "+1234567890"
```

**Required arguments:**
- `--phone`: The contact's phone number

**What it does:**
- Downloads the latest audio message from the contact
- Converts it to MP3 format
- Plays the audio
- Transcribes the Cantonese audio to text
- Translates the transcribed text to English
- Displays both the original transcription and translation

### 3. Interactive Mode
Interactive mode that prompts you to choose between send and receive operations.

```bash
python main.py --mode interactive --phone "+1234567890" --name "John Doe"
```

**Optional arguments:**
- `--phone`: The contact's phone number (used for both send and receive operations)
- `--name`: The contact's name (used for send operations)

**Interactive prompts:**
1. Choose operation (1 for send, 2 for receive)
2. If sending: Enter the text message
3. If sending: Confirm before sending the audio message

### Common Options
All modes support these optional arguments:
- `--verbose` or `-v`: Enable verbose logging (DEBUG level)
- `--help` or `-h`: Show help message

### Examples

**Send a quick message:**
```bash
python main.py --mode send --text "I love you!" --phone "+1234567890" --name "Mom"
```

**Check for new messages:**
```bash
python main.py --mode receive --phone "+1234567890"
```

**Interactive session:**
```bash
python main.py --mode interactive --phone "+1234567890" --name "Dad"
```

**Verbose logging:**
```bash
python main.py --mode send --text "Hello" --phone "+1234567890" --name "John" --verbose
```

## Roadmap
- [x] Migrate from bedrock model to litellm backend for flexible model selection
- [x] Add Cantonese ASR so voice notes can be transcribed and translated back to English
- [ ] Set up an agent to handle the different commands rather than using main.py with cli inputs
- [ ] Integrate WhatsApp MCP server into the main agent for prompt-based message sending/receiving
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

# Create a virtual environment and install dependencies
uv sync  # This creates .venv and installs all dependencies from uv.lock

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# .venv\Scripts\activate   # On Windows

# Install the package in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

**Alternative: Use the Makefile for setup**
```sh
# This will check dependencies and set up the environment
make setup

# Then install in editable mode
make install-dev
```

### 2. Configure LLM Provider (OpenAI - Default)
Create a `.env` file in the project root with your API key:

```sh
# OpenAI (default provider)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Custom OpenAI base URL (for Azure OpenAI or other compatible endpoints)
# OPENAI_API_BASE=https://your-custom-endpoint.com/v1

# Optional: OpenAI organization ID
# OPENAI_ORGANIZATION=your_organization_id
```

#### Alternative Providers
You can use other providers by setting the appropriate environment variables:

**Anthropic:**
```sh
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Azure OpenAI:**
```sh
AZURE_API_KEY=your_azure_api_key_here
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
```

**AWS Bedrock:**
```sh
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
```

**Cohere:**
```sh
COHERE_API_KEY=your_cohere_api_key_here
```

**Google (Vertex AI):**
```sh
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_PROJECT_ID=your_project_id
```

**Mistral:**
```sh
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 3. Set up WhatsApp MCP Bridge
```sh
cd whatsapp_mcp/whatsapp-bridge
# Run the Go server (first time will show a QR code to scan in WhatsApp)
go run main.go
```
You may need to re-authenticate every month.

### 4. Download Model Files
Model files will be downloaded automatically on first run. Make sure you have internet access.

### 5. Run the Main Application
The application supports multiple modes. Here are some examples:

**Check for new messages from a contact:**
```sh
python main.py --mode receive --phone "+1234567890"
```

**Send a message to a contact:**
```sh
python main.py --mode send --text "Hello, how are you?" --phone "+1234567890" --name "John Doe"
```

**Interactive mode:**
```sh
python main.py --mode interactive --phone "+1234567890" --name "John Doe"
```

For more details on all available options, see the [Usage](#usage) section above.

## Development

### Dependency Management
This project uses `uv` for dependency management. Here are the key commands:

**Initial Setup:**
```sh
# Install all dependencies from lock file
uv sync

# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

**Adding/Removing Dependencies:**
```sh
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Remove a dependency
uv remove package-name

# Update dependencies
uv sync
```

**Running Scripts:**
```sh
# Run a script with proper dependencies
uv run script.py
```

**Quick Reference:**
```sh
make setup        # Complete development setup
make sync         # Sync dependencies from lock file
make install-dev  # Install dev dependencies in editable mode
make test         # Run tests
make format       # Format code
make clean        # Clean up generated files
```

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
- Check code style and sort imports with Ruff

### Code Formatting
The project uses Black for code formatting and isort for import sorting. These are configured in `pyproject.toml` and will run automatically via pre-commit hooks.

### Testing
Run tests with pytest:
```sh
make test
```

## Contributing
Contributions are welcome! Please open issues or submit pull requests to help improve the project.

Before submitting a pull request, please ensure:
1. All pre-commit hooks pass
2. Code is properly formatted
3. Tests pass (if applicable)

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- LiteLLM for multi-provider LLM integration
- WhatsApp MCP for message delivery
- Hugging Face and hon9kon9ize and alvanlii for Cantonese TTS/ASR resources
