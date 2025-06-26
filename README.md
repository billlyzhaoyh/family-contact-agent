# Family Contact Agent

A passion project to foster love and communication within families, especially when members communicate in different languages via voice notes. This tool helps bridge language gaps so everyone can stay connected. 💙

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Roadmap](#roadmap)
- [Setup Instructions](#setup-instructions)
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
<!-- Add detailed setup steps here -->

## Contributing
Contributions are welcome! Please open issues or submit pull requests to help improve the project.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements
- AWS Bedrock Translation Agent
- WhatsApp MCP
- Hugging Face and hon9kon9ize for Cantonese TTS/ASR resources
