# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-28

### Added

- **JavaScript/TypeScript SDK**
  - `Echostash` client for connecting to any PLP-compliant server
  - `LoadedPrompt` class with fluent API for variable substitution
  - `PromptQuery` builder for fetching prompts with version support
  - Provider converters: OpenAI, Anthropic, Google/Gemini, Vercel AI SDK, LangChain
  - Full TypeScript type definitions
  - Config extractors for each provider

- **Python SDK**
  - `Echostash` client with context manager support
  - `LoadedPrompt` class with fluent API
  - `PromptQuery` builder for fetching prompts
  - Provider converters matching JS SDK
  - Full type hints
  - Support for Python 3.9+

### Features

- Connect to any PLP-compliant prompt library
- Fluent, chainable API: `es.prompt('id').vars({ name: 'Alice' }).openai()`
- Zero LLM dependencies - just returns the right format
- Multi-modal content support (text + images)
- Model config extraction from prompt metadata
- Variable substitution with customizable parameter symbols
