# Echostash SDK

**The universal prompt SDK.** Fetch prompts from any PLP-compliant library and use them with any AI provider.

## Why Echostash SDK?

- **Works with any PLP server** - Echostash Cloud, local servers, your own registry
- **Provider-agnostic** - OpenAI, Anthropic, Google, Vercel AI, LangChain, and more
- **Fluent API** - Beautiful, intuitive, chainable methods
- **Type-safe** - Full TypeScript and Python type support
- **Zero LLM dependencies** - We format, you call

## Installation

### JavaScript / TypeScript

```bash
npm install echostash
```

### Python

```bash
pip install echostash
```

## Quick Start

### JavaScript

```typescript
import { Echostash } from 'echostash';

const es = new Echostash('https://api.echostash.com', { apiKey: 'sk_...' });

// One-liner: fetch + substitute + convert
const message = await es.prompt('welcome').vars({ name: 'Alice' }).openai();

// Use with OpenAI
import OpenAI from 'openai';
const openai = new OpenAI();
await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [message],
});
```

### Python

```python
from echostash import Echostash

es = Echostash("https://api.echostash.com", api_key="sk_...")

# One-liner: fetch + substitute + convert
message = es.prompt("welcome").vars(name="Alice").openai()

# Use with OpenAI
from openai import OpenAI
client = OpenAI()
client.chat.completions.create(
    model="gpt-4",
    messages=[message]
)
```

## The Fluent API

```typescript
// Fetch a prompt
const prompt = await es.prompt('welcome-email').get();

// Substitute variables
const rendered = prompt.with({ name: 'Alice', company: 'Acme' });

// Convert to any provider
const openaiMsg = rendered.openai();
const anthropicMsg = rendered.anthropic();
const googleMsg = rendered.google();
const vercelMsg = rendered.vercel();
const langchainMsg = rendered.langchain();

// Or do it all in one line
const msg = await es.prompt('welcome').vars({ name: 'Alice' }).openai();
```

## Provider Support

| Provider | JavaScript | Python |
|----------|------------|--------|
| OpenAI | `.openai()` | `.openai()` |
| Anthropic | `.anthropic()` | `.anthropic()` |
| Google / Gemini | `.google()` / `.gemini()` | `.google()` / `.gemini()` |
| Vercel AI SDK | `.vercel()` | `.vercel()` |
| LangChain | `.langchain()` / `.langchainTemplate()` | `.langchain()` / `.langchain_template()` |

## Connect to Any PLP Server

The SDK implements the [Prompt Library Protocol (PLP)](https://github.com/GoReal-AI/plp), so it works with:

- **Echostash Cloud** - Our hosted prompt management platform
- **Local PLP servers** - Run your own with [plp-express](https://github.com/GoReal-AI/plp-express)
- **Any PLP-compliant registry** - The protocol is open and anyone can implement it

```typescript
// Echostash Cloud
const es = new Echostash('https://api.echostash.com', { apiKey: 'sk_...' });

// Your local server
const local = new Echostash('http://localhost:3000');

// Any PLP-compliant server
const custom = new Echostash('https://prompts.yourcompany.com');
```

## Packages

| Package | Language | npm/PyPI |
|---------|----------|----------|
| `echostash` | JavaScript/TypeScript | [![npm](https://img.shields.io/npm/v/echostash)](https://www.npmjs.com/package/echostash) |
| `echostash` | Python | [![PyPI](https://img.shields.io/pypi/v/echostash)](https://pypi.org/project/echostash/) |

## Documentation

- [JavaScript SDK Documentation](./packages/js/README.md)
- [Python SDK Documentation](./packages/python/README.md)
- [PLP Protocol Specification](https://github.com/GoReal-AI/plp)
- [Echostash Platform](https://echostash.com)

## License

MIT
