# Echostash SDK for JavaScript/TypeScript

The universal prompt SDK. Fetch prompts from any PLP-compliant library and use them with any AI provider.

## Installation

```bash
npm install echostash
# or
pnpm add echostash
# or
yarn add echostash
```

## Quick Start

```typescript
import { Echostash } from 'echostash';
import OpenAI from 'openai';

// Connect to any PLP-compliant server
const es = new Echostash('https://api.echostash.com', {
  apiKey: 'sk_...'
});

// Fetch a prompt and use with OpenAI
const prompt = await es.prompt('welcome-email').get();
const message = prompt.with({ name: 'Alice' }).openai();

const openai = new OpenAI();
const response = await openai.chat.completions.create({
  model: 'gpt-4',
  messages: [message],
});
```

## The Fluent API

The SDK is designed to be intuitive and chainable:

```typescript
// Fetch + substitute + convert in one line
const msg = await es.prompt('welcome').vars({ name: 'Alice' }).openai();

// Or step by step
const prompt = await es.prompt('welcome').get();
const rendered = prompt.with({ name: 'Alice' });
const message = rendered.openai();

// Fetch a specific version
const v1 = await es.prompt('welcome').version('1.0.0').get();

// Short version alias
const v2 = await es.prompt('welcome').v('2.0.0').get();
```

## Provider Support

### OpenAI

```typescript
const message = prompt.with({ name: 'Alice' }).openai({ role: 'system' });
const config = prompt.openaiConfig();

await openai.chat.completions.create({
  ...config,
  messages: [message],
});
```

### Anthropic

```typescript
import Anthropic from '@anthropic-ai/sdk';

// As user message
const message = prompt.with({ name: 'Bob' }).anthropic();

// As system message (Anthropic handles system separately)
const system = prompt.anthropicSystem();
const config = prompt.anthropicConfig();

const anthropic = new Anthropic();
await anthropic.messages.create({
  ...config,
  system,
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

### Google / Gemini

```typescript
const message = prompt.google({ role: 'user' });
// or
const message = prompt.gemini();

const config = prompt.googleConfig();
```

### Vercel AI SDK

```typescript
import { generateText } from 'ai';
import { openai } from '@ai-sdk/openai';

const message = prompt.vercel({ role: 'system' });

await generateText({
  model: openai('gpt-4'),
  messages: [message],
});
```

### LangChain

```typescript
import { PromptTemplate } from '@langchain/core/prompts';

// As message
const message = prompt.langchain({ type: 'human' });

// As template (with input variables)
const { template, inputVariables } = prompt.langchainTemplate();
const promptTemplate = PromptTemplate.fromTemplate(template);
```

## Connect to Any PLP Server

The SDK works with any PLP-compliant prompt library:

```typescript
// Echostash Cloud
const es = new Echostash('https://api.echostash.com', { apiKey: 'sk_...' });

// Local PLP server
const local = new Echostash('http://localhost:3000');

// Your company's prompt registry
const corp = new Echostash('https://prompts.mycompany.com', {
  apiKey: process.env.PROMPT_API_KEY,
});

// Any PLP-compliant service
const other = new Echostash('https://plp.example.com');
```

## Working with Prompts

### Fetch Prompts

```typescript
// Latest version
const prompt = await es.prompt('marketing/welcome').get();

// Specific version
const v1 = await es.prompt('marketing/welcome').version('1.0.0').get();

// With variables pre-set
const prompt = await es.prompt('welcome').vars({ name: 'Alice' }).get();
```

### Access Content

```typescript
const prompt = await es.prompt('welcome').get();

// Raw content (string or ContentBlock[])
const raw = prompt.raw();

// As plain text
const text = prompt.text();

// As string (same as text)
console.log(String(prompt));
```

### Save Prompts

```typescript
await es.save('my-new-prompt', {
  content: 'Hello {{name}}!',
  meta: {
    version: '1.0.0',
    author: 'me',
  },
});
```

### Delete Prompts

```typescript
await es.delete('my-old-prompt');
```

### Server Discovery

```typescript
const info = await es.discover();
console.log(info.plpVersion);    // "1.0"
console.log(info.capabilities);  // { versioning: true, ... }
```

## Type Safety

Full TypeScript support with all types exported:

```typescript
import type {
  Prompt,
  PromptContent,
  PromptMeta,
  ModelConfig,
  OpenAIMessage,
  AnthropicMessage,
  GoogleMessage,
  VercelMessage,
  LangChainMessage,
} from 'echostash';
```

## Advanced: Direct Provider Functions

For advanced use cases, you can import provider functions directly:

```typescript
import {
  toOpenAI,
  toAnthropic,
  toGoogle,
  toVercel,
  toLangChain,
  extractOpenAIConfig,
} from 'echostash/providers';

const content = 'Hello {{name}}!';
const message = toOpenAI(content, { role: 'system' });
```

## Configuration

```typescript
const es = new Echostash('https://api.echostash.com', {
  // API key for authentication
  apiKey: 'sk_...',

  // Custom headers
  headers: {
    'X-Custom-Header': 'value',
  },

  // Request timeout (ms)
  timeout: 10000,

  // Default parameter symbol for variable substitution
  defaultParameterSymbol: '{{}}',
});
```

## Error Handling

```typescript
import { EchostashError } from 'echostash';

try {
  const prompt = await es.prompt('not-found').get();
} catch (error) {
  if (error instanceof EchostashError) {
    console.error(error.message);     // "HTTP 404"
    console.error(error.statusCode);  // 404
  }
}
```

## License

MIT
