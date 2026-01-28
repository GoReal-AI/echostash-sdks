/**
 * Echostash SDK - The universal prompt SDK
 *
 * Fetch prompts from any PLP-compliant library and convert them
 * to any AI provider's format.
 *
 * @example
 * ```ts
 * import { Echostash } from 'echostash';
 *
 * // Connect to any PLP-compliant server
 * const es = new Echostash('https://api.echostash.com', {
 *   apiKey: 'sk_...'
 * });
 *
 * // Fetch a prompt
 * const prompt = await es.prompt('welcome-email').get();
 *
 * // Use with OpenAI
 * const message = prompt.with({ name: 'Alice' }).openai();
 * await openai.chat.completions.create({
 *   model: 'gpt-4',
 *   messages: [message],
 * });
 *
 * // Use with Anthropic
 * const anthropicMsg = prompt.with({ name: 'Bob' }).anthropic();
 *
 * // Use with Vercel AI SDK
 * const vercelMsg = prompt.vars({ name: 'Charlie' }).vercel();
 *
 * // One-liner: fetch + substitute + convert
 * const msg = await es.prompt('welcome').vars({ name: 'Dave' }).openai();
 * ```
 *
 * @packageDocumentation
 */

// Main exports
export { Echostash, LoadedPrompt, PromptQuery, EchostashError } from './client.js';

// Type exports
export type {
  // Core types
  Prompt,
  PromptContent,
  PromptMeta,
  Variables,
  EchostashConfig,
  ModelConfig,
  ContentBlock,
  TextContent,
  ImageContent,

  // Provider message types
  OpenAIMessage,
  OpenAIContentPart,
  AnthropicMessage,
  AnthropicContentBlock,
  GoogleMessage,
  GooglePart,
  VercelMessage,
  LangChainMessage,
  LangChainMessageType,

  // Options
  ConversionOptions,
  OpenAIOptions,
  AnthropicOptions,
  GoogleOptions,
  VercelOptions,
  LangChainOptions,
} from './types.js';

// Provider converters (for advanced usage)
export * from './providers/index.js';

// Default export for convenience
export { Echostash as default } from './client.js';
