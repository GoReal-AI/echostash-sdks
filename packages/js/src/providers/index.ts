/**
 * Provider adapters for converting prompts to provider-specific formats
 *
 * Each provider module exports functions to convert prompt content
 * to the exact format that provider's SDK expects.
 */

// OpenAI
export {
  toOpenAI,
  promptToOpenAI,
  extractOpenAIConfig,
  hasImages,
} from './openai.js';

// Anthropic
export {
  toAnthropic,
  toAnthropicSystem,
  promptToAnthropic,
  extractAnthropicConfig,
} from './anthropic.js';

// Google / Gemini
export {
  toGoogle,
  promptToGoogle,
  extractGoogleConfig,
} from './google.js';

// Vercel AI SDK
export {
  toVercel,
  promptToVercel,
  toCoreMessages,
} from './vercel.js';

// LangChain
export {
  toLangChain,
  promptToLangChain,
  toLangChainTemplate,
  toChatPromptTemplate,
} from './langchain.js';
