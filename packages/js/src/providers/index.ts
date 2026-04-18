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
  toOpenAIPromptResult,
} from './openai.js';

// Anthropic
export {
  toAnthropic,
  toAnthropicSystem,
  promptToAnthropic,
  extractAnthropicConfig,
  toAnthropicPromptResult,
} from './anthropic.js';

// Google / Gemini
export {
  toGoogle,
  promptToGoogle,
  extractGoogleConfig,
  toGooglePromptResult,
} from './google.js';

// Vercel AI SDK
export {
  toVercel,
  promptToVercel,
  toCoreMessages,
  toVercelPromptResult,
} from './vercel.js';

// LangChain
export {
  toLangChain,
  promptToLangChain,
  toLangChainTemplate,
  toChatPromptTemplate,
  toLangChainPromptResult,
} from './langchain.js';

// Skills
export { buildSkillLoadingTool, mergeToolsWithSkills } from './skills.js';
