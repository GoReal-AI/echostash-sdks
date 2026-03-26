import type {
  Prompt,
  PromptContent,
  ContentBlock,
  OpenAIMessage,
  OpenAIContentPart,
  OpenAIOptions,
  ModelConfig,
  Message,
  ToolDefinition,
  OpenAIPromptResult,
  SkillDefinition,
} from '../types.js';

import { mergeToolsWithSkills } from './skills.js';

/**
 * Convert prompt content to OpenAI message format
 */
export function toOpenAI(
  content: PromptContent,
  options: OpenAIOptions = {}
): OpenAIMessage {
  const role = options.role ?? 'user';

  // String content - simple case
  if (typeof content === 'string') {
    return { role, content };
  }

  // Multi-modal content
  const parts: OpenAIContentPart[] = content.map((block) => {
    if (block.type === 'text') {
      return { type: 'text', text: block.text };
    }
    if (block.type === 'image_url') {
      return {
        type: 'image_url',
        image_url: {
          url: block.image_url.url,
          detail: block.image_url.detail ?? 'auto',
        },
      };
    }
    // Fallback for unknown types
    return { type: 'text', text: '' };
  });

  return { role, content: parts };
}

/**
 * Convert full prompt to OpenAI messages array
 */
export function promptToOpenAI(
  prompt: Prompt,
  options: OpenAIOptions = {}
): OpenAIMessage[] {
  return [toOpenAI(prompt.content, options)];
}

/**
 * Extract OpenAI-compatible model config from prompt metadata
 */
export function extractOpenAIConfig(config?: ModelConfig): Record<string, unknown> {
  if (!config) return {};

  const result: Record<string, unknown> = {};

  if (config.model) result.model = config.model;
  if (config.temperature !== undefined) result.temperature = config.temperature;
  if (config.topP !== undefined) result.top_p = config.topP;
  if (config.maxTokens !== undefined) result.max_tokens = config.maxTokens;
  if (config.seed !== undefined) result.seed = config.seed;
  if (config.stop !== undefined) result.stop = config.stop;
  if (config.presencePenalty !== undefined) result.presence_penalty = config.presencePenalty;
  if (config.frequencyPenalty !== undefined) result.frequency_penalty = config.frequencyPenalty;

  return result;
}

/**
 * Check if content contains images
 */
export function hasImages(content: PromptContent): boolean {
  if (typeof content === 'string') return false;
  return content.some((block) => block.type === 'image_url');
}

/**
 * Convert messages + tools to OpenAI prompt result format
 */
export function toOpenAIPromptResult(
  messages: Message[],
  tools: ToolDefinition[] | undefined,
  config: ModelConfig | undefined,
  skills?: SkillDefinition[],
): OpenAIPromptResult {
  const result: OpenAIPromptResult = {
    messages: messages.map((msg) => {
      const content = msg.content;
      if (content.length === 1 && content[0].type === 'text') {
        return { role: msg.role, content: (content[0] as { type: 'text'; text: string }).text };
      }
      return {
        role: msg.role,
        content: content.map((block) => {
          if (block.type === 'text') {
            return { type: 'text', text: (block as { type: 'text'; text: string }).text };
          }
          if (block.type === 'image_url') {
            return {
              type: 'image_url',
              image_url: {
                url: (block as { type: 'image_url'; image_url: { url: string; detail?: string } }).image_url.url,
                detail: (block as { type: 'image_url'; image_url: { url: string; detail?: string } }).image_url.detail ?? 'auto',
              },
            };
          }
          return { type: 'text', text: '' };
        }),
      };
    }),
    ...extractOpenAIConfig(config),
  };

  const allTools = mergeToolsWithSkills(tools, skills);
  if (allTools && allTools.length > 0) {
    result.tools = allTools;
  }

  return result;
}
