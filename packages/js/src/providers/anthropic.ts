import type {
  Prompt,
  PromptContent,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicOptions,
  ModelConfig,
  Message,
  ToolDefinition,
  AnthropicPromptResult,
  SkillDefinition,
} from '../types.js';

import { mergeToolsWithSkills } from './skills.js';

/**
 * Convert prompt content to Anthropic message format
 *
 * Note: Anthropic handles system messages separately via the `system` parameter,
 * not in the messages array. Use `asSystem: true` to get system content.
 */
export function toAnthropic(
  content: PromptContent,
  options: AnthropicOptions = {}
): AnthropicMessage {
  const role = options.role ?? 'user';

  // String content - simple case
  if (typeof content === 'string') {
    return { role, content };
  }

  // Multi-modal content
  const blocks: AnthropicContentBlock[] = content.map((block) => {
    if (block.type === 'text') {
      return { type: 'text', text: block.text };
    }
    if (block.type === 'image_url') {
      const url = block.image_url.url;

      // Handle base64 data URLs
      if (url.startsWith('data:')) {
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          return {
            type: 'image' as const,
            source: {
              type: 'base64' as const,
              media_type: match[1],
              data: match[2],
            },
          };
        }
      }

      // Handle regular URLs (Anthropic requires base64 for images)
      // Return as-is, user should handle URL fetching
      return {
        type: 'image' as const,
        source: {
          type: 'url' as const,
          url: url,
        },
      };
    }
    return { type: 'text', text: '' };
  });

  return { role, content: blocks };
}

/**
 * Get system message content for Anthropic
 * (Anthropic uses a separate `system` param, not in messages)
 */
export function toAnthropicSystem(content: PromptContent): string {
  if (typeof content === 'string') return content;

  return content
    .filter((block) => block.type === 'text')
    .map((block) => (block as { type: 'text'; text: string }).text)
    .join('\n');
}

/**
 * Convert full prompt to Anthropic format
 * Returns both system and messages for flexibility
 */
export function promptToAnthropic(
  prompt: Prompt,
  options: AnthropicOptions = {}
): { system?: string; messages: AnthropicMessage[] } {
  if (options.asSystem) {
    return {
      system: toAnthropicSystem(prompt.content),
      messages: [],
    };
  }

  return {
    messages: [toAnthropic(prompt.content, options)],
  };
}

/**
 * Extract Anthropic-compatible model config from prompt metadata
 */
export function extractAnthropicConfig(config?: ModelConfig): Record<string, unknown> {
  if (!config) return {};

  const result: Record<string, unknown> = {};

  if (config.model) result.model = config.model;
  if (config.temperature !== undefined) result.temperature = config.temperature;
  if (config.topP !== undefined) result.top_p = config.topP;
  if (config.topK !== undefined) result.top_k = config.topK;
  if (config.maxTokens !== undefined) result.max_tokens = config.maxTokens;
  if (config.stop !== undefined) result.stop_sequences = Array.isArray(config.stop) ? config.stop : [config.stop];

  return result;
}

/**
 * Convert messages + tools to Anthropic prompt result format
 */
export function toAnthropicPromptResult(
  messages: Message[],
  tools: ToolDefinition[] | undefined,
  config: ModelConfig | undefined,
  skills?: SkillDefinition[],
): AnthropicPromptResult {
  let system: string | undefined;
  const anthropicMessages: Array<{
    role: string;
    content: string | Array<{ type: string; [key: string]: unknown }>;
  }> = [];

  for (const msg of messages) {
    if (msg.role === 'system') {
      // Anthropic uses a separate system parameter
      system = msg.content
        .filter((b) => b.type === 'text')
        .map((b) => (b as { type: 'text'; text: string }).text)
        .join('\n');
      continue;
    }

    const content = msg.content;
    if (content.length === 1 && content[0].type === 'text') {
      anthropicMessages.push({
        role: msg.role,
        content: (content[0] as { type: 'text'; text: string }).text,
      });
    } else {
      anthropicMessages.push({
        role: msg.role,
        content: content.map((block) => {
          if (block.type === 'text') {
            return { type: 'text', text: (block as { type: 'text'; text: string }).text };
          }
          if (block.type === 'image_url') {
            const url = (block as { type: 'image_url'; image_url: { url: string } }).image_url.url;
            if (url.startsWith('data:')) {
              const match = url.match(/^data:([^;]+);base64,(.+)$/);
              if (match) {
                return {
                  type: 'image',
                  source: { type: 'base64', media_type: match[1], data: match[2] },
                };
              }
            }
            return { type: 'image', source: { type: 'url', url } };
          }
          return { type: 'text', text: '' };
        }),
      });
    }
  }

  const result: AnthropicPromptResult = {
    messages: anthropicMessages,
    ...extractAnthropicConfig(config),
  };

  if (system) {
    result.system = system;
  }

  const allTools = mergeToolsWithSkills(tools, skills);
  if (allTools && allTools.length > 0) {
    result.tools = allTools.map((t) => ({
      name: t.function.name,
      description: t.function.description,
      input_schema: t.function.parameters,
    }));
  }

  return result;
}
