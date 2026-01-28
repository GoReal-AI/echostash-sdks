import type {
  Prompt,
  PromptContent,
  AnthropicMessage,
  AnthropicContentBlock,
  AnthropicOptions,
  ModelConfig,
} from '../types.js';

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
