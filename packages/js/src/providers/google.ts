import type {
  Prompt,
  PromptContent,
  GoogleMessage,
  GooglePart,
  GoogleOptions,
  ModelConfig,
} from '../types.js';

/**
 * Convert prompt content to Google/Gemini message format
 */
export function toGoogle(
  content: PromptContent,
  options: GoogleOptions = {}
): GoogleMessage {
  const role = options.role ?? 'user';

  // String content - simple case
  if (typeof content === 'string') {
    return { role, parts: [{ text: content }] };
  }

  // Multi-modal content
  const parts: GooglePart[] = content.map((block) => {
    if (block.type === 'text') {
      return { text: block.text };
    }
    if (block.type === 'image_url') {
      const url = block.image_url.url;

      // Handle base64 data URLs
      if (url.startsWith('data:')) {
        const match = url.match(/^data:([^;]+);base64,(.+)$/);
        if (match) {
          return {
            inlineData: {
              mimeType: match[1],
              data: match[2],
            },
          };
        }
      }

      // Google Gemini requires inline data, not URLs
      // For URLs, you'd need to fetch and convert to base64
      // We'll return as text with the URL for now
      return { text: `[Image: ${url}]` };
    }
    return { text: '' };
  });

  return { role, parts };
}

/**
 * Convert full prompt to Google messages array
 */
export function promptToGoogle(
  prompt: Prompt,
  options: GoogleOptions = {}
): GoogleMessage[] {
  return [toGoogle(prompt.content, options)];
}

/**
 * Extract Google-compatible model config from prompt metadata
 */
export function extractGoogleConfig(config?: ModelConfig): Record<string, unknown> {
  if (!config) return {};

  const result: Record<string, unknown> = {};
  const generationConfig: Record<string, unknown> = {};

  if (config.model) result.model = config.model;
  if (config.temperature !== undefined) generationConfig.temperature = config.temperature;
  if (config.topP !== undefined) generationConfig.topP = config.topP;
  if (config.topK !== undefined) generationConfig.topK = config.topK;
  if (config.maxTokens !== undefined) generationConfig.maxOutputTokens = config.maxTokens;
  if (config.stop !== undefined) generationConfig.stopSequences = Array.isArray(config.stop) ? config.stop : [config.stop];

  if (Object.keys(generationConfig).length > 0) {
    result.generationConfig = generationConfig;
  }

  return result;
}
