import type {
  Prompt,
  PromptContent,
  GoogleMessage,
  GooglePart,
  GoogleOptions,
  ModelConfig,
  Message,
  ToolDefinition,
  GooglePromptResult,
  SkillDefinition,
} from '../types.js';

import { mergeToolsWithSkills } from './skills.js';

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

/**
 * Convert messages + tools to Google/Gemini prompt result format
 */
export function toGooglePromptResult(
  messages: Message[],
  tools: ToolDefinition[] | undefined,
  config: ModelConfig | undefined,
  skills?: SkillDefinition[],
): GooglePromptResult {
  const contents: GooglePromptResult['contents'] = messages.map((msg) => {
    const role = msg.role === 'assistant' ? 'model' : msg.role === 'system' ? 'user' : msg.role;
    const parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }> = [];

    for (const block of msg.content) {
      if (block.type === 'text') {
        parts.push({ text: (block as { type: 'text'; text: string }).text });
      } else if (block.type === 'image_url') {
        const url = (block as { type: 'image_url'; image_url: { url: string } }).image_url.url;
        if (url.startsWith('data:')) {
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
            continue;
          }
        }
        parts.push({ text: `[Image: ${url}]` });
      }
    }

    return { role, parts };
  });

  const result: GooglePromptResult = { contents };

  const googleConfig = extractGoogleConfig(config);
  if (googleConfig.generationConfig) {
    result.generationConfig = googleConfig.generationConfig as Record<string, unknown>;
  }

  const allTools = mergeToolsWithSkills(tools, skills);
  if (allTools && allTools.length > 0) {
    result.tools = [{
      functionDeclarations: allTools.map((t) => ({
        name: t.function.name,
        description: t.function.description,
        parameters: t.function.parameters,
      })),
    }];
  }

  return result;
}
