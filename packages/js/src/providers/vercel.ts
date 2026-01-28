import type {
  Prompt,
  PromptContent,
  VercelMessage,
  VercelOptions,
} from '../types.js';

/**
 * Convert prompt content to Vercel AI SDK message format
 *
 * Vercel AI SDK uses simple string content for messages.
 * Multi-modal content is flattened to text.
 */
export function toVercel(
  content: PromptContent,
  options: VercelOptions = {}
): VercelMessage {
  const role = options.role ?? 'user';

  // String content - simple case
  if (typeof content === 'string') {
    return { role, content };
  }

  // Multi-modal - extract text content
  const text = content
    .filter((block) => block.type === 'text')
    .map((block) => (block as { type: 'text'; text: string }).text)
    .join('\n');

  return { role, content: text };
}

/**
 * Convert full prompt to Vercel AI SDK messages array
 */
export function promptToVercel(
  prompt: Prompt,
  options: VercelOptions = {}
): VercelMessage[] {
  return [toVercel(prompt.content, options)];
}

/**
 * Get prompt as CoreMessage format for Vercel AI SDK
 * This is compatible with the `generateText` and `streamText` functions
 */
export function toCoreMessages(
  prompt: Prompt,
  options: VercelOptions = {}
): Array<{ role: 'system' | 'user' | 'assistant'; content: string }> {
  return promptToVercel(prompt, options);
}
