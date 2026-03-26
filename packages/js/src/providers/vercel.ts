import type {
  Prompt,
  PromptContent,
  VercelMessage,
  VercelOptions,
  Message,
  ToolDefinition,
  VercelPromptResult,
  SkillDefinition,
} from '../types.js';

import { mergeToolsWithSkills } from './skills.js';

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

/**
 * Convert messages + tools to Vercel AI SDK prompt result format
 */
export function toVercelPromptResult(
  messages: Message[],
  tools: ToolDefinition[] | undefined,
  config: unknown,
  skills?: SkillDefinition[],
): VercelPromptResult {
  const vercelMessages: VercelPromptResult['messages'] = messages.map((msg) => {
    const text = msg.content
      .filter((b) => b.type === 'text')
      .map((b) => (b as { type: 'text'; text: string }).text)
      .join('\n');
    return { role: msg.role, content: text };
  });

  const result: VercelPromptResult = { messages: vercelMessages };

  const allTools = mergeToolsWithSkills(tools, skills);
  if (allTools && allTools.length > 0) {
    const toolsMap: Record<string, { description: string; parameters: Record<string, unknown> }> = {};
    for (const t of allTools) {
      toolsMap[t.function.name] = {
        description: t.function.description,
        parameters: t.function.parameters,
      };
    }
    result.tools = toolsMap;
  }

  return result;
}
