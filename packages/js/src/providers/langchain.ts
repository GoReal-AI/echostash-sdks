import type {
  Prompt,
  PromptContent,
  LangChainMessage,
  LangChainMessageType,
  LangChainOptions,
} from '../types.js';

/**
 * Convert prompt content to LangChain message format
 *
 * LangChain uses 'human' instead of 'user' and 'ai' instead of 'assistant'
 */
export function toLangChain(
  content: PromptContent,
  options: LangChainOptions = {}
): LangChainMessage {
  const type = options.type ?? 'human';

  // String content - simple case
  if (typeof content === 'string') {
    return { type, content };
  }

  // Multi-modal - extract text content
  const text = content
    .filter((block) => block.type === 'text')
    .map((block) => (block as { type: 'text'; text: string }).text)
    .join('\n');

  return { type, content: text };
}

/**
 * Convert full prompt to LangChain messages array
 */
export function promptToLangChain(
  prompt: Prompt,
  options: LangChainOptions = {}
): LangChainMessage[] {
  return [toLangChain(prompt.content, options)];
}

/**
 * Get prompt as a LangChain PromptTemplate-compatible format
 * Returns the template string with {variable} syntax
 */
export function toLangChainTemplate(
  prompt: Prompt
): { template: string; inputVariables: string[] } {
  const content = typeof prompt.content === 'string'
    ? prompt.content
    : prompt.content
        .filter((block) => block.type === 'text')
        .map((block) => (block as { type: 'text'; text: string }).text)
        .join('\n');

  // Convert from {{var}} or other formats to {var}
  const paramSymbol = prompt.parameterSymbol ?? '{{}}';
  const mid = Math.floor(paramSymbol.length / 2);
  const prefix = paramSymbol.slice(0, mid);
  const suffix = paramSymbol.slice(mid);

  // Escape regex special chars
  const escapeRegex = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const prefixEscaped = escapeRegex(prefix);
  const suffixEscaped = escapeRegex(suffix);

  // Find all variables
  const regex = new RegExp(`${prefixEscaped}([\\w.]+)${suffixEscaped}`, 'g');
  const inputVariables: string[] = [];
  let match;
  while ((match = regex.exec(content)) !== null) {
    if (!inputVariables.includes(match[1])) {
      inputVariables.push(match[1]);
    }
  }

  // Convert to LangChain format {var}
  const template = content.replace(regex, '{$1}');

  return { template, inputVariables };
}

/**
 * Get prompt as a ChatPromptTemplate-compatible tuple format
 * Returns array of [role, content] tuples
 */
export function toChatPromptTemplate(
  prompt: Prompt,
  role: 'system' | 'human' | 'ai' = 'human'
): Array<[string, string]> {
  const { template } = toLangChainTemplate(prompt);
  return [[role, template]];
}
