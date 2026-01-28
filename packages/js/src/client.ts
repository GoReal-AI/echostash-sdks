import type {
  Prompt,
  PromptContent,
  PromptMeta,
  Variables,
  EchostashConfig,
  OpenAIMessage,
  AnthropicMessage,
  GoogleMessage,
  VercelMessage,
  LangChainMessage,
  OpenAIOptions,
  AnthropicOptions,
  GoogleOptions,
  VercelOptions,
  LangChainOptions,
  ContentBlock,
} from './types.js';

import {
  toOpenAI,
  extractOpenAIConfig,
  toAnthropic,
  toAnthropicSystem,
  extractAnthropicConfig,
  toGoogle,
  extractGoogleConfig,
  toVercel,
  toLangChain,
  toLangChainTemplate,
} from './providers/index.js';

// ============================================================================
// Variable Substitution
// ============================================================================

/**
 * Substitute variables in content using the parameter symbol
 */
function substituteVariables(
  content: PromptContent,
  variables: Variables,
  parameterSymbol: string
): PromptContent {
  if (Object.keys(variables).length === 0) return content;

  // Parse parameter symbol (e.g., "{{}}" -> prefix: "{{", suffix: "}}")
  const mid = Math.floor(parameterSymbol.length / 2);
  const prefix = parameterSymbol.slice(0, mid);
  const suffix = parameterSymbol.slice(mid);

  const escapeRegex = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const prefixEscaped = escapeRegex(prefix);
  const suffixEscaped = escapeRegex(suffix);

  const substitute = (text: string): string => {
    let result = text;
    for (const [key, value] of Object.entries(variables)) {
      const regex = new RegExp(`${prefixEscaped}${escapeRegex(key)}${suffixEscaped}`, 'g');
      result = result.replace(regex, value?.toString() ?? '');
    }
    return result;
  };

  if (typeof content === 'string') {
    return substitute(content);
  }

  return content.map((block) => {
    if (block.type === 'text') {
      return { ...block, text: substitute(block.text) };
    }
    return block;
  });
}

/**
 * Extract text content from prompt content
 */
function getTextContent(content: PromptContent): string {
  if (typeof content === 'string') return content;
  return content
    .filter((block) => block.type === 'text')
    .map((block) => (block as { type: 'text'; text: string }).text)
    .join('\n');
}

// ============================================================================
// Prompt Builder - The fluent API for working with fetched prompts
// ============================================================================

/**
 * A loaded prompt with fluent methods for variable substitution and provider conversion.
 *
 * @example
 * ```ts
 * const prompt = await es.prompt('welcome').get();
 *
 * // Substitute variables
 * const rendered = prompt.with({ name: 'Alice' });
 *
 * // Convert to provider formats
 * const openaiMsg = rendered.openai();
 * const anthropicMsg = rendered.anthropic();
 * ```
 */
export class LoadedPrompt {
  readonly id: string;
  readonly name?: string;
  readonly description?: string;
  readonly content: PromptContent;
  readonly meta: PromptMeta;
  readonly parameterSymbol: string;

  constructor(prompt: Prompt) {
    this.id = prompt.id;
    this.name = prompt.name;
    this.description = prompt.description;
    this.content = prompt.content;
    this.meta = prompt.meta;
    this.parameterSymbol = prompt.parameterSymbol ?? '{{}}';
  }

  // --------------------------------------------------------------------------
  // Variable Substitution
  // --------------------------------------------------------------------------

  /**
   * Substitute variables in the prompt content
   *
   * @example
   * ```ts
   * const rendered = prompt.with({ name: 'Alice', age: 30 });
   * ```
   */
  with(variables: Variables): LoadedPrompt {
    const newContent = substituteVariables(this.content, variables, this.parameterSymbol);
    return new LoadedPrompt({
      id: this.id,
      name: this.name,
      description: this.description,
      content: newContent,
      meta: this.meta,
      parameterSymbol: this.parameterSymbol,
    });
  }

  /**
   * Alias for `with()` - substitute variables
   */
  vars(variables: Variables): LoadedPrompt {
    return this.with(variables);
  }

  /**
   * Alias for `with()` - substitute variables
   */
  render(variables: Variables): LoadedPrompt {
    return this.with(variables);
  }

  // --------------------------------------------------------------------------
  // Content Access
  // --------------------------------------------------------------------------

  /**
   * Get the raw content
   */
  raw(): PromptContent {
    return this.content;
  }

  /**
   * Get content as plain text (extracts text from multi-modal content)
   */
  text(): string {
    return getTextContent(this.content);
  }

  /**
   * Get content as string (alias for text())
   */
  toString(): string {
    return this.text();
  }

  // --------------------------------------------------------------------------
  // Provider Conversions
  // --------------------------------------------------------------------------

  /**
   * Convert to OpenAI message format
   *
   * @example
   * ```ts
   * const message = prompt.openai({ role: 'system' });
   * // { role: 'system', content: '...' }
   *
   * await openai.chat.completions.create({
   *   model: 'gpt-4',
   *   messages: [message],
   * });
   * ```
   */
  openai(options?: OpenAIOptions): OpenAIMessage {
    return toOpenAI(this.content, options);
  }

  /**
   * Get OpenAI-compatible model config from prompt metadata
   */
  openaiConfig(): Record<string, unknown> {
    return extractOpenAIConfig(this.meta.modelConfig);
  }

  /**
   * Convert to Anthropic message format
   *
   * @example
   * ```ts
   * const message = prompt.anthropic();
   * // { role: 'user', content: '...' }
   *
   * await anthropic.messages.create({
   *   model: 'claude-3-opus-20240229',
   *   messages: [message],
   * });
   * ```
   */
  anthropic(options?: AnthropicOptions): AnthropicMessage {
    return toAnthropic(this.content, options);
  }

  /**
   * Get content as Anthropic system message (string)
   * Use this for the `system` parameter in Anthropic API
   */
  anthropicSystem(): string {
    return toAnthropicSystem(this.content);
  }

  /**
   * Get Anthropic-compatible model config from prompt metadata
   */
  anthropicConfig(): Record<string, unknown> {
    return extractAnthropicConfig(this.meta.modelConfig);
  }

  /**
   * Convert to Google/Gemini message format
   *
   * @example
   * ```ts
   * const message = prompt.google();
   * // { role: 'user', parts: [{ text: '...' }] }
   * ```
   */
  google(options?: GoogleOptions): GoogleMessage {
    return toGoogle(this.content, options);
  }

  /**
   * Alias for google()
   */
  gemini(options?: GoogleOptions): GoogleMessage {
    return this.google(options);
  }

  /**
   * Get Google-compatible model config from prompt metadata
   */
  googleConfig(): Record<string, unknown> {
    return extractGoogleConfig(this.meta.modelConfig);
  }

  /**
   * Convert to Vercel AI SDK message format
   *
   * @example
   * ```ts
   * const message = prompt.vercel({ role: 'system' });
   *
   * import { generateText } from 'ai';
   * await generateText({
   *   model: openai('gpt-4'),
   *   messages: [message],
   * });
   * ```
   */
  vercel(options?: VercelOptions): VercelMessage {
    return toVercel(this.content, options);
  }

  /**
   * Convert to LangChain message format
   *
   * @example
   * ```ts
   * const message = prompt.langchain({ type: 'system' });
   * // { type: 'system', content: '...' }
   * ```
   */
  langchain(options?: LangChainOptions): LangChainMessage {
    return toLangChain(this.content, options);
  }

  /**
   * Get as LangChain PromptTemplate-compatible format
   *
   * @example
   * ```ts
   * const { template, inputVariables } = prompt.langchainTemplate();
   * const promptTemplate = PromptTemplate.fromTemplate(template);
   * ```
   */
  langchainTemplate(): { template: string; inputVariables: string[] } {
    return toLangChainTemplate({
      id: this.id,
      content: this.content,
      meta: this.meta,
      parameterSymbol: this.parameterSymbol,
    });
  }

  // --------------------------------------------------------------------------
  // JSON serialization
  // --------------------------------------------------------------------------

  toJSON(): Prompt {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      content: this.content,
      meta: this.meta,
      parameterSymbol: this.parameterSymbol,
    };
  }
}

// ============================================================================
// Prompt Query Builder - Fluent API for fetching prompts
// ============================================================================

/**
 * A builder for fetching and configuring prompts.
 * Allows chaining of version selection and variable substitution before or after fetch.
 *
 * @example
 * ```ts
 * // Fetch latest version
 * const prompt = await es.prompt('welcome').get();
 *
 * // Fetch specific version
 * const prompt = await es.prompt('welcome').version('1.2.0').get();
 *
 * // Fetch and substitute variables in one chain
 * const msg = await es.prompt('welcome').vars({ name: 'Alice' }).openai();
 * ```
 */
export class PromptQuery {
  private readonly client: Echostash;
  private readonly promptId: string;
  private requestedVersion?: string;
  private pendingVariables?: Variables;

  constructor(client: Echostash, promptId: string) {
    this.client = client;
    this.promptId = promptId;
  }

  /**
   * Request a specific version of the prompt
   */
  version(version: string): PromptQuery {
    this.requestedVersion = version;
    return this;
  }

  /**
   * Alias for version()
   */
  v(version: string): PromptQuery {
    return this.version(version);
  }

  /**
   * Pre-set variables to substitute after fetching
   */
  vars(variables: Variables): PromptQuery {
    this.pendingVariables = variables;
    return this;
  }

  /**
   * Alias for vars()
   */
  with(variables: Variables): PromptQuery {
    return this.vars(variables);
  }

  /**
   * Fetch the prompt and return a LoadedPrompt
   */
  async get(): Promise<LoadedPrompt> {
    const prompt = await this.client.fetchPrompt(this.promptId, this.requestedVersion);
    let loaded = new LoadedPrompt(prompt);

    if (this.pendingVariables) {
      loaded = loaded.with(this.pendingVariables);
    }

    return loaded;
  }

  /**
   * Alias for get()
   */
  async fetch(): Promise<LoadedPrompt> {
    return this.get();
  }

  // --------------------------------------------------------------------------
  // Shorthand methods - fetch + convert in one call
  // --------------------------------------------------------------------------

  /**
   * Fetch prompt and convert to OpenAI format
   */
  async openai(options?: OpenAIOptions): Promise<OpenAIMessage> {
    const loaded = await this.get();
    return loaded.openai(options);
  }

  /**
   * Fetch prompt and convert to Anthropic format
   */
  async anthropic(options?: AnthropicOptions): Promise<AnthropicMessage> {
    const loaded = await this.get();
    return loaded.anthropic(options);
  }

  /**
   * Fetch prompt and convert to Google/Gemini format
   */
  async google(options?: GoogleOptions): Promise<GoogleMessage> {
    const loaded = await this.get();
    return loaded.google(options);
  }

  /**
   * Fetch prompt and convert to Vercel AI SDK format
   */
  async vercel(options?: VercelOptions): Promise<VercelMessage> {
    const loaded = await this.get();
    return loaded.vercel(options);
  }

  /**
   * Fetch prompt and convert to LangChain format
   */
  async langchain(options?: LangChainOptions): Promise<LangChainMessage> {
    const loaded = await this.get();
    return loaded.langchain(options);
  }

  /**
   * Fetch prompt and get as plain text
   */
  async text(): Promise<string> {
    const loaded = await this.get();
    return loaded.text();
  }
}

// ============================================================================
// Main Client
// ============================================================================

/**
 * The Echostash client - connects to any PLP-compliant prompt library.
 *
 * @example
 * ```ts
 * // Connect to Echostash Cloud
 * const es = new Echostash('https://api.echostash.com', {
 *   apiKey: 'sk_...'
 * });
 *
 * // Connect to a local PLP server
 * const local = new Echostash('http://localhost:3000');
 *
 * // Connect to any PLP-compliant library
 * const custom = new Echostash('https://prompts.mycompany.com');
 * ```
 */
export class Echostash {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly headers: Record<string, string>;
  private readonly timeout: number;
  private readonly defaultParameterSymbol: string;

  constructor(baseUrl: string, config: EchostashConfig = {}) {
    // Normalize base URL (remove trailing slash)
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.apiKey = config.apiKey;
    this.headers = config.headers ?? {};
    this.timeout = config.timeout ?? 10000;
    this.defaultParameterSymbol = config.defaultParameterSymbol ?? '{{}}';
  }

  /**
   * Start building a prompt query
   *
   * @example
   * ```ts
   * const prompt = await es.prompt('marketing/welcome-email').get();
   * ```
   */
  prompt(promptId: string): PromptQuery {
    return new PromptQuery(this, promptId);
  }

  /**
   * Alias for prompt()
   */
  get(promptId: string): PromptQuery {
    return this.prompt(promptId);
  }

  /**
   * Internal method to fetch a prompt from the server
   */
  async fetchPrompt(promptId: string, version?: string): Promise<Prompt> {
    const path = version
      ? `/v1/prompts/${encodeURIComponent(promptId)}/${encodeURIComponent(version)}`
      : `/v1/prompts/${encodeURIComponent(promptId)}`;

    const response = await this.request('GET', path);
    return this.normalizePrompt(response);
  }

  /**
   * Save a prompt to the server
   *
   * @example
   * ```ts
   * await es.save('marketing/new-prompt', {
   *   content: 'Hello {{name}}!',
   *   meta: { version: '1.0.0' }
   * });
   * ```
   */
  async save(
    promptId: string,
    input: { content: PromptContent; meta?: PromptMeta }
  ): Promise<LoadedPrompt> {
    const response = await this.request('PUT', `/v1/prompts/${encodeURIComponent(promptId)}`, input);
    return new LoadedPrompt(this.normalizePrompt(response));
  }

  /**
   * Delete a prompt from the server
   */
  async delete(promptId: string): Promise<void> {
    await this.request('DELETE', `/v1/prompts/${encodeURIComponent(promptId)}`);
  }

  /**
   * Check server capabilities via PLP discovery endpoint
   */
  async discover(): Promise<{
    plpVersion: string;
    server?: string;
    capabilities?: Record<string, boolean>;
  }> {
    try {
      const response = await this.request('GET', '/.well-known/plp');
      return {
        plpVersion: response.plp_version ?? '1.0',
        server: response.server,
        capabilities: response.capabilities,
      };
    } catch {
      // Discovery is optional in PLP
      return { plpVersion: '1.0' };
    }
  }

  // --------------------------------------------------------------------------
  // HTTP Request Helper
  // --------------------------------------------------------------------------

  private async request(method: string, path: string, body?: unknown): Promise<any> {
    const url = `${this.baseUrl}${path}`;

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      ...this.headers,
    };

    if (this.apiKey) {
      headers['Authorization'] = `Bearer ${this.apiKey}`;
      // Also support X-API-KEY for Echostash backend
      headers['X-API-KEY'] = this.apiKey;
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorBody = await response.json() as { error?: string; message?: string };
          errorMessage = errorBody.error || errorBody.message || errorMessage;
        } catch {
          // Ignore JSON parse errors
        }
        throw new EchostashError(errorMessage, response.status);
      }

      if (response.status === 204) {
        return null;
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // --------------------------------------------------------------------------
  // Normalize prompt from different server formats
  // --------------------------------------------------------------------------

  private normalizePrompt(data: any): Prompt {
    // Handle PLP format
    if (data.id && data.content !== undefined && data.meta !== undefined) {
      return {
        id: data.id,
        name: data.name,
        description: data.description ?? data.meta?.description,
        content: this.normalizeContent(data.content),
        meta: this.normalizeMeta(data.meta),
        parameterSymbol: data.parameterSymbol ?? this.defaultParameterSymbol,
      };
    }

    // Handle Echostash format (with promptMetaData)
    if (data.id && data.content !== undefined) {
      return {
        id: String(data.id),
        name: data.name,
        description: data.description,
        content: this.normalizeContent(data.content),
        meta: this.normalizeMeta(data.promptMetaData ?? data.meta ?? {}),
        parameterSymbol: data.parameterSymbol ?? this.defaultParameterSymbol,
      };
    }

    throw new EchostashError('Invalid prompt format received from server');
  }

  private normalizeContent(content: any): PromptContent {
    // Already a string
    if (typeof content === 'string') {
      return content;
    }

    // Array of content blocks
    if (Array.isArray(content)) {
      return content.map((block) => {
        if (block.type === 'text') {
          return { type: 'text', text: block.text };
        }
        if (block.type === 'image_url') {
          return {
            type: 'image_url',
            image_url: {
              url: block.image_url?.url ?? block.url,
              detail: block.image_url?.detail ?? block.detail,
            },
          };
        }
        // Fallback: treat as text
        return { type: 'text', text: String(block.text ?? block) };
      });
    }

    // Unknown format - convert to string
    return String(content);
  }

  private normalizeMeta(meta: any): PromptMeta {
    if (!meta) return {};

    return {
      version: meta.version,
      author: meta.author,
      description: meta.description,
      tokenCount: meta.tokenCount,
      modelConfig: meta.modelConfig ?? meta.model_config ?? meta.modelData,
      ...meta,
    };
  }
}

// ============================================================================
// Error Class
// ============================================================================

export class EchostashError extends Error {
  readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'EchostashError';
    this.statusCode = statusCode;
  }
}
