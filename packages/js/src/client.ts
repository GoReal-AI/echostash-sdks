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
  VersionSpecifier,
  RenderResponse,
  BatchRenderItem,
  BatchRenderResponse,
  Message,
  ToolDefinition,
  OpenAIPromptResult,
  AnthropicPromptResult,
  GooglePromptResult,
  VercelPromptResult,
  LangChainPromptResult,
  ModelConfig,
  RenderResult,
  ObservationItem,
} from './types.js';

import {
  toOpenAI,
  toOpenAIPromptResult,
  extractOpenAIConfig,
  toAnthropic,
  toAnthropicSystem,
  toAnthropicPromptResult,
  extractAnthropicConfig,
  toGoogle,
  toGooglePromptResult,
  extractGoogleConfig,
  toVercel,
  toVercelPromptResult,
  toLangChain,
  toLangChainPromptResult,
  toLangChainTemplate,
} from './providers/index.js';

// ============================================================================
// Helpers
// ============================================================================

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseRetryAfter(header: string | null): number | null {
  if (!header) return null;
  const seconds = Number(header);
  if (!Number.isNaN(seconds) && seconds >= 0) return seconds;
  const date = Date.parse(header);
  if (!Number.isNaN(date)) {
    const delta = Math.max(0, Math.ceil((date - Date.now()) / 1000));
    return delta;
  }
  return null;
}

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

/**
 * Substitute variables in message content blocks
 */
function substituteMessagesContent(
  blocks: ContentBlock[],
  variables: Variables,
  parameterSymbol: string
): ContentBlock[] {
  if (Object.keys(variables).length === 0) return blocks;

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

  return blocks.map((block) => {
    if (block.type === 'text') {
      return { ...block, text: substitute(block.text) };
    }
    return block;
  });
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
  readonly messages: Message[];
  readonly tools: ToolDefinition[];
  /** Meta config from rendered meta template (server-side render result) */
  readonly renderMeta: Record<string, unknown>;

  constructor(prompt: Prompt & { renderMeta?: Record<string, unknown> }) {
    this.id = prompt.id;
    this.name = prompt.name;
    this.description = prompt.description;
    this.content = prompt.content;
    this.meta = prompt.meta;
    this.parameterSymbol = prompt.parameterSymbol ?? '{{}}';
    this.tools = prompt.tools ?? [];
    this.messages = this.normalizeMessages(prompt);
    this.renderMeta = prompt.renderMeta ?? {};
  }

  /**
   * Normalize messages from the prompt data.
   * If the server returned messages, use them. Otherwise, derive from content.
   */
  private normalizeMessages(prompt: Prompt): Message[] {
    // If server returned messages, use them
    if (prompt.messages && prompt.messages.length > 0) {
      return prompt.messages;
    }
    // Legacy: wrap content in a single user message
    const content = prompt.content;
    if (typeof content === 'string') {
      return [{ role: 'user', content: [{ type: 'text', text: content }] }];
    }
    if (Array.isArray(content)) {
      return [{ role: 'user', content: content as ContentBlock[] }];
    }
    return [{ role: 'user', content: [] }];
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

    // Substitute variables in messages too
    const newMessages = this.messages.map((msg) => ({
      ...msg,
      content: substituteMessagesContent(msg.content, variables, this.parameterSymbol),
    }));

    return new LoadedPrompt({
      id: this.id,
      name: this.name,
      description: this.description,
      content: newContent,
      meta: this.meta,
      parameterSymbol: this.parameterSymbol,
      messages: newMessages,
      tools: this.tools,
      renderMeta: this.renderMeta,
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
  // Provider Conversions (Messages + Tools)
  // --------------------------------------------------------------------------

  /**
   * Get the effective model config by merging prompt meta with rendered meta template.
   * Rendered meta (from server-side meta template) takes precedence over prompt-level modelConfig.
   */
  private getEffectiveModelConfig(): ModelConfig | undefined {
    const base = this.meta.modelConfig;
    if (!this.renderMeta || Object.keys(this.renderMeta).length === 0) {
      return base;
    }
    // Merge: renderMeta overrides base modelConfig
    return { ...base, ...this.renderMeta } as ModelConfig;
  }

  /**
   * Convert to OpenAI prompt result format with messages array + tools.
   *
   * When called with no arguments, returns the full prompt result with
   * messages, tools, and model config (including rendered meta template overrides).
   *
   * When called with OpenAIOptions (backward compatible), returns a single
   * OpenAI message for legacy usage.
   *
   * @example
   * ```ts
   * // New: full prompt result with messages + tools
   * const result = prompt.openai();
   * await openai.chat.completions.create(result);
   *
   * // Legacy: single message
   * const message = prompt.openai({ role: 'system' });
   * await openai.chat.completions.create({
   *   model: 'gpt-4',
   *   messages: [message],
   * });
   * ```
   */
  openai(): OpenAIPromptResult;
  openai(options: OpenAIOptions): OpenAIMessage;
  openai(options?: OpenAIOptions): OpenAIPromptResult | OpenAIMessage {
    if (options && Object.keys(options).length > 0) {
      return toOpenAI(this.content, options);
    }
    return toOpenAIPromptResult(this.messages, this.tools, this.getEffectiveModelConfig());
  }

  /**
   * Get OpenAI-compatible model config from prompt metadata
   */
  openaiConfig(): Record<string, unknown> {
    return extractOpenAIConfig(this.meta.modelConfig);
  }

  /**
   * Convert to Anthropic prompt result format with system + messages + tools.
   *
   * When called with no arguments, returns the full prompt result.
   * When called with AnthropicOptions (backward compatible), returns a single message.
   *
   * @example
   * ```ts
   * // New: full prompt result with system + messages + tools
   * const result = prompt.anthropic();
   * await anthropic.messages.create(result);
   *
   * // Legacy: single message
   * const message = prompt.anthropic({ role: 'user' });
   * ```
   */
  anthropic(): AnthropicPromptResult;
  anthropic(options: AnthropicOptions): AnthropicMessage;
  anthropic(options?: AnthropicOptions): AnthropicPromptResult | AnthropicMessage {
    if (options && Object.keys(options).length > 0) {
      return toAnthropic(this.content, options);
    }
    return toAnthropicPromptResult(this.messages, this.tools, this.getEffectiveModelConfig());
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
   * Convert to Google/Gemini prompt result format with contents + tools.
   *
   * When called with no arguments, returns the full prompt result.
   * When called with GoogleOptions (backward compatible), returns a single message.
   *
   * @example
   * ```ts
   * // New: full prompt result with contents + tools
   * const result = prompt.google();
   *
   * // Legacy: single message
   * const message = prompt.google({ role: 'user' });
   * ```
   */
  google(): GooglePromptResult;
  google(options: GoogleOptions): GoogleMessage;
  google(options?: GoogleOptions): GooglePromptResult | GoogleMessage {
    if (options && Object.keys(options).length > 0) {
      return toGoogle(this.content, options);
    }
    return toGooglePromptResult(this.messages, this.tools, this.getEffectiveModelConfig());
  }

  /**
   * Alias for google()
   */
  gemini(): GooglePromptResult;
  gemini(options: GoogleOptions): GoogleMessage;
  gemini(options?: GoogleOptions): GooglePromptResult | GoogleMessage {
    return (this.google as (options?: GoogleOptions) => GooglePromptResult | GoogleMessage)(options);
  }

  /**
   * Get Google-compatible model config from prompt metadata
   */
  googleConfig(): Record<string, unknown> {
    return extractGoogleConfig(this.meta.modelConfig);
  }

  /**
   * Convert to Vercel AI SDK prompt result format with messages + tools.
   *
   * When called with no arguments, returns the full prompt result.
   * When called with VercelOptions (backward compatible), returns a single message.
   *
   * @example
   * ```ts
   * // New: full prompt result with messages + tools
   * const result = prompt.vercel();
   *
   * // Legacy: single message
   * const message = prompt.vercel({ role: 'system' });
   * ```
   */
  vercel(): VercelPromptResult;
  vercel(options: VercelOptions): VercelMessage;
  vercel(options?: VercelOptions): VercelPromptResult | VercelMessage {
    if (options && Object.keys(options).length > 0) {
      return toVercel(this.content, options);
    }
    return toVercelPromptResult(this.messages, this.tools, this.getEffectiveModelConfig());
  }

  /**
   * Convert to LangChain prompt result format with messages + tools.
   *
   * When called with no arguments, returns the full prompt result.
   * When called with LangChainOptions (backward compatible), returns a single message.
   *
   * @example
   * ```ts
   * // New: full prompt result with messages + tools
   * const result = prompt.langchain();
   *
   * // Legacy: single message
   * const message = prompt.langchain({ type: 'system' });
   * ```
   */
  langchain(): LangChainPromptResult;
  langchain(options: LangChainOptions): LangChainMessage;
  langchain(options?: LangChainOptions): LangChainPromptResult | LangChainMessage {
    if (options && Object.keys(options).length > 0) {
      return toLangChain(this.content, options);
    }
    return toLangChainPromptResult(this.messages, this.tools, this.getEffectiveModelConfig());
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

  toJSON(): Prompt & { renderMeta?: Record<string, unknown> } {
    return {
      id: this.id,
      name: this.name,
      description: this.description,
      content: this.content,
      meta: this.meta,
      parameterSymbol: this.parameterSymbol,
      messages: this.messages,
      tools: this.tools,
      ...(Object.keys(this.renderMeta).length > 0 && { renderMeta: this.renderMeta }),
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
  private requestedVersion?: VersionSpecifier | string;
  private pendingVariables?: Variables;

  constructor(client: Echostash, promptId: string) {
    this.client = client;
    this.promptId = promptId;
  }

  /**
   * Request a specific version of the prompt
   *
   * @param version - Version number, 'published', 'staging', or a semver string
   */
  version(version: VersionSpecifier | string): PromptQuery {
    this.requestedVersion = version;
    return this;
  }

  /**
   * Alias for version()
   */
  v(version: VersionSpecifier | string): PromptQuery {
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
   * Fetch the prompt and return a LoadedPrompt (client-side substitution)
   */
  async get(): Promise<LoadedPrompt> {
    const versionStr = this.requestedVersion != null ? String(this.requestedVersion) : undefined;
    const prompt = await this.client.fetchPrompt(this.promptId, versionStr);
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

  /**
   * Server-side render: sends variables to the server for rendering.
   * Only available in 'echostash' mode.
   *
   * @example
   * ```ts
   * const result = await es.prompt(123).version('staging').render({ name: 'Alice' });
   * console.log(result.content); // "Hello Alice!"
   * ```
   */
  async render(variables?: Record<string, string>): Promise<RenderResponse> {
    return this.client.renderPrompt(this.promptId, this.requestedVersion, variables);
  }

  // --------------------------------------------------------------------------
  // Shorthand methods - fetch + convert in one call
  // --------------------------------------------------------------------------

  /**
   * Fetch prompt and convert to OpenAI format
   */
  async openai(): Promise<OpenAIPromptResult>;
  async openai(options: OpenAIOptions): Promise<OpenAIMessage>;
  async openai(options?: OpenAIOptions): Promise<OpenAIPromptResult | OpenAIMessage> {
    const loaded = await this.get();
    return (loaded.openai as (options?: OpenAIOptions) => OpenAIPromptResult | OpenAIMessage)(options);
  }

  /**
   * Fetch prompt and convert to Anthropic format
   */
  async anthropic(): Promise<AnthropicPromptResult>;
  async anthropic(options: AnthropicOptions): Promise<AnthropicMessage>;
  async anthropic(options?: AnthropicOptions): Promise<AnthropicPromptResult | AnthropicMessage> {
    const loaded = await this.get();
    return (loaded.anthropic as (options?: AnthropicOptions) => AnthropicPromptResult | AnthropicMessage)(options);
  }

  /**
   * Fetch prompt and convert to Google/Gemini format
   */
  async google(): Promise<GooglePromptResult>;
  async google(options: GoogleOptions): Promise<GoogleMessage>;
  async google(options?: GoogleOptions): Promise<GooglePromptResult | GoogleMessage> {
    const loaded = await this.get();
    return (loaded.google as (options?: GoogleOptions) => GooglePromptResult | GoogleMessage)(options);
  }

  /**
   * Fetch prompt and convert to Vercel AI SDK format
   */
  async vercel(): Promise<VercelPromptResult>;
  async vercel(options: VercelOptions): Promise<VercelMessage>;
  async vercel(options?: VercelOptions): Promise<VercelPromptResult | VercelMessage> {
    const loaded = await this.get();
    return (loaded.vercel as (options?: VercelOptions) => VercelPromptResult | VercelMessage)(options);
  }

  /**
   * Fetch prompt and convert to LangChain format
   */
  async langchain(): Promise<LangChainPromptResult>;
  async langchain(options: LangChainOptions): Promise<LangChainMessage>;
  async langchain(options?: LangChainOptions): Promise<LangChainPromptResult | LangChainMessage> {
    const loaded = await this.get();
    return (loaded.langchain as (options?: LangChainOptions) => LangChainPromptResult | LangChainMessage)(options);
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
  private readonly mode: 'echostash' | 'plp';

  // Observation buffer
  private observationBuffer: ObservationItem[] = [];
  private observationTimer: ReturnType<typeof setInterval> | null = null;
  private observationsForbidden = false;

  constructor(baseUrl: string, config: EchostashConfig = {}) {
    // Normalize base URL (remove trailing slash)
    this.baseUrl = baseUrl.replace(/\/+$/, '');
    this.apiKey = config.apiKey;
    this.headers = config.headers ?? {};
    this.timeout = config.timeout ?? 10000;
    this.defaultParameterSymbol = config.defaultParameterSymbol ?? '{{}}';
    this.mode = config.mode ?? 'echostash';
  }

  /**
   * Start building a prompt query
   *
   * @example
   * ```ts
   * const prompt = await es.prompt('marketing/welcome-email').get();
   * ```
   */
  prompt(promptId: string | number): PromptQuery {
    return new PromptQuery(this, String(promptId));
  }

  /**
   * Alias for prompt()
   */
  get(promptId: string | number): PromptQuery {
    return this.prompt(promptId);
  }

  /**
   * Internal method to fetch a prompt from the server
   */
  async fetchPrompt(promptId: string, version?: string): Promise<Prompt> {
    if (this.mode === 'echostash') {
      const path = version
        ? `/api/sdk/prompts/${encodeURIComponent(promptId)}/versions/${encodeURIComponent(version)}`
        : `/api/sdk/prompts/${encodeURIComponent(promptId)}`;
      const response = await this.request('GET', path);
      return this.normalizePrompt(response);
    }

    // PLP mode
    const path = version
      ? `/v1/prompts/${encodeURIComponent(promptId)}/${encodeURIComponent(version)}`
      : `/v1/prompts/${encodeURIComponent(promptId)}`;
    const response = await this.request('GET', path);
    return this.normalizePrompt(response);
  }

  /**
   * Server-side render a prompt. Only available in 'echostash' mode.
   */
  async renderPrompt(
    promptId: string,
    version?: VersionSpecifier | string,
    variables?: Record<string, string>,
  ): Promise<RenderResponse> {
    if (this.mode !== 'echostash') {
      throw new EchostashError('Server-side render is only available in echostash mode');
    }

    const body: { version?: string | number | null; variables?: Record<string, string> } = {};
    if (version !== undefined) {
      body.version = version;
    }
    if (variables && Object.keys(variables).length > 0) {
      body.variables = variables;
    }

    return await this.request(
      'POST',
      `/api/sdk/prompts/${encodeURIComponent(promptId)}/render`,
      body,
    );
  }

  /**
   * Batch render multiple prompts in a single request.
   * Only available in 'echostash' mode. Maximum 50 items.
   *
   * @example
   * ```ts
   * const result = await es.batchRender([
   *   { promptId: 1, version: 'published', variables: { name: 'Alice' } },
   *   { promptId: 2, variables: { greeting: 'Hello' } },
   * ]);
   * console.log(result.results['1'].content);
   * ```
   */
  async batchRender(items: BatchRenderItem[]): Promise<BatchRenderResponse> {
    if (this.mode !== 'echostash') {
      throw new EchostashError('Batch render is only available in echostash mode');
    }

    if (items.length > 50) {
      throw new EchostashError(`Batch render supports a maximum of 50 items, got ${items.length}`);
    }

    return await this.request('POST', '/api/sdk/prompts/batch', { items });
  }

  // --------------------------------------------------------------------------
  // Observations - Client-side render metrics reporting
  // --------------------------------------------------------------------------

  /**
   * Record an observation from a client-side render.
   * Observations are buffered and sent to the server every 60 seconds.
   * Only available in 'echostash' mode. Silently ignored for free users (403).
   *
   * @example
   * ```ts
   * const start = Date.now();
   * const rendered = prompt.with({ name: 'Alice' });
   * es.observeRender({
   *   promptId: 123,
   *   versionNo: 1,
   *   latencyMs: Date.now() - start,
   *   success: true,
   *   variableKeys: ['name'],
   * });
   * ```
   */
  observeRender(observation: ObservationItem): void {
    if (this.mode !== 'echostash' || this.observationsForbidden) {
      return;
    }

    this.observationBuffer.push({
      ...observation,
      timestamp: observation.timestamp ?? new Date().toISOString(),
    });

    // Start the flush timer on first observation
    if (!this.observationTimer) {
      this.observationTimer = setInterval(() => {
        void this.flush();
      }, 60_000);
      // Allow Node.js to exit even if the timer is still running
      if (typeof this.observationTimer === 'object' && 'unref' in this.observationTimer) {
        this.observationTimer.unref();
      }
    }
  }

  /**
   * Manually flush buffered observations to the server.
   * Called automatically every 60 seconds when observations are being recorded.
   */
  async flush(): Promise<void> {
    if (this.observationBuffer.length === 0 || this.observationsForbidden) {
      return;
    }

    // Drain buffer
    const items = this.observationBuffer.splice(0);

    try {
      await this.request('POST', '/api/sdk/observations', { items });
    } catch (error) {
      if (error instanceof EchostashError && error.statusCode === 403) {
        // Free user — stop sending observations
        this.observationsForbidden = true;
        if (this.observationTimer) {
          clearInterval(this.observationTimer);
          this.observationTimer = null;
        }
        return;
      }
      // On other errors (429 handled by request retry logic), put items back
      this.observationBuffer.unshift(...items);
    }
  }

  /**
   * Stop the observation flush timer and flush remaining observations.
   * Call this when shutting down the client.
   */
  async destroy(): Promise<void> {
    if (this.observationTimer) {
      clearInterval(this.observationTimer);
      this.observationTimer = null;
    }
    await this.flush();
  }

  /**
   * Save a prompt to the server (PLP mode only)
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
   * Delete a prompt from the server (PLP mode only)
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

  private async request(method: string, path: string, body?: unknown, attempt = 0): Promise<any> {
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
        const retryAfter = parseRetryAfter(response.headers.get('retry-after'));

        if (response.status === 429 && attempt < 3) {
          const delay = retryAfter != null
            ? retryAfter * 1000
            : Math.min(1000 * Math.pow(2, attempt), 8000);
          await sleep(delay);
          return this.request(method, path, body, attempt + 1);
        }

        let errorMessage = `HTTP ${response.status}`;
        try {
          const errorBody = await response.json() as { error?: string; message?: string };
          errorMessage = errorBody.message || errorBody.error || errorMessage;
        } catch {
          // Ignore JSON parse errors
        }
        throw new EchostashError(errorMessage, response.status, retryAfter ?? undefined);
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
        messages: this.normalizeServerMessages(data.messages),
        tools: this.normalizeServerTools(data.tools),
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
        messages: this.normalizeServerMessages(data.messages),
        tools: this.normalizeServerTools(data.tools),
      };
    }

    throw new EchostashError('Invalid prompt format received from server');
  }

  private normalizeServerMessages(messages: any): Message[] | undefined {
    if (!Array.isArray(messages) || messages.length === 0) return undefined;

    return messages.map((msg: any) => ({
      role: msg.role ?? 'user',
      content: Array.isArray(msg.content)
        ? (this.normalizeContent(msg.content) as ContentBlock[])
        : typeof msg.content === 'string'
          ? [{ type: 'text' as const, text: msg.content }]
          : [],
    }));
  }

  private normalizeServerTools(tools: any): ToolDefinition[] | undefined {
    if (!Array.isArray(tools) || tools.length === 0) return undefined;

    return tools.map((tool: any) => ({
      type: 'function' as const,
      function: {
        name: tool.function?.name ?? tool.name ?? '',
        description: tool.function?.description ?? tool.description ?? '',
        parameters: tool.function?.parameters ?? tool.parameters ?? {},
      },
    }));
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
  readonly retryAfter?: number;

  constructor(message: string, statusCode?: number, retryAfter?: number) {
    super(message);
    this.name = 'EchostashError';
    this.statusCode = statusCode;
    this.retryAfter = retryAfter;
  }

  get isRateLimited(): boolean {
    return this.statusCode === 429;
  }
}
