// ============================================================================
// Core Types - The foundation of Echostash SDK
// ============================================================================

/**
 * Content block types for multi-modal prompts
 */
export interface TextContent {
  type: 'text';
  text: string;
}

export interface ImageContent {
  type: 'image_url';
  image_url: {
    url: string;
    detail?: 'auto' | 'low' | 'high';
  };
}

export interface ToolCallContentBlock {
  type: 'tool_call';
  tool_call: {
    id?: string;
    name: string;
    arguments: Record<string, unknown>;
  };
}

export type ContentBlock = TextContent | ImageContent | ToolCallContentBlock;
export type PromptContent = string | ContentBlock[];

// ============================================================================
// Message & Tool Types
// ============================================================================

export type MessageRole = 'system' | 'user' | 'assistant';

export interface Message {
  role: MessageRole;
  content: ContentBlock[];
}

export interface ToolDefinition {
  type: 'function';
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
}

// ============================================================================
// Provider-specific Result Types (Messages + Tools)
// ============================================================================

export interface OpenAIPromptResult {
  messages: Array<{
    role: string;
    content: string | Array<{ type: string; [key: string]: unknown }>;
  }>;
  tools?: Array<{
    type: 'function';
    function: { name: string; description: string; parameters: Record<string, unknown> };
  }>;
  model?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  [key: string]: unknown;
}

export interface AnthropicPromptResult {
  system?: string;
  messages: Array<{
    role: string;
    content: string | Array<{ type: string; [key: string]: unknown }>;
  }>;
  tools?: Array<{
    name: string;
    description: string;
    input_schema: Record<string, unknown>;
  }>;
  model?: string;
  max_tokens?: number;
  temperature?: number;
  [key: string]: unknown;
}

export interface GooglePromptResult {
  contents: Array<{
    role: string;
    parts: Array<{ text?: string; inlineData?: { mimeType: string; data: string } }>;
  }>;
  tools?: Array<{
    functionDeclarations: Array<{
      name: string;
      description: string;
      parameters: Record<string, unknown>;
    }>;
  }>;
  generationConfig?: Record<string, unknown>;
}

export interface VercelPromptResult {
  messages: Array<{
    role: string;
    content: string;
  }>;
  tools?: Record<string, {
    description: string;
    parameters: Record<string, unknown>;
  }>;
}

export interface LangChainPromptResult {
  messages: Array<{
    type: string;
    content: string;
  }>;
  tools?: Array<{
    name: string;
    description: string;
    schema: Record<string, unknown>;
  }>;
}

/**
 * Model configuration hints from the prompt
 */
export interface ModelConfig {
  provider?: string;
  model?: string;
  temperature?: number;
  topP?: number;
  topK?: number;
  maxTokens?: number;
  seed?: number;
  stop?: string | string[];
  presencePenalty?: number;
  frequencyPenalty?: number;
  [key: string]: unknown;
}

/**
 * Prompt metadata
 */
export interface PromptMeta {
  version?: string;
  author?: string;
  description?: string;
  modelConfig?: ModelConfig;
  tokenCount?: number;
  [key: string]: unknown;
}

export interface SkillDefinition {
  type: 'skill';
  skill: {
    name: string;
    description: string;
    source?: string;
    parameters: Record<string, unknown>;
  };
}

export interface SkillDiscoveryResult {
  id: number;
  name: string;
  description: string;
  tags: string[];
}

export interface DiscoverSkillsOptions {
  tagIds?: number[];
  query?: string;
}

/**
 * Result from server-side rendering with meta template support.
 * The `meta` field contains key-value pairs from the rendered meta template.
 */
export interface RenderResult {
  messages: Array<{ role: string; content: string }>;
  tools?: ToolDefinition[];
  skills?: SkillDefinition[];
  meta?: Record<string, unknown>;
}

/**
 * The universal prompt envelope - works with any PLP server
 */
export interface Prompt {
  id: string;
  name?: string;
  description?: string;
  type?: string;
  content: PromptContent;
  meta: PromptMeta;
  parameterSymbol?: string;
  messages?: Message[];
  tools?: ToolDefinition[];
  skills?: SkillDefinition[];
}

/**
 * SDK client configuration
 */
export interface EchostashConfig {
  /** API key for authentication (optional for some servers) */
  apiKey?: string;
  /** Custom headers to send with requests */
  headers?: Record<string, string>;
  /** Request timeout in milliseconds (default: 10000) */
  timeout?: number;
  /** Default parameter symbol for variable substitution (default: "{{}}") */
  defaultParameterSymbol?: string;
  /** API mode: 'echostash' uses /api/sdk/ endpoints, 'plp' uses /v1/ endpoints (default: 'echostash') */
  mode?: 'echostash' | 'plp';
}

// ============================================================================
// Server-side Render Types
// ============================================================================

/**
 * Version specifier for server-side render.
 * - 'published': the published version
 * - 'staging': the staging version
 * - number: a specific version number
 * - null/undefined: defaults to published
 */
export type VersionSpecifier = 'published' | 'staging' | number | null | undefined;

/** Request body for POST /api/sdk/prompts/{id}/render */
export interface RenderRequest {
  version?: string | number | null;
  variables?: Record<string, string>;
}

/** Response from POST /api/sdk/prompts/{id}/render */
export interface RenderResponse {
  content: string;
  promptId: number;
  versionNo: number;
}

/** A single item in a batch render request */
export interface BatchRenderItem {
  promptId: number;
  version?: string | number | null;
  variables?: Record<string, string>;
}

/** Result for a single prompt in a batch render response */
export interface BatchRenderResult {
  content: string;
  versionNo: number;
  error: string | null;
}

/** Response from POST /api/sdk/prompts/batch */
export interface BatchRenderResponse {
  results: Record<string, BatchRenderResult>;
  successCount: number;
  errorCount: number;
}

/**
 * Variables for prompt rendering
 */
export type Variables = Record<string, string | number | boolean | null | undefined>;

// ============================================================================
// Observation Types
// ============================================================================

/** A single observation item for client-side render metrics */
export interface ObservationItem {
  promptId: number;
  versionNo: number;
  latencyMs: number;
  success: boolean;
  variableKeys?: string[];
  timestamp?: string;
}

/** Request body for POST /api/sdk/observations */
export interface ObservationBatchRequest {
  items: ObservationItem[];
}

// ============================================================================
// Provider-specific message types
// ============================================================================

/** OpenAI message format */
export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | OpenAIContentPart[];
}

export interface OpenAIContentPart {
  type: 'text' | 'image_url';
  text?: string;
  image_url?: { url: string; detail?: 'auto' | 'low' | 'high' };
}

/** Anthropic message format */
export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: string | AnthropicContentBlock[];
}

export interface AnthropicContentBlock {
  type: 'text' | 'image';
  text?: string;
  source?: {
    type: 'base64' | 'url';
    media_type?: string;
    data?: string;
    url?: string;
  };
}

/** Google/Gemini message format */
export interface GoogleMessage {
  role: 'user' | 'model';
  parts: GooglePart[];
}

export interface GooglePart {
  text?: string;
  inlineData?: {
    mimeType: string;
    data: string;
  };
}

/** Vercel AI SDK message format */
export interface VercelMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/** LangChain message types */
export type LangChainMessageType = 'system' | 'human' | 'ai';

export interface LangChainMessage {
  type: LangChainMessageType;
  content: string;
}

// ============================================================================
// Conversion options
// ============================================================================

export interface ConversionOptions {
  /** Include model config in the output where supported */
  includeConfig?: boolean;
}

export interface OpenAIOptions extends ConversionOptions {
  role?: 'system' | 'user' | 'assistant';
}

export interface AnthropicOptions extends ConversionOptions {
  role?: 'user' | 'assistant';
  /** Separate system message (Anthropic handles system differently) */
  asSystem?: boolean;
}

export interface GoogleOptions extends ConversionOptions {
  role?: 'user' | 'model';
}

export interface VercelOptions extends ConversionOptions {
  role?: 'system' | 'user' | 'assistant';
}

export interface LangChainOptions extends ConversionOptions {
  type?: LangChainMessageType;
}
