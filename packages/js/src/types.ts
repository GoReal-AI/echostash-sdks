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

export type ContentBlock = TextContent | ImageContent;
export type PromptContent = string | ContentBlock[];

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

/**
 * The universal prompt envelope - works with any PLP server
 */
export interface Prompt {
  id: string;
  name?: string;
  description?: string;
  content: PromptContent;
  meta: PromptMeta;
  parameterSymbol?: string;
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
}

/**
 * Variables for prompt rendering
 */
export type Variables = Record<string, string | number | boolean | null | undefined>;

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
