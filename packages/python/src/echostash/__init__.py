"""
Echostash SDK - The universal prompt SDK

Fetch prompts from any PLP-compliant library and convert them
to any AI provider's format.

Example:
    >>> from echostash import Echostash
    >>>
    >>> # Connect to any PLP-compliant server
    >>> es = Echostash("https://api.echostash.com", api_key="sk_...")
    >>>
    >>> # Fetch a prompt
    >>> prompt = es.prompt("welcome-email").get()
    >>>
    >>> # Use with OpenAI
    >>> message = prompt.with_vars(name="Alice").openai()
    >>> openai.chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[message]
    ... )
    >>>
    >>> # Use with Anthropic
    >>> message = prompt.with_vars(name="Bob").anthropic()
    >>>
    >>> # One-liner: fetch + substitute + convert
    >>> msg = es.prompt("welcome").vars(name="Dave").openai()
"""

from .client import Echostash, LoadedPrompt, PromptQuery, EchostashError
from .types import (
    Prompt,
    PromptContent,
    PromptMeta,
    ModelConfig,
    TextContent,
    ImageContent,
    ToolCallContentBlock,
    ContentBlock,
    Message,
    MessageRole,
    ToolDefinition,
    EchostashConfig,
    VersionSpecifier,
    RenderResponse,
    BatchRenderItem,
    BatchRenderResult,
    BatchRenderResponse,
    ObservationItem,
)
from .providers import (
    # OpenAI
    to_openai,
    to_openai_prompt_result,
    prompt_to_openai,
    extract_openai_config,
    # Anthropic
    to_anthropic,
    to_anthropic_system,
    to_anthropic_prompt_result,
    prompt_to_anthropic,
    extract_anthropic_config,
    # Google
    to_google,
    to_google_prompt_result,
    prompt_to_google,
    extract_google_config,
    # Vercel
    to_vercel,
    to_vercel_prompt_result,
    prompt_to_vercel,
    # LangChain
    to_langchain,
    to_langchain_prompt_result,
    prompt_to_langchain,
    to_langchain_template,
)

__version__ = "1.1.0"
__all__ = [
    # Main classes
    "Echostash",
    "LoadedPrompt",
    "PromptQuery",
    "EchostashError",
    # Types
    "Prompt",
    "PromptContent",
    "PromptMeta",
    "ModelConfig",
    "TextContent",
    "ImageContent",
    "ToolCallContentBlock",
    "ContentBlock",
    "Message",
    "MessageRole",
    "ToolDefinition",
    "EchostashConfig",
    # Server-side render types
    "VersionSpecifier",
    "RenderResponse",
    "BatchRenderItem",
    "BatchRenderResult",
    "BatchRenderResponse",
    # Observation types
    "ObservationItem",
    # Providers
    "to_openai",
    "to_openai_prompt_result",
    "prompt_to_openai",
    "extract_openai_config",
    "to_anthropic",
    "to_anthropic_system",
    "to_anthropic_prompt_result",
    "prompt_to_anthropic",
    "extract_anthropic_config",
    "to_google",
    "to_google_prompt_result",
    "prompt_to_google",
    "extract_google_config",
    "to_vercel",
    "to_vercel_prompt_result",
    "prompt_to_vercel",
    "to_langchain",
    "to_langchain_prompt_result",
    "prompt_to_langchain",
    "to_langchain_template",
]
