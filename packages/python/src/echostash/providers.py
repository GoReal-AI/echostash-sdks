"""
Provider adapters for converting prompts to provider-specific formats.

Each function converts prompt content to the exact format that provider's SDK expects.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .types import (
    Prompt,
    PromptContent,
    ModelConfig,
    TextContent,
    ImageContent,
    ContentBlock,
    OpenAIMessage,
    OpenAIContentPart,
    AnthropicMessage,
    AnthropicContentBlock,
    GoogleMessage,
    GooglePart,
    VercelMessage,
    LangChainMessage,
)


def _get_text_content(content: PromptContent) -> str:
    """Extract text from prompt content."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.text for block in content if isinstance(block, TextContent)
    )


# ============================================================================
# OpenAI
# ============================================================================


def to_openai(
    content: PromptContent,
    role: Literal["system", "user", "assistant"] = "user",
) -> OpenAIMessage:
    """Convert prompt content to OpenAI message format."""
    if isinstance(content, str):
        return {"role": role, "content": content}

    parts: List[OpenAIContentPart] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": block.image_url.url,
                    "detail": block.image_url.detail or "auto",
                },
            })

    return {"role": role, "content": parts}


def prompt_to_openai(
    prompt: Prompt,
    role: Literal["system", "user", "assistant"] = "user",
) -> List[OpenAIMessage]:
    """Convert full prompt to OpenAI messages array."""
    return [to_openai(prompt.content, role)]


def extract_openai_config(config: Optional[ModelConfig]) -> Dict[str, Any]:
    """Extract OpenAI-compatible model config from prompt metadata."""
    if not config:
        return {}

    result: Dict[str, Any] = {}
    if config.model:
        result["model"] = config.model
    if config.temperature is not None:
        result["temperature"] = config.temperature
    if config.top_p is not None:
        result["top_p"] = config.top_p
    if config.max_tokens is not None:
        result["max_tokens"] = config.max_tokens
    if config.seed is not None:
        result["seed"] = config.seed
    if config.stop is not None:
        result["stop"] = config.stop
    if config.presence_penalty is not None:
        result["presence_penalty"] = config.presence_penalty
    if config.frequency_penalty is not None:
        result["frequency_penalty"] = config.frequency_penalty

    return result


# ============================================================================
# Anthropic
# ============================================================================


def to_anthropic(
    content: PromptContent,
    role: Literal["user", "assistant"] = "user",
) -> AnthropicMessage:
    """Convert prompt content to Anthropic message format."""
    if isinstance(content, str):
        return {"role": role, "content": content}

    blocks: List[AnthropicContentBlock] = []
    for block in content:
        if isinstance(block, TextContent):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageContent):
            url = block.image_url.url
            if url.startswith("data:"):
                match = re.match(r"^data:([^;]+);base64,(.+)$", url)
                if match:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": match.group(1),
                            "data": match.group(2),
                        },
                    })
            else:
                blocks.append({
                    "type": "image",
                    "source": {"type": "url", "url": url},
                })

    return {"role": role, "content": blocks}


def to_anthropic_system(content: PromptContent) -> str:
    """Get content as Anthropic system message (string)."""
    return _get_text_content(content)


def prompt_to_anthropic(
    prompt: Prompt,
    role: Literal["user", "assistant"] = "user",
    as_system: bool = False,
) -> Dict[str, Any]:
    """Convert full prompt to Anthropic format.

    Returns dict with 'system' and/or 'messages' keys.
    """
    if as_system:
        return {"system": to_anthropic_system(prompt.content), "messages": []}
    return {"messages": [to_anthropic(prompt.content, role)]}


def extract_anthropic_config(config: Optional[ModelConfig]) -> Dict[str, Any]:
    """Extract Anthropic-compatible model config from prompt metadata."""
    if not config:
        return {}

    result: Dict[str, Any] = {}
    if config.model:
        result["model"] = config.model
    if config.temperature is not None:
        result["temperature"] = config.temperature
    if config.top_p is not None:
        result["top_p"] = config.top_p
    if config.top_k is not None:
        result["top_k"] = config.top_k
    if config.max_tokens is not None:
        result["max_tokens"] = config.max_tokens
    if config.stop is not None:
        result["stop_sequences"] = [config.stop] if isinstance(config.stop, str) else config.stop

    return result


# ============================================================================
# Google / Gemini
# ============================================================================


def to_google(
    content: PromptContent,
    role: Literal["user", "model"] = "user",
) -> GoogleMessage:
    """Convert prompt content to Google/Gemini message format."""
    if isinstance(content, str):
        return {"role": role, "parts": [{"text": content}]}

    parts: List[GooglePart] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append({"text": block.text})
        elif isinstance(block, ImageContent):
            url = block.image_url.url
            if url.startswith("data:"):
                match = re.match(r"^data:([^;]+);base64,(.+)$", url)
                if match:
                    parts.append({
                        "inlineData": {
                            "mimeType": match.group(1),
                            "data": match.group(2),
                        },
                    })
            else:
                parts.append({"text": f"[Image: {url}]"})

    return {"role": role, "parts": parts}


def prompt_to_google(
    prompt: Prompt,
    role: Literal["user", "model"] = "user",
) -> List[GoogleMessage]:
    """Convert full prompt to Google messages array."""
    return [to_google(prompt.content, role)]


def extract_google_config(config: Optional[ModelConfig]) -> Dict[str, Any]:
    """Extract Google-compatible model config from prompt metadata."""
    if not config:
        return {}

    result: Dict[str, Any] = {}
    generation_config: Dict[str, Any] = {}

    if config.model:
        result["model"] = config.model
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    if config.top_p is not None:
        generation_config["topP"] = config.top_p
    if config.top_k is not None:
        generation_config["topK"] = config.top_k
    if config.max_tokens is not None:
        generation_config["maxOutputTokens"] = config.max_tokens
    if config.stop is not None:
        generation_config["stopSequences"] = [config.stop] if isinstance(config.stop, str) else config.stop

    if generation_config:
        result["generationConfig"] = generation_config

    return result


# ============================================================================
# Vercel AI SDK
# ============================================================================


def to_vercel(
    content: PromptContent,
    role: Literal["system", "user", "assistant"] = "user",
) -> VercelMessage:
    """Convert prompt content to Vercel AI SDK message format."""
    return {"role": role, "content": _get_text_content(content)}


def prompt_to_vercel(
    prompt: Prompt,
    role: Literal["system", "user", "assistant"] = "user",
) -> List[VercelMessage]:
    """Convert full prompt to Vercel AI SDK messages array."""
    return [to_vercel(prompt.content, role)]


# ============================================================================
# LangChain
# ============================================================================


def to_langchain(
    content: PromptContent,
    message_type: Literal["system", "human", "ai"] = "human",
) -> LangChainMessage:
    """Convert prompt content to LangChain message format."""
    return {"type": message_type, "content": _get_text_content(content)}


def prompt_to_langchain(
    prompt: Prompt,
    message_type: Literal["system", "human", "ai"] = "human",
) -> List[LangChainMessage]:
    """Convert full prompt to LangChain messages array."""
    return [to_langchain(prompt.content, message_type)]


def to_langchain_template(prompt: Prompt) -> Tuple[str, List[str]]:
    """Get prompt as LangChain PromptTemplate-compatible format.

    Returns:
        Tuple of (template, input_variables)
    """
    content = _get_text_content(prompt.content)

    # Parse parameter symbol
    param_symbol = prompt.parameter_symbol or "{{}}"
    mid = len(param_symbol) // 2
    prefix = param_symbol[:mid]
    suffix = param_symbol[mid:]

    # Escape regex special chars
    prefix_escaped = re.escape(prefix)
    suffix_escaped = re.escape(suffix)

    # Find all variables
    pattern = re.compile(rf"{prefix_escaped}([\w.]+){suffix_escaped}")
    input_variables = list(set(pattern.findall(content)))

    # Convert to LangChain format {var}
    template = pattern.sub(r"{\1}", content)

    return template, input_variables
