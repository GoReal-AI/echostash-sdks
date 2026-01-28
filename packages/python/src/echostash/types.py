"""
Type definitions for Echostash SDK
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union

# ============================================================================
# Content Types
# ============================================================================


@dataclass
class TextContent:
    """Text content block"""

    type: Literal["text"] = "text"
    text: str = ""


@dataclass
class ImageUrl:
    """Image URL details"""

    url: str
    detail: Optional[Literal["auto", "low", "high"]] = None


@dataclass
class ImageContent:
    """Image content block"""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrl = field(default_factory=lambda: ImageUrl(url=""))


ContentBlock = Union[TextContent, ImageContent]
PromptContent = Union[str, List[ContentBlock]]


# ============================================================================
# Model Configuration
# ============================================================================


@dataclass
class ModelConfig:
    """Model configuration hints from the prompt"""

    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


# ============================================================================
# Prompt Types
# ============================================================================


@dataclass
class PromptMeta:
    """Prompt metadata"""

    version: Optional[str] = None
    author: Optional[str] = None
    description: Optional[str] = None
    model_config: Optional[ModelConfig] = None
    token_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """The universal prompt envelope"""

    id: str
    content: PromptContent
    meta: PromptMeta = field(default_factory=PromptMeta)
    name: Optional[str] = None
    description: Optional[str] = None
    parameter_symbol: str = "{{}}"


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class EchostashConfig:
    """SDK client configuration"""

    api_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 10
    default_parameter_symbol: str = "{{}}"


# ============================================================================
# Provider Message Types
# ============================================================================


class OpenAIContentPart(TypedDict, total=False):
    """OpenAI content part"""

    type: str
    text: str
    image_url: Dict[str, Any]


class OpenAIMessage(TypedDict):
    """OpenAI message format"""

    role: Literal["system", "user", "assistant"]
    content: Union[str, List[OpenAIContentPart]]


class AnthropicContentBlock(TypedDict, total=False):
    """Anthropic content block"""

    type: str
    text: str
    source: Dict[str, Any]


class AnthropicMessage(TypedDict):
    """Anthropic message format"""

    role: Literal["user", "assistant"]
    content: Union[str, List[AnthropicContentBlock]]


class GooglePart(TypedDict, total=False):
    """Google/Gemini content part"""

    text: str
    inlineData: Dict[str, str]


class GoogleMessage(TypedDict):
    """Google/Gemini message format"""

    role: Literal["user", "model"]
    parts: List[GooglePart]


class VercelMessage(TypedDict):
    """Vercel AI SDK message format"""

    role: Literal["system", "user", "assistant"]
    content: str


class LangChainMessage(TypedDict):
    """LangChain message format"""

    type: Literal["system", "human", "ai"]
    content: str


# Type aliases for variables
Variables = Dict[str, Any]
