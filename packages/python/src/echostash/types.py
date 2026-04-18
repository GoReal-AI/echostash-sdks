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


@dataclass
class ToolCallContentBlock:
    """Tool call content block"""

    type: Literal["tool_call"] = "tool_call"
    tool_call: Dict[str, Any] = field(default_factory=lambda: {"name": "", "arguments": {}})


ContentBlock = Union[TextContent, ImageContent, ToolCallContentBlock]
PromptContent = Union[str, List[ContentBlock]]


# ============================================================================
# Message & Tool Types
# ============================================================================

MessageRole = Literal["system", "user", "assistant"]


@dataclass
class Message:
    """A message with role and content blocks"""

    role: MessageRole
    content: List[Dict[str, Any]]  # ContentBlock dicts or dataclass instances


@dataclass
class ToolDefinition:
    """A tool/function definition for function calling"""

    type: str = "function"
    function: Dict[str, Any] = field(
        default_factory=lambda: {"name": "", "description": "", "parameters": {}}
    )


@dataclass
class SkillDefinition:
    """A skill definition for skill-based prompts"""

    type: str = "skill"
    skill: Dict[str, Any] = field(
        default_factory=lambda: {"name": "", "description": "", "parameters": {}}
    )


@dataclass
class SkillDiscoveryResult:
    """Result from skill discovery endpoint"""

    id: int = 0
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)


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
    type: Optional[str] = None
    parameter_symbol: str = "{{}}"
    messages: Optional[List[Message]] = None
    tools: Optional[List[ToolDefinition]] = None
    skills: Optional[List[SkillDefinition]] = None


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
    mode: Literal["echostash", "plp"] = "echostash"


# ============================================================================
# Server-side Render Types
# ============================================================================

VersionSpecifier = Union[Literal["published", "staging"], int, None]


@dataclass
class RenderResponse:
    """Response from POST /api/sdk/prompts/{id}/render"""

    content: str
    prompt_id: int
    version_no: int


@dataclass
class BatchRenderItem:
    """A single item in a batch render request"""

    prompt_id: int
    version: Union[str, int, None] = None
    variables: Optional[Dict[str, str]] = None


@dataclass
class BatchRenderResult:
    """Result for a single prompt in a batch render response"""

    content: str
    version_no: int
    error: Optional[str] = None


@dataclass
class BatchRenderResponse:
    """Response from POST /api/sdk/prompts/batch"""

    results: Dict[str, BatchRenderResult]
    success_count: int
    error_count: int


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


# ============================================================================
# Observation Types
# ============================================================================


@dataclass
class ObservationItem:
    """A single observation item for client-side render metrics"""

    prompt_id: int
    version_no: int
    latency_ms: int
    success: bool
    variable_keys: Optional[List[str]] = None
    timestamp: Optional[str] = None
