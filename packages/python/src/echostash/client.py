"""
Echostash client - The fluent API for fetching and using prompts.
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests

from .types import (
    Prompt,
    PromptContent,
    PromptMeta,
    ModelConfig,
    TextContent,
    ImageContent,
    ImageUrl,
    ContentBlock,
    EchostashConfig,
    Variables,
    OpenAIMessage,
    AnthropicMessage,
    GoogleMessage,
    VercelMessage,
    LangChainMessage,
)
from .providers import (
    to_openai,
    extract_openai_config,
    to_anthropic,
    to_anthropic_system,
    extract_anthropic_config,
    to_google,
    extract_google_config,
    to_vercel,
    to_langchain,
    to_langchain_template,
)


class EchostashError(Exception):
    """Exception raised for Echostash SDK errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


def _substitute_variables(
    content: PromptContent,
    variables: Variables,
    parameter_symbol: str,
) -> PromptContent:
    """Substitute variables in content using the parameter symbol."""
    if not variables:
        return content

    # Parse parameter symbol (e.g., "{{}}" -> prefix: "{{", suffix: "}}")
    mid = len(parameter_symbol) // 2
    prefix = parameter_symbol[:mid]
    suffix = parameter_symbol[mid:]

    prefix_escaped = re.escape(prefix)
    suffix_escaped = re.escape(suffix)

    def substitute(text: str) -> str:
        result = text
        for key, value in variables.items():
            pattern = re.compile(rf"{prefix_escaped}{re.escape(key)}{suffix_escaped}")
            result = pattern.sub(str(value) if value is not None else "", result)
        return result

    if isinstance(content, str):
        return substitute(content)

    result: List[ContentBlock] = []
    for block in content:
        if isinstance(block, TextContent):
            result.append(TextContent(text=substitute(block.text)))
        else:
            result.append(block)
    return result


def _get_text_content(content: PromptContent) -> str:
    """Extract text from prompt content."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block.text for block in content if isinstance(block, TextContent)
    )


class LoadedPrompt:
    """A loaded prompt with fluent methods for variable substitution and provider conversion.

    Example:
        >>> prompt = es.prompt("welcome").get()
        >>>
        >>> # Substitute variables
        >>> rendered = prompt.with_vars(name="Alice")
        >>>
        >>> # Convert to provider formats
        >>> openai_msg = rendered.openai()
        >>> anthropic_msg = rendered.anthropic()
    """

    def __init__(self, prompt: Prompt):
        self.id = prompt.id
        self.name = prompt.name
        self.description = prompt.description
        self.content = prompt.content
        self.meta = prompt.meta
        self.parameter_symbol = prompt.parameter_symbol

    # --------------------------------------------------------------------------
    # Variable Substitution
    # --------------------------------------------------------------------------

    def with_vars(self, **variables: Any) -> LoadedPrompt:
        """Substitute variables in the prompt content.

        Example:
            >>> rendered = prompt.with_vars(name="Alice", age=30)
        """
        new_content = _substitute_variables(
            self.content, variables, self.parameter_symbol
        )
        return LoadedPrompt(Prompt(
            id=self.id,
            name=self.name,
            description=self.description,
            content=new_content,
            meta=self.meta,
            parameter_symbol=self.parameter_symbol,
        ))

    def vars(self, **variables: Any) -> LoadedPrompt:
        """Alias for with_vars()."""
        return self.with_vars(**variables)

    def render(self, **variables: Any) -> LoadedPrompt:
        """Alias for with_vars()."""
        return self.with_vars(**variables)

    # --------------------------------------------------------------------------
    # Content Access
    # --------------------------------------------------------------------------

    def raw(self) -> PromptContent:
        """Get the raw content."""
        return self.content

    def text(self) -> str:
        """Get content as plain text."""
        return _get_text_content(self.content)

    def __str__(self) -> str:
        """Get content as string."""
        return self.text()

    # --------------------------------------------------------------------------
    # Provider Conversions
    # --------------------------------------------------------------------------

    def openai(
        self,
        role: Literal["system", "user", "assistant"] = "user",
    ) -> OpenAIMessage:
        """Convert to OpenAI message format.

        Example:
            >>> message = prompt.openai(role="system")
            >>> openai.chat.completions.create(
            ...     model="gpt-4",
            ...     messages=[message]
            ... )
        """
        return to_openai(self.content, role)

    def openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-compatible model config from prompt metadata."""
        return extract_openai_config(self.meta.model_config)

    def anthropic(
        self,
        role: Literal["user", "assistant"] = "user",
    ) -> AnthropicMessage:
        """Convert to Anthropic message format.

        Example:
            >>> message = prompt.anthropic()
            >>> anthropic.messages.create(
            ...     model="claude-3-opus-20240229",
            ...     messages=[message]
            ... )
        """
        return to_anthropic(self.content, role)

    def anthropic_system(self) -> str:
        """Get content as Anthropic system message (string)."""
        return to_anthropic_system(self.content)

    def anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic-compatible model config from prompt metadata."""
        return extract_anthropic_config(self.meta.model_config)

    def google(
        self,
        role: Literal["user", "model"] = "user",
    ) -> GoogleMessage:
        """Convert to Google/Gemini message format."""
        return to_google(self.content, role)

    def gemini(
        self,
        role: Literal["user", "model"] = "user",
    ) -> GoogleMessage:
        """Alias for google()."""
        return self.google(role)

    def google_config(self) -> Dict[str, Any]:
        """Get Google-compatible model config from prompt metadata."""
        return extract_google_config(self.meta.model_config)

    def vercel(
        self,
        role: Literal["system", "user", "assistant"] = "user",
    ) -> VercelMessage:
        """Convert to Vercel AI SDK message format."""
        return to_vercel(self.content, role)

    def langchain(
        self,
        message_type: Literal["system", "human", "ai"] = "human",
    ) -> LangChainMessage:
        """Convert to LangChain message format."""
        return to_langchain(self.content, message_type)

    def langchain_template(self) -> Tuple[str, List[str]]:
        """Get as LangChain PromptTemplate-compatible format.

        Returns:
            Tuple of (template, input_variables)

        Example:
            >>> template, vars = prompt.langchain_template()
            >>> prompt_template = PromptTemplate.from_template(template)
        """
        return to_langchain_template(Prompt(
            id=self.id,
            content=self.content,
            meta=self.meta,
            parameter_symbol=self.parameter_symbol,
        ))

    # --------------------------------------------------------------------------
    # JSON serialization
    # --------------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "meta": self.meta,
            "parameter_symbol": self.parameter_symbol,
        }


class PromptQuery:
    """A builder for fetching and configuring prompts.

    Example:
        >>> # Fetch latest version
        >>> prompt = es.prompt("welcome").get()
        >>>
        >>> # Fetch specific version
        >>> prompt = es.prompt("welcome").version("1.2.0").get()
        >>>
        >>> # Fetch and substitute in one chain
        >>> msg = es.prompt("welcome").vars(name="Alice").openai()
    """

    def __init__(self, client: Echostash, prompt_id: str):
        self._client = client
        self._prompt_id = prompt_id
        self._version: Optional[str] = None
        self._pending_variables: Optional[Variables] = None

    def version(self, version: str) -> PromptQuery:
        """Request a specific version of the prompt."""
        self._version = version
        return self

    def v(self, version: str) -> PromptQuery:
        """Alias for version()."""
        return self.version(version)

    def vars(self, **variables: Any) -> PromptQuery:
        """Pre-set variables to substitute after fetching."""
        self._pending_variables = variables
        return self

    def with_vars(self, **variables: Any) -> PromptQuery:
        """Alias for vars()."""
        return self.vars(**variables)

    def get(self) -> LoadedPrompt:
        """Fetch the prompt and return a LoadedPrompt."""
        prompt = self._client.fetch_prompt(self._prompt_id, self._version)
        loaded = LoadedPrompt(prompt)

        if self._pending_variables:
            loaded = loaded.with_vars(**self._pending_variables)

        return loaded

    def fetch(self) -> LoadedPrompt:
        """Alias for get()."""
        return self.get()

    # --------------------------------------------------------------------------
    # Shorthand methods - fetch + convert in one call
    # --------------------------------------------------------------------------

    def openai(
        self,
        role: Literal["system", "user", "assistant"] = "user",
    ) -> OpenAIMessage:
        """Fetch prompt and convert to OpenAI format."""
        return self.get().openai(role)

    def anthropic(
        self,
        role: Literal["user", "assistant"] = "user",
    ) -> AnthropicMessage:
        """Fetch prompt and convert to Anthropic format."""
        return self.get().anthropic(role)

    def google(
        self,
        role: Literal["user", "model"] = "user",
    ) -> GoogleMessage:
        """Fetch prompt and convert to Google/Gemini format."""
        return self.get().google(role)

    def vercel(
        self,
        role: Literal["system", "user", "assistant"] = "user",
    ) -> VercelMessage:
        """Fetch prompt and convert to Vercel AI SDK format."""
        return self.get().vercel(role)

    def langchain(
        self,
        message_type: Literal["system", "human", "ai"] = "human",
    ) -> LangChainMessage:
        """Fetch prompt and convert to LangChain format."""
        return self.get().langchain(message_type)

    def text(self) -> str:
        """Fetch prompt and get as plain text."""
        return self.get().text()


class Echostash:
    """The Echostash client - connects to any PLP-compliant prompt library.

    Example:
        >>> # Connect to Echostash Cloud
        >>> es = Echostash("https://api.echostash.com", api_key="sk_...")
        >>>
        >>> # Connect to a local PLP server
        >>> local = Echostash("http://localhost:3000")
        >>>
        >>> # Connect to any PLP-compliant library
        >>> custom = Echostash("https://prompts.mycompany.com")
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        default_parameter_symbol: str = "{{}}",
    ):
        """Initialize the Echostash client.

        Args:
            base_url: The base URL of the PLP-compliant server
            api_key: Optional API key for authentication
            headers: Optional additional headers
            timeout: Request timeout in seconds
            default_parameter_symbol: Default symbol for variable substitution
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = headers or {}
        self._timeout = timeout
        self._default_parameter_symbol = default_parameter_symbol
        self._session = requests.Session()

    def __enter__(self) -> Echostash:
        return self

    def __exit__(self, *args: Any) -> None:
        self._session.close()

    def prompt(self, prompt_id: str) -> PromptQuery:
        """Start building a prompt query.

        Example:
            >>> prompt = es.prompt("marketing/welcome-email").get()
        """
        return PromptQuery(self, prompt_id)

    def get(self, prompt_id: str) -> PromptQuery:
        """Alias for prompt()."""
        return self.prompt(prompt_id)

    def fetch_prompt(self, prompt_id: str, version: Optional[str] = None) -> Prompt:
        """Internal method to fetch a prompt from the server."""
        if version:
            path = f"/v1/prompts/{prompt_id}/{version}"
        else:
            path = f"/v1/prompts/{prompt_id}"

        response = self._request("GET", path)
        return self._normalize_prompt(response)

    def save(
        self,
        prompt_id: str,
        content: PromptContent,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LoadedPrompt:
        """Save a prompt to the server.

        Example:
            >>> es.save(
            ...     "marketing/new-prompt",
            ...     content="Hello {{name}}!",
            ...     meta={"version": "1.0.0"}
            ... )
        """
        body = {"content": content, "meta": meta or {}}
        response = self._request("PUT", f"/v1/prompts/{prompt_id}", body)
        return LoadedPrompt(self._normalize_prompt(response))

    def delete(self, prompt_id: str) -> None:
        """Delete a prompt from the server."""
        self._request("DELETE", f"/v1/prompts/{prompt_id}")

    def discover(self) -> Dict[str, Any]:
        """Check server capabilities via PLP discovery endpoint."""
        try:
            response = self._request("GET", "/.well-known/plp")
            return {
                "plp_version": response.get("plp_version", "1.0"),
                "server": response.get("server"),
                "capabilities": response.get("capabilities"),
            }
        except EchostashError:
            return {"plp_version": "1.0"}

    # --------------------------------------------------------------------------
    # HTTP Request Helper
    # --------------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an HTTP request to the server."""
        url = f"{self._base_url}{path}"

        headers = {
            "Content-Type": "application/json",
            **self._headers,
        }

        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            headers["X-API-KEY"] = self._api_key

        response = self._session.request(
            method=method,
            url=url,
            headers=headers,
            json=body,
            timeout=self._timeout,
        )

        if not response.ok:
            error_message = f"HTTP {response.status_code}"
            try:
                error_body = response.json()
                error_message = error_body.get("error") or error_body.get("message") or error_message
            except Exception:
                pass
            raise EchostashError(error_message, response.status_code)

        if response.status_code == 204:
            return None

        return response.json()

    # --------------------------------------------------------------------------
    # Normalize prompt from different server formats
    # --------------------------------------------------------------------------

    def _normalize_prompt(self, data: Dict[str, Any]) -> Prompt:
        """Normalize prompt from server response."""
        # Handle PLP format
        if "id" in data and "content" in data and "meta" in data:
            return Prompt(
                id=str(data["id"]),
                name=data.get("name"),
                description=data.get("description") or data.get("meta", {}).get("description"),
                content=self._normalize_content(data["content"]),
                meta=self._normalize_meta(data.get("meta", {})),
                parameter_symbol=data.get("parameterSymbol", self._default_parameter_symbol),
            )

        # Handle Echostash format
        if "id" in data and "content" in data:
            return Prompt(
                id=str(data["id"]),
                name=data.get("name"),
                description=data.get("description"),
                content=self._normalize_content(data["content"]),
                meta=self._normalize_meta(data.get("promptMetaData") or data.get("meta") or {}),
                parameter_symbol=data.get("parameterSymbol", self._default_parameter_symbol),
            )

        raise EchostashError("Invalid prompt format received from server")

    def _normalize_content(self, content: Any) -> PromptContent:
        """Normalize content from various formats."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            result: List[ContentBlock] = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        result.append(TextContent(text=block.get("text", "")))
                    elif block.get("type") == "image_url":
                        img_url = block.get("image_url", {})
                        if isinstance(img_url, str):
                            result.append(ImageContent(image_url=ImageUrl(url=img_url)))
                        else:
                            result.append(ImageContent(
                                image_url=ImageUrl(
                                    url=img_url.get("url", ""),
                                    detail=img_url.get("detail"),
                                )
                            ))
                    else:
                        # Fallback to text
                        result.append(TextContent(text=str(block.get("text", block))))
            return result

        return str(content)

    def _normalize_meta(self, meta: Dict[str, Any]) -> PromptMeta:
        """Normalize metadata from various formats."""
        model_config_data = meta.get("modelConfig") or meta.get("model_config") or meta.get("modelData")
        model_config = None

        if model_config_data:
            model_config = ModelConfig(
                provider=model_config_data.get("provider"),
                model=model_config_data.get("model") or model_config_data.get("modelName"),
                temperature=model_config_data.get("temperature"),
                top_p=model_config_data.get("topP") or model_config_data.get("top_p"),
                top_k=model_config_data.get("topK") or model_config_data.get("top_k"),
                max_tokens=model_config_data.get("maxTokens") or model_config_data.get("max_tokens"),
                seed=model_config_data.get("seed"),
                stop=model_config_data.get("stop"),
                presence_penalty=model_config_data.get("presencePenalty") or model_config_data.get("presence_penalty"),
                frequency_penalty=model_config_data.get("frequencyPenalty") or model_config_data.get("frequency_penalty"),
            )

        return PromptMeta(
            version=meta.get("version"),
            author=meta.get("author"),
            description=meta.get("description"),
            model_config=model_config,
            token_count=meta.get("tokenCount") or meta.get("token_count"),
            extra={k: v for k, v in meta.items() if k not in {
                "version", "author", "description", "modelConfig", "model_config",
                "modelData", "tokenCount", "token_count"
            }},
        )
