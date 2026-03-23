"""
Echostash client - The fluent API for fetching and using prompts.
"""

from __future__ import annotations
import re
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
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
    Message,
    ToolDefinition,
    EchostashConfig,
    Variables,
    OpenAIMessage,
    AnthropicMessage,
    GoogleMessage,
    VercelMessage,
    LangChainMessage,
    VersionSpecifier,
    RenderResponse,
    BatchRenderItem,
    BatchRenderResult,
    BatchRenderResponse,
    ObservationItem,
)
from .providers import (
    to_openai,
    to_openai_prompt_result,
    extract_openai_config,
    to_anthropic,
    to_anthropic_system,
    to_anthropic_prompt_result,
    extract_anthropic_config,
    to_google,
    to_google_prompt_result,
    extract_google_config,
    to_vercel,
    to_vercel_prompt_result,
    to_langchain,
    to_langchain_prompt_result,
    to_langchain_template,
)


def _parse_retry_after(header: Optional[str]) -> Optional[float]:
    """Parse Retry-After header value into seconds."""
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        dt = parsedate_to_datetime(header)
        return max(0.0, (dt.timestamp() - time.time()))
    except Exception:
        pass
    return None


class EchostashError(Exception):
    """Exception raised for Echostash SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retry_after: Optional[float] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429


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


def _substitute_messages_content(
    content_blocks: List[Any],
    variables: Variables,
    parameter_symbol: str,
) -> List[Any]:
    """Substitute variables in message content blocks."""
    if not variables:
        return content_blocks

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

    result = []
    for block in content_blocks:
        if isinstance(block, TextContent):
            result.append(TextContent(text=substitute(block.text)))
        elif isinstance(block, dict) and block.get("type") == "text":
            result.append({**block, "text": substitute(block.get("text", ""))})
        else:
            result.append(block)
    return result


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
        self.tools: List[ToolDefinition] = prompt.tools or []
        self.messages: List[Message] = self._normalize_messages(prompt)

    def _normalize_messages(self, prompt: Prompt) -> List[Message]:
        """Normalize messages from the prompt data.

        If the server returned messages, use them. Otherwise, derive from content.
        """
        if prompt.messages and len(prompt.messages) > 0:
            return prompt.messages

        # Legacy: wrap content in a single user message
        content = prompt.content
        if isinstance(content, str):
            return [Message(role="user", content=[TextContent(text=content)])]
        if isinstance(content, list):
            return [Message(role="user", content=list(content))]
        return [Message(role="user", content=[])]

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

        # Substitute variables in messages too
        new_messages = [
            Message(
                role=msg.role,
                content=_substitute_messages_content(
                    msg.content, variables, self.parameter_symbol
                ),
            )
            for msg in self.messages
        ]

        return LoadedPrompt(Prompt(
            id=self.id,
            name=self.name,
            description=self.description,
            content=new_content,
            meta=self.meta,
            parameter_symbol=self.parameter_symbol,
            messages=new_messages,
            tools=self.tools,
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
    # Provider Conversions (Messages + Tools)
    # --------------------------------------------------------------------------

    def openai(
        self,
        role: Optional[Literal["system", "user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], OpenAIMessage]:
        """Convert to OpenAI format.

        When called with no arguments, returns the full prompt result
        with messages array + tools + model config.

        When called with a role (backward compatible), returns a single
        OpenAI message.

        Example:
            >>> # New: full prompt result
            >>> result = prompt.openai()
            >>> openai.chat.completions.create(**result)
            >>>
            >>> # Legacy: single message
            >>> message = prompt.openai(role="system")
        """
        if role is not None:
            return to_openai(self.content, role)
        return to_openai_prompt_result(self.messages, self.tools, self.meta.model_config)

    def openai_config(self) -> Dict[str, Any]:
        """Get OpenAI-compatible model config from prompt metadata."""
        return extract_openai_config(self.meta.model_config)

    def anthropic(
        self,
        role: Optional[Literal["user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], AnthropicMessage]:
        """Convert to Anthropic format.

        When called with no arguments, returns the full prompt result
        with system + messages + tools.

        When called with a role (backward compatible), returns a single message.

        Example:
            >>> # New: full prompt result
            >>> result = prompt.anthropic()
            >>> anthropic.messages.create(**result)
            >>>
            >>> # Legacy: single message
            >>> message = prompt.anthropic(role="user")
        """
        if role is not None:
            return to_anthropic(self.content, role)
        return to_anthropic_prompt_result(self.messages, self.tools, self.meta.model_config)

    def anthropic_system(self) -> str:
        """Get content as Anthropic system message (string)."""
        return to_anthropic_system(self.content)

    def anthropic_config(self) -> Dict[str, Any]:
        """Get Anthropic-compatible model config from prompt metadata."""
        return extract_anthropic_config(self.meta.model_config)

    def google(
        self,
        role: Optional[Literal["user", "model"]] = None,
    ) -> Union[Dict[str, Any], GoogleMessage]:
        """Convert to Google/Gemini format.

        When called with no arguments, returns the full prompt result
        with contents + tools.

        When called with a role (backward compatible), returns a single message.
        """
        if role is not None:
            return to_google(self.content, role)
        return to_google_prompt_result(self.messages, self.tools, self.meta.model_config)

    def gemini(
        self,
        role: Optional[Literal["user", "model"]] = None,
    ) -> Union[Dict[str, Any], GoogleMessage]:
        """Alias for google()."""
        return self.google(role)

    def google_config(self) -> Dict[str, Any]:
        """Get Google-compatible model config from prompt metadata."""
        return extract_google_config(self.meta.model_config)

    def vercel(
        self,
        role: Optional[Literal["system", "user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], VercelMessage]:
        """Convert to Vercel AI SDK format.

        When called with no arguments, returns the full prompt result
        with messages + tools.

        When called with a role (backward compatible), returns a single message.
        """
        if role is not None:
            return to_vercel(self.content, role)
        return to_vercel_prompt_result(self.messages, self.tools, self.meta.model_config)

    def langchain(
        self,
        message_type: Optional[Literal["system", "human", "ai"]] = None,
    ) -> Union[Dict[str, Any], LangChainMessage]:
        """Convert to LangChain format.

        When called with no arguments, returns the full prompt result
        with messages + tools.

        When called with a message_type (backward compatible), returns a single message.
        """
        if message_type is not None:
            return to_langchain(self.content, message_type)
        return to_langchain_prompt_result(self.messages, self.tools, self.meta.model_config)

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
            "messages": self.messages,
            "tools": self.tools,
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
        >>>
        >>> # Server-side render
        >>> result = es.prompt(1).version("staging").render(name="Alice")
    """

    def __init__(self, client: Echostash, prompt_id: str):
        self._client = client
        self._prompt_id = prompt_id
        self._version: Optional[Union[str, int]] = None
        self._pending_variables: Optional[Variables] = None

    def version(self, version: Union[str, int]) -> PromptQuery:
        """Request a specific version of the prompt.

        Args:
            version: Version number, 'published', 'staging', or a semver string.
        """
        self._version = version
        return self

    def v(self, version: Union[str, int]) -> PromptQuery:
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
        """Fetch the prompt and return a LoadedPrompt (client-side substitution)."""
        version_str = str(self._version) if self._version is not None else None
        prompt = self._client.fetch_prompt(self._prompt_id, version_str)
        loaded = LoadedPrompt(prompt)

        if self._pending_variables:
            loaded = loaded.with_vars(**self._pending_variables)

        return loaded

    def fetch(self) -> LoadedPrompt:
        """Alias for get()."""
        return self.get()

    def render(self, **variables: str) -> RenderResponse:
        """Server-side render: sends variables to the server for rendering.

        Only available in 'echostash' mode.

        Example:
            >>> result = es.prompt(1).version("staging").render(name="Alice")
            >>> print(result.content)  # "Hello Alice!"
        """
        return self._client.render_prompt(
            self._prompt_id, self._version, variables if variables else None
        )

    # --------------------------------------------------------------------------
    # Shorthand methods - fetch + convert in one call
    # --------------------------------------------------------------------------

    def openai(
        self,
        role: Optional[Literal["system", "user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], OpenAIMessage]:
        """Fetch prompt and convert to OpenAI format."""
        return self.get().openai(role)

    def anthropic(
        self,
        role: Optional[Literal["user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], AnthropicMessage]:
        """Fetch prompt and convert to Anthropic format."""
        return self.get().anthropic(role)

    def google(
        self,
        role: Optional[Literal["user", "model"]] = None,
    ) -> Union[Dict[str, Any], GoogleMessage]:
        """Fetch prompt and convert to Google/Gemini format."""
        return self.get().google(role)

    def vercel(
        self,
        role: Optional[Literal["system", "user", "assistant"]] = None,
    ) -> Union[Dict[str, Any], VercelMessage]:
        """Fetch prompt and convert to Vercel AI SDK format."""
        return self.get().vercel(role)

    def langchain(
        self,
        message_type: Optional[Literal["system", "human", "ai"]] = None,
    ) -> Union[Dict[str, Any], LangChainMessage]:
        """Fetch prompt and convert to LangChain format."""
        return self.get().langchain(message_type)

    def text(self) -> str:
        """Fetch prompt and get as plain text."""
        return self.get().text()


class Echostash:
    """The Echostash client - connects to any PLP-compliant prompt library.

    Example:
        >>> # Connect to Echostash Cloud (default echostash mode)
        >>> es = Echostash("https://api.echostash.com", api_key="sk_...")
        >>>
        >>> # Connect in PLP mode
        >>> plp = Echostash("http://localhost:3000", mode="plp")
        >>>
        >>> # Server-side render
        >>> result = es.prompt(1).version("staging").render(name="Alice")
        >>>
        >>> # Batch render
        >>> batch = es.batch_render([
        ...     BatchRenderItem(prompt_id=1, variables={"name": "Alice"}),
        ...     BatchRenderItem(prompt_id=2, version="staging"),
        ... ])
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 10,
        default_parameter_symbol: str = "{{}}",
        mode: Literal["echostash", "plp"] = "echostash",
    ):
        """Initialize the Echostash client.

        Args:
            base_url: The base URL of the server
            api_key: Optional API key for authentication
            headers: Optional additional headers
            timeout: Request timeout in seconds
            default_parameter_symbol: Default symbol for variable substitution
            mode: 'echostash' (default, uses /api/sdk/) or 'plp' (uses /v1/)
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._headers = headers or {}
        self._timeout = timeout
        self._default_parameter_symbol = default_parameter_symbol
        self._mode = mode
        self._session = requests.Session()

        # Observation buffer
        self._observation_buffer: List[Dict[str, Any]] = []
        self._observation_lock = threading.Lock()
        self._observation_timer: Optional[threading.Timer] = None
        self._observations_forbidden = False

    def __enter__(self) -> Echostash:
        return self

    def __exit__(self, *args: Any) -> None:
        self.destroy()
        self._session.close()

    def prompt(self, prompt_id: Union[str, int]) -> PromptQuery:
        """Start building a prompt query.

        Example:
            >>> prompt = es.prompt("marketing/welcome-email").get()
            >>> prompt = es.prompt(123).get()
        """
        return PromptQuery(self, str(prompt_id))

    def get(self, prompt_id: Union[str, int]) -> PromptQuery:
        """Alias for prompt()."""
        return self.prompt(prompt_id)

    def fetch_prompt(self, prompt_id: str, version: Optional[str] = None) -> Prompt:
        """Internal method to fetch a prompt from the server."""
        if self._mode == "echostash":
            if version:
                path = f"/api/sdk/prompts/{prompt_id}/versions/{version}"
            else:
                path = f"/api/sdk/prompts/{prompt_id}"
        else:
            # PLP mode
            if version:
                path = f"/v1/prompts/{prompt_id}/{version}"
            else:
                path = f"/v1/prompts/{prompt_id}"

        response = self._request("GET", path)
        return self._normalize_prompt(response)

    def render_prompt(
        self,
        prompt_id: str,
        version: Optional[Union[str, int]] = None,
        variables: Optional[Dict[str, str]] = None,
    ) -> RenderResponse:
        """Server-side render a prompt. Only available in 'echostash' mode."""
        if self._mode != "echostash":
            raise EchostashError("Server-side render is only available in echostash mode")

        body: Dict[str, Any] = {}
        if version is not None:
            body["version"] = version
        if variables:
            body["variables"] = variables

        data = self._request("POST", f"/api/sdk/prompts/{prompt_id}/render", body)
        return RenderResponse(
            content=data["content"],
            prompt_id=data["promptId"],
            version_no=data["versionNo"],
        )

    def batch_render(self, items: List[BatchRenderItem]) -> BatchRenderResponse:
        """Batch render multiple prompts in a single request.

        Only available in 'echostash' mode. Maximum 50 items.

        Example:
            >>> result = es.batch_render([
            ...     BatchRenderItem(prompt_id=1, version="published", variables={"name": "Alice"}),
            ...     BatchRenderItem(prompt_id=2, variables={"greeting": "Hello"}),
            ... ])
            >>> print(result.results["1"].content)
        """
        if self._mode != "echostash":
            raise EchostashError("Batch render is only available in echostash mode")

        if len(items) > 50:
            raise EchostashError(f"Batch render supports a maximum of 50 items, got {len(items)}")

        request_items = []
        for item in items:
            entry: Dict[str, Any] = {"promptId": item.prompt_id}
            if item.version is not None:
                entry["version"] = item.version
            if item.variables:
                entry["variables"] = item.variables
            request_items.append(entry)

        data = self._request("POST", "/api/sdk/prompts/batch", {"items": request_items})

        results: Dict[str, BatchRenderResult] = {}
        for key, val in data["results"].items():
            results[key] = BatchRenderResult(
                content=val["content"],
                version_no=val["versionNo"],
                error=val.get("error"),
            )

        return BatchRenderResponse(
            results=results,
            success_count=data["successCount"],
            error_count=data["errorCount"],
        )

    # --------------------------------------------------------------------------
    # Observations - Client-side render metrics reporting
    # --------------------------------------------------------------------------

    def observe_render(self, observation: ObservationItem) -> None:
        """Record an observation from a client-side render.

        Observations are buffered and sent to the server every 60 seconds.
        Only available in 'echostash' mode. Silently ignored for free users (403).

        Example:
            >>> import time
            >>> start = time.time()
            >>> rendered = prompt.with_vars(name="Alice")
            >>> es.observe_render(ObservationItem(
            ...     prompt_id=123, version_no=1,
            ...     latency_ms=int((time.time() - start) * 1000),
            ...     success=True, variable_keys=["name"],
            ... ))
        """
        if self._mode != "echostash" or self._observations_forbidden:
            return

        item: Dict[str, Any] = {
            "promptId": observation.prompt_id,
            "versionNo": observation.version_no,
            "latencyMs": observation.latency_ms,
            "success": observation.success,
        }
        if observation.variable_keys is not None:
            item["variableKeys"] = observation.variable_keys
        item["timestamp"] = observation.timestamp or datetime.now(timezone.utc).isoformat()

        with self._observation_lock:
            self._observation_buffer.append(item)

            # Start the flush timer on first observation
            if self._observation_timer is None:
                self._start_flush_timer()

    def _start_flush_timer(self) -> None:
        """Start a daemon timer to flush observations every 60 seconds."""
        self._observation_timer = threading.Timer(60.0, self._flush_timer_callback)
        self._observation_timer.daemon = True
        self._observation_timer.start()

    def _flush_timer_callback(self) -> None:
        """Timer callback that flushes and reschedules."""
        self.flush()
        with self._observation_lock:
            if self._observation_buffer and not self._observations_forbidden:
                self._start_flush_timer()
            else:
                self._observation_timer = None

    def flush(self) -> None:
        """Manually flush buffered observations to the server."""
        with self._observation_lock:
            if not self._observation_buffer or self._observations_forbidden:
                return
            items = list(self._observation_buffer)
            self._observation_buffer.clear()

        try:
            self._request("POST", "/api/sdk/observations", {"items": items})
        except EchostashError as e:
            if e.status_code == 403:
                self._observations_forbidden = True
                with self._observation_lock:
                    if self._observation_timer:
                        self._observation_timer.cancel()
                        self._observation_timer = None
                return
            # On other errors, put items back for retry
            with self._observation_lock:
                self._observation_buffer = items + self._observation_buffer

    def destroy(self) -> None:
        """Stop the observation flush timer and flush remaining observations."""
        with self._observation_lock:
            if self._observation_timer:
                self._observation_timer.cancel()
                self._observation_timer = None
        self.flush()

    def save(
        self,
        prompt_id: str,
        content: PromptContent,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LoadedPrompt:
        """Save a prompt to the server (PLP mode only).

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
        """Delete a prompt from the server (PLP mode only)."""
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
        _attempt: int = 0,
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
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))

            if response.status_code == 429 and _attempt < 3:
                delay = retry_after if retry_after is not None else min(1.0 * (2 ** _attempt), 8.0)
                time.sleep(delay)
                return self._request(method, path, body, _attempt + 1)

            error_message = f"HTTP {response.status_code}"
            try:
                error_body = response.json()
                error_message = error_body.get("message") or error_body.get("error") or error_message
            except Exception:
                pass
            raise EchostashError(error_message, response.status_code, retry_after)

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
                messages=self._normalize_server_messages(data.get("messages")),
                tools=self._normalize_server_tools(data.get("tools")),
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
                messages=self._normalize_server_messages(data.get("messages")),
                tools=self._normalize_server_tools(data.get("tools")),
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

    def _normalize_server_messages(
        self, messages: Optional[List[Any]]
    ) -> Optional[List[Message]]:
        """Normalize messages from the server response."""
        if not messages or not isinstance(messages, list):
            return None

        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")
            content = msg.get("content", [])

            if isinstance(content, str):
                content_blocks: List[Any] = [TextContent(text=content)]
            elif isinstance(content, list):
                content_blocks = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_blocks.append(TextContent(text=block.get("text", "")))
                        elif block.get("type") == "image_url":
                            img_url = block.get("image_url", {})
                            if isinstance(img_url, str):
                                content_blocks.append(ImageContent(image_url=ImageUrl(url=img_url)))
                            else:
                                content_blocks.append(ImageContent(
                                    image_url=ImageUrl(
                                        url=img_url.get("url", ""),
                                        detail=img_url.get("detail"),
                                    )
                                ))
                        else:
                            content_blocks.append(TextContent(text=str(block.get("text", block))))
            else:
                content_blocks = []

            result.append(Message(role=role, content=content_blocks))

        return result if result else None

    def _normalize_server_tools(
        self, tools: Optional[List[Any]]
    ) -> Optional[List[ToolDefinition]]:
        """Normalize tool definitions from the server response."""
        if not tools or not isinstance(tools, list):
            return None

        result = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function", {})
            result.append(ToolDefinition(
                type="function",
                function={
                    "name": func.get("name", "") or tool.get("name", ""),
                    "description": func.get("description", "") or tool.get("description", ""),
                    "parameters": func.get("parameters", {}) or tool.get("parameters", {}),
                },
            ))

        return result if result else None
