# Echostash SDK for Python

The universal prompt SDK. Fetch prompts from any PLP-compliant library and use them with any AI provider.

## Installation

```bash
pip install echostash
```

## Quick Start

```python
from echostash import Echostash
from openai import OpenAI

# Connect to any PLP-compliant server
es = Echostash("https://api.echostash.com", api_key="sk_...")

# Fetch a prompt and use with OpenAI
prompt = es.prompt("welcome-email").get()
message = prompt.with_vars(name="Alice").openai()

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[message]
)
```

## The Fluent API

The SDK is designed to be intuitive and chainable:

```python
# Fetch + substitute + convert in one line
msg = es.prompt("welcome").vars(name="Alice").openai()

# Or step by step
prompt = es.prompt("welcome").get()
rendered = prompt.with_vars(name="Alice")
message = rendered.openai()

# Fetch a specific version
v1 = es.prompt("welcome").version("1.0.0").get()

# Short version alias
v2 = es.prompt("welcome").v("2.0.0").get()
```

## Provider Support

### OpenAI

```python
from openai import OpenAI

message = prompt.with_vars(name="Alice").openai(role="system")
config = prompt.openai_config()

client = OpenAI()
response = client.chat.completions.create(
    **config,
    messages=[message]
)
```

### Anthropic

```python
import anthropic

# As user message
message = prompt.with_vars(name="Bob").anthropic()

# As system message (Anthropic handles system separately)
system = prompt.anthropic_system()
config = prompt.anthropic_config()

client = anthropic.Anthropic()
response = client.messages.create(
    **config,
    system=system,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Google / Gemini

```python
message = prompt.google(role="user")
# or
message = prompt.gemini()

config = prompt.google_config()
```

### Vercel AI SDK (for Node.js interop)

```python
message = prompt.vercel(role="system")
```

### LangChain

```python
from langchain.prompts import PromptTemplate

# As message
message = prompt.langchain(message_type="human")

# As template (with input variables)
template, input_variables = prompt.langchain_template()
prompt_template = PromptTemplate.from_template(template)
```

## Connect to Any PLP Server

The SDK works with any PLP-compliant prompt library:

```python
import os

# Echostash Cloud
es = Echostash("https://api.echostash.com", api_key="sk_...")

# Local PLP server
local = Echostash("http://localhost:3000")

# Your company's prompt registry
corp = Echostash(
    "https://prompts.mycompany.com",
    api_key=os.environ["PROMPT_API_KEY"]
)

# Any PLP-compliant service
other = Echostash("https://plp.example.com")
```

## Working with Prompts

### Fetch Prompts

```python
# Latest version
prompt = es.prompt("marketing/welcome").get()

# Specific version
v1 = es.prompt("marketing/welcome").version("1.0.0").get()

# With variables pre-set
prompt = es.prompt("welcome").vars(name="Alice").get()
```

### Access Content

```python
prompt = es.prompt("welcome").get()

# Raw content (string or list of ContentBlock)
raw = prompt.raw()

# As plain text
text = prompt.text()

# As string (same as text)
print(str(prompt))
```

### Save Prompts

```python
es.save(
    "my-new-prompt",
    content="Hello {{name}}!",
    meta={
        "version": "1.0.0",
        "author": "me"
    }
)
```

### Delete Prompts

```python
es.delete("my-old-prompt")
```

### Server Discovery

```python
info = es.discover()
print(info["plp_version"])   # "1.0"
print(info["capabilities"])  # {"versioning": True, ...}
```

## Context Manager Support

```python
with Echostash("https://api.echostash.com", api_key="sk_...") as es:
    prompt = es.prompt("welcome").get()
    # Connection is properly closed after the block
```

## Type Hints

Full type hint support:

```python
from echostash import (
    Prompt,
    PromptContent,
    PromptMeta,
    ModelConfig,
    LoadedPrompt,
)
from echostash.types import (
    OpenAIMessage,
    AnthropicMessage,
    GoogleMessage,
    VercelMessage,
    LangChainMessage,
)
```

## Advanced: Direct Provider Functions

For advanced use cases, you can import provider functions directly:

```python
from echostash.providers import (
    to_openai,
    to_anthropic,
    to_google,
    to_vercel,
    to_langchain,
    extract_openai_config,
)

content = "Hello {{name}}!"
message = to_openai(content, role="system")
```

## Configuration

```python
es = Echostash(
    "https://api.echostash.com",

    # API key for authentication
    api_key="sk_...",

    # Custom headers
    headers={
        "X-Custom-Header": "value"
    },

    # Request timeout (seconds)
    timeout=10,

    # Default parameter symbol for variable substitution
    default_parameter_symbol="{{}}"
)
```

## Error Handling

```python
from echostash import EchostashError

try:
    prompt = es.prompt("not-found").get()
except EchostashError as e:
    print(e.args[0])      # "HTTP 404"
    print(e.status_code)  # 404
```

## License

MIT
