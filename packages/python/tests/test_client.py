"""Tests for Echostash Python SDK"""

import pytest
from echostash import (
    Echostash,
    LoadedPrompt,
    Prompt,
    PromptMeta,
    to_openai,
    to_anthropic,
    to_google,
    to_vercel,
    to_langchain,
)


class TestLoadedPrompt:
    def test_substitute_variables(self):
        prompt = LoadedPrompt(Prompt(
            id="test",
            content="Hello {{name}}!",
            meta=PromptMeta(),
            parameter_symbol="{{}}"
        ))

        rendered = prompt.with_vars(name="Alice")
        assert rendered.text() == "Hello Alice!"

    def test_handle_missing_variables(self):
        prompt = LoadedPrompt(Prompt(
            id="test",
            content="Hello {{name}}!",
            meta=PromptMeta(),
            parameter_symbol="{{}}"
        ))

        rendered = prompt.with_vars()
        assert rendered.text() == "Hello {{name}}!"

    def test_convert_to_openai(self):
        prompt = LoadedPrompt(Prompt(
            id="test",
            content="Hello world",
            meta=PromptMeta()
        ))

        msg = prompt.openai(role="system")
        assert msg == {"role": "system", "content": "Hello world"}

    def test_convert_to_anthropic(self):
        prompt = LoadedPrompt(Prompt(
            id="test",
            content="Hello world",
            meta=PromptMeta()
        ))

        msg = prompt.anthropic(role="user")
        assert msg == {"role": "user", "content": "Hello world"}


class TestProviderConverters:
    def test_to_openai_string_content(self):
        msg = to_openai("Hello world", role="user")
        assert msg == {"role": "user", "content": "Hello world"}

    def test_to_anthropic_string_content(self):
        msg = to_anthropic("Hello world", role="user")
        assert msg == {"role": "user", "content": "Hello world"}

    def test_to_google_string_content(self):
        msg = to_google("Hello world", role="user")
        assert msg == {"role": "user", "parts": [{"text": "Hello world"}]}

    def test_to_vercel_string_content(self):
        msg = to_vercel("Hello world", role="system")
        assert msg == {"role": "system", "content": "Hello world"}

    def test_to_langchain_string_content(self):
        msg = to_langchain("Hello world", message_type="human")
        assert msg == {"type": "human", "content": "Hello world"}


class TestEchostashClient:
    def test_create_client(self):
        client = Echostash("https://api.example.com", api_key="test-key")
        assert client is not None

    def test_create_prompt_query(self):
        client = Echostash("https://api.example.com")
        query = client.prompt("test-prompt")
        assert query is not None
