from __future__ import annotations

import json

import pytest

from latticememory.ide.config import IdeConfig
from latticememory.ide.providers import ProviderError, chat_completion


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps({"choices": [{"message": {"content": "hello back"}}]}).encode("utf-8")


def test_chat_completion_sends_openai_compatible_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = chat_completion(
        IdeConfig(base_url="https://api.example.com/v1/", model="demo", api_key="secret"),
        "Say hello",
    )

    assert result == "hello back"
    assert captured["url"] == "https://api.example.com/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["body"]["model"] == "demo"
    assert captured["body"]["messages"][-1] == {"role": "user", "content": "Say hello"}
    assert captured["timeout"] == 60


def test_chat_completion_requires_provider_fields():
    with pytest.raises(ProviderError, match="base URL"):
        chat_completion(IdeConfig(model="demo", api_key="secret"), "hi")
    with pytest.raises(ProviderError, match="model"):
        chat_completion(IdeConfig(base_url="https://api.example.com/v1", api_key="secret"), "hi")
    with pytest.raises(ProviderError, match="API key"):
        chat_completion(IdeConfig(base_url="https://api.example.com/v1", model="demo"), "hi")
