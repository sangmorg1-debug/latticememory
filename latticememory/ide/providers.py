from __future__ import annotations

import json
import urllib.error
import urllib.request

from .config import IdeConfig


class ProviderError(RuntimeError):
    pass


def chat_completion(
    config: IdeConfig,
    prompt: str,
    *,
    system_prompt: str | None = None,
    timeout: int = 60,
) -> str:
    if not config.base_url:
        raise ProviderError("Provider base URL is not configured.")
    if not config.model:
        raise ProviderError("Provider model is not configured.")
    if not config.api_key:
        raise ProviderError("Provider API key is not configured.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = json.dumps({"model": config.model, "messages": messages}).encode("utf-8")
    request = urllib.request.Request(
        f"{config.base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise ProviderError(f"Provider HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise ProviderError(f"Provider request failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderError("Provider request timed out.") from exc

    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError("Provider response did not contain choices[0].message.content.") from exc
