"""
llm_client.py — Thin wrapper around a local Ollama instance.

Responsibilities:
  - Send (system_prompt, user_prompt) pairs to the model.
  - Measure wall-clock latency in milliseconds.
  - Extract real prompt + completion token counts from the Ollama response.
  - Return a structured LLMResponse so callers never touch raw HTTP.
"""
from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration — change these to point at your Ollama instance / model.
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
DEFAULT_MODEL: str = "qwen2.5:7b-instruct"


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------
@dataclass
class LLMResponse:
    """Structured result returned by every call to `chat()`."""
    content: str          # The assistant's reply text
    prompt_tokens: int    # Tokens consumed by the prompt (from Ollama)
    completion_tokens: int  # Tokens generated in the completion (from Ollama)
    total_tokens: int     # prompt_tokens + completion_tokens
    latency_ms: int       # Wall-clock time of the HTTP round-trip, in ms


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    temperature: float = 0.0,
    response_format: str | None = None,
) -> LLMResponse:
    """
    Send a system + user message to an Ollama model and return an LLMResponse.

    Parameters
    ----------
    system_prompt:
        The system role message that sets the agent's behaviour.
    user_prompt:
        The user turn content (question, context, etc.).
    model:
        Ollama model tag.  Defaults to DEFAULT_MODEL.
    base_url:
        Base URL of the running Ollama server.
    temperature:
        Sampling temperature.  Use 0.0 for deterministic / eval tasks.
    response_format:
        When set to ``"json"``, passes ``format="json"`` to Ollama, which
        instructs the model to guarantee that its output is a valid JSON
        object.  Leave as ``None`` for free-form text responses (Actor,
        Reflector in non-strict mode, etc.).

    Returns
    -------
    LLMResponse
        Parsed assistant reply with token counts and measured latency.

    Raises
    ------
    RuntimeError
        If the HTTP request fails or the response cannot be parsed.
    """
    url = f"{base_url}/api/chat"

    body: dict = {
        "model": model,
        "stream": False,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Ollama accepts a top-level "format" key to constrain output grammar.
    # Passing "json" guarantees the completion is a parseable JSON object,
    # which eliminates markdown fences and prose preambles at the model level.
    if response_format == "json":
        body["format"] = "json"

    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t_start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except Exception as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    t_end = time.perf_counter()

    latency_ms = int((t_end - t_start) * 1000)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse Ollama response as JSON: {exc}") from exc

    # Extract content ---------------------------------------------------------
    try:
        content: str = data["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected Ollama response structure (missing message.content): {data}"
        ) from exc

    # Extract token counts (Ollama uses prompt_eval_count / eval_count) -------
    prompt_tokens: int = data.get("prompt_eval_count", 0)
    completion_tokens: int = data.get("eval_count", 0)
    total_tokens: int = prompt_tokens + completion_tokens

    return LLMResponse(
        content=content.strip(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
    )
