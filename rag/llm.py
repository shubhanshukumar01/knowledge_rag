"""Chat-model adapters with one interface: generate(messages) -> str and stream(messages) -> pieces.

messages use the OpenAI shape [{"role": "system"|"user"|"assistant", "content": str}]. Adapters convert it
to each provider's native format (Anthropic and Gemini take the system prompt separately).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator

from .config import Config, require_api_key
from .errors import ConfigError, MissingDependency, ProviderError
from .utils import ollama_post, ollama_stream

Message = dict[str, str]


def split_system(messages: list[Message]) -> tuple[str, list[Message]]:
    """Anthropic and Gemini take the system prompt as a separate argument; pull it out of the list."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    conversation = [m for m in messages if m["role"] != "system"]
    return system, conversation


class LLM(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Return the complete answer text."""

    def stream(self, messages: list[Message]) -> Iterator[str]:
        """Yield the answer in pieces. Default: one piece, from generate()."""
        yield self.generate(messages)


# ---------------------------------------------------------------------------------------
# OpenAI and Grok (OpenAI-compatible)

class OpenAICompatLLM(LLM):
    def __init__(self, api_key: str, model: str, base_url: str | None = None, temperature: float | None = 0.2,
                 max_tokens: int | None = 8192, label: str = "openai") -> None:
        try:
            from openai import BadRequestError, OpenAI
        except ImportError:
            raise MissingDependency(f"The openai package is needed for {label}:  pip install openai") from None
        self._bad_request = BadRequestError
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_openai = not base_url
        self.name = f"{label}/{model}"

    def _params(self, messages: list[Message], stream: bool) -> dict:
        params: dict = {"model": self.model, "messages": messages, "stream": stream}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_tokens:
            # OpenAI's newer models only accept max_completion_tokens; compatible APIs expect max_tokens.
            params["max_completion_tokens" if self.is_openai else "max_tokens"] = self.max_tokens
        return params

    def _create(self, messages: list[Message], stream: bool):
        try:
            return self.client.chat.completions.create(**self._params(messages, stream))
        except self._bad_request as exc:
            # Reasoning models (gpt-5 family, o-series) accept only the default temperature: retry without it.
            if self.temperature is not None and "temperature" in str(exc).lower():
                self.temperature = None
                return self.client.chat.completions.create(**self._params(messages, stream))
            raise

    def generate(self, messages):
        response = self._create(messages, stream=False)
        return (response.choices[0].message.content or "").strip()

    def stream(self, messages):
        for chunk in self._create(messages, stream=True):
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


# ------------------------------------------------------------------------------------
# Anthropic (native SDK)

class AnthropicLLM(LLM):
    def __init__(self, api_key: str | None, model: str = "claude-opus-5-5", max_tokens: int = 8192,
                 effort: str | None = "low", refusal_fallbacks: bool = True) -> None:
        try:
            import anthropic
        except ImportError:
            raise MissingDependency("The anthropic package is needed for Anthropic:  pip install anthropic") from None
        # api_key=None lets the SDK use ANTHROPIC_API_KEY or an `ant auth login` profile.
        self.client = anthropic.Anthropic(api_key=api_key or None)
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.refusal_fallbacks = refusal_fallbacks
        self.name = f"anthropic/{model}"

    def _request(self, messages: list[Message]) -> dict:
        system, conversation = split_system(messages)
        params: dict = {"model": self.model, "max_tokens": self.max_tokens, "messages": conversation}
        if system:
            params["system"] = system
        if self.effort:
            # Claude 5.x: thinking is adaptive and always on; `effort` sets its depth. temperature is not accepted.
            params["output_config"] = {"effort": self.effort}
        return params

    def _api(self):
        if self.refusal_fallbacks:
            # Server-side refusal fallbacks: if a safety classifier declines the request, the API re-runs it on a
            # fallback model inside the same call instead of returning an empty answer.
            return self.client.beta.messages, {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return self.client.messages, {}

    def generate(self, messages):
        api, extra = self._api()
        response = api.create(**self._request(messages), **extra)
        if response.stop_reason == "refusal":
            raise ProviderError(_refusal_message(response))
        return "".join(block.text for block in response.content if block.type == "text").strip()

    def stream(self, messages):
        api, extra = self._api()
        with api.stream(**self._request(messages), **extra) as stream:
            for event in stream:                    # SDK helper events; "text" carries each new piece of the answer
                if event.type == "text":
                    yield event.text
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n[The model declined to continue this answer.]"



def _refusal_message(response) -> str:
    details = getattr(response, "stop_details", None)
    explanation = getattr(details, "explanation", None) if details else None
    return "The model declined this request (safety refusal)" + (f": {explanation}" if explanation else ".")


# ---------------------------------------------------------------------------------------
# Google Gemini (native SDK)

class GeminiLLM(LLM):
    def __init__(self, api_key: str, model: str = "gemini-3.0-flash", temperature: float = 0.2,
                 max_tokens: int = 8192) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise MissingDependency("The google-genai package is needed for Gemini:  pip install google-genai") from None
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.name = f"gemini/{model}"

    def _prepare(self, messages: list[Message]):
        system, conversation = split_system(messages)
        contents = [
            self.types.Content(role="user" if m["role"] == "user" else "model",
                               parts=[self.types.Part(text=m["content"])])
            for m in conversation
        ]
        config = self.types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        return contents, config

    def generate(self, messages):
        contents, config = self._prepare(messages)
        response = self.client.models.generate_content(model=self.model, contents=contents, config=config)
        try:
            text = response.text
        except Exception:  # noqa: BLE001 - .text raises when the response was blocked
            text = None
        if not text:
            raise ProviderError("Gemini returned no text (the response may have been blocked or was empty).")
        return text.strip()

    def stream(self, messages):
        contents, config = self._prepare(messages)
        for chunk in self.client.models.generate_content_stream(model=self.model, contents=contents, config=config):
            try:
                text = chunk.text
            except Exception:  # noqa: BLE001
                text = None
            if text:
                yield text


# ---------------------------------------------------------------------------------------
# Ollama (local, plain HTTP)

class OllamaLLM(LLM):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2", temperature: float = 0.2,
                 max_tokens: int = 8192, num_ctx: int = 8192) -> None:
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.num_ctx = num_ctx
        self.name = f"ollama/{model}"

    def _payload(self, messages: list[Message], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            # num_ctx matters: Ollama's default context (2-4k tokens) would silently truncate our prompt
            # (system rules + several chunks). It must be larger than the context budget in config.yaml.
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens, "num_ctx": self.num_ctx},
        }

    def generate(self, messages):
        data = ollama_post(self.base_url, "/api/chat", self._payload(messages, stream=False))
        return (data.get("message", {}).get("content") or "").strip()

    def stream(self, messages):
        for event in ollama_stream(self.base_url, "/api/chat", self._payload(messages, stream=True)):
            content = event.get("message", {}).get("content")
            if content:
                yield content
            if event.get("done"):
                break


# ---------------------------------------------------------------------------------------

def get_llm(cfg: Config) -> LLM:
    """Build the chat model selected by llm.provider in config.yaml."""
    provider = cfg.llm_provider
    settings = cfg.llm_settings
    if provider == "openai":
        return OpenAICompatLLM(api_key=require_api_key(cfg, "openai"), model=settings.get("model") or "gpt-4.1-mini",
                               base_url=settings.get("base_url") or None, temperature=cfg.temperature,
                               max_tokens=cfg.max_tokens, label="openai")
    if provider == "grok":
        return OpenAICompatLLM(api_key=require_api_key(cfg, "grok"), model=settings.get("model") or "grok-4",
                               base_url=settings.get("base_url") or "https://api.x.ai/v1", temperature=cfg.temperature,
                               max_tokens=cfg.max_tokens, label="grok")
    if provider == "anthropic":
        return AnthropicLLM(api_key=require_api_key(cfg, "anthropic"), model=settings.get("model") or "claude-opus-5-5",
                            max_tokens=cfg.max_tokens, effort=settings.get("effort", "low"),
                            refusal_fallbacks=bool(settings.get("refusal_fallbacks", True)))
    if provider == "gemini":
        return GeminiLLM(api_key=require_api_key(cfg, "gemini"), model=settings.get("model") or "gemini-3.0-flash",
                         temperature=cfg.temperature, max_tokens=cfg.max_tokens)
    if provider == "ollama":
        return OllamaLLM(base_url=settings.get("base_url") or "http://localhost:11434",
                         model=settings.get("model") or "llama3.2", temperature=cfg.temperature,
                         max_tokens=cfg.max_tokens, num_ctx=int(settings.get("num_ctx") or 8192))
    raise ConfigError(f"Unknown llm.provider '{provider}'.")
