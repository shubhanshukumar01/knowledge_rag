"""Exception types shared across the package.

They all derive from RAGError so the command-line tools can catch one type and print a clean
message instead of a stack trace.
"""


class RAGError(Exception):
    """Base class for errors that should be shown to the user as a plain sentence."""


class ConfigError(RAGError):
    """config.yaml is missing, malformed, or asks for something impossible (unknown provider, no key)."""


class MissingDependency(RAGError):
    """An optional package needed for the selected provider or file type is not installed."""


class ProviderError(RAGError):
    """An LLM or embedding provider call failed in a way worth explaining (connectivity, model name, refusal)."""
