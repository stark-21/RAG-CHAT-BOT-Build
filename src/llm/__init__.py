"""P5 - LLM layer: prompt building, providers, and output validation."""

from src.llm.generator import (
    cap_sentences,
    extract_urls,
    generate,
    numbers_in,
    parse_answer,
)
from src.llm.prompt_builder import SYSTEM_RULES, build_prompt, context_budget
from src.llm.provider import LLMProvider, ProviderError, StubProvider, get_provider

__all__ = [
    "LLMProvider",
    "ProviderError",
    "SYSTEM_RULES",
    "StubProvider",
    "build_prompt",
    "cap_sentences",
    "context_budget",
    "extract_urls",
    "generate",
    "get_provider",
    "numbers_in",
    "parse_answer",
]
