"""Unified LLM provider abstraction.

Lets the rest of the app stay provider-agnostic: it asks for a chat model or
embeddings via LLMConfig, and we return a LangChain client for whichever
backend the user picked (Google Gemini, OpenAI, or Anthropic Claude).

Claude has no embeddings endpoint — ``has_embeddings`` lets callers skip
embedding-dependent flows (RAG indexing, semantic re-ranking) gracefully
instead of crashing at call time.
"""
from __future__ import annotations
from dataclasses import dataclass

GOOGLE = "Google Gemini"
OPENAI = "OpenAI (ChatGPT)"
ANTHROPIC = "Anthropic (Claude)"

PROVIDERS: dict[str, dict] = {
    GOOGLE: {
        "chat_model": "gemini-2.0-flash",
        "embed_model": "models/gemini-embedding-001",
        "has_embeddings": True,
        "key_label": "Google Gemini API Key",
        "key_hint": "https://aistudio.google.com/apikey",
    },
    OPENAI: {
        "chat_model": "gpt-4o-mini",
        "embed_model": "text-embedding-3-small",
        "has_embeddings": True,
        "key_label": "OpenAI API Key",
        "key_hint": "https://platform.openai.com/api-keys",
    },
    ANTHROPIC: {
        "chat_model": "claude-sonnet-4-5",
        "embed_model": None,
        "has_embeddings": False,
        "key_label": "Anthropic API Key",
        "key_hint": "https://console.anthropic.com/settings/keys",
    },
}


@dataclass
class LLMConfig:
    provider: str
    api_key: str

    @property
    def has_embeddings(self) -> bool:
        return PROVIDERS[self.provider]["has_embeddings"]

    @property
    def chat_model(self) -> str:
        return PROVIDERS[self.provider]["chat_model"]

    @property
    def embed_model(self) -> str | None:
        return PROVIDERS[self.provider]["embed_model"]


def get_chat_llm(config: LLMConfig, temperature: float = 0.1):
    if config.provider == GOOGLE:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=config.chat_model,
            google_api_key=config.api_key,
            temperature=temperature,
        )
    if config.provider == OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.chat_model,
            api_key=config.api_key,
            temperature=temperature,
        )
    if config.provider == ANTHROPIC:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=config.chat_model,
            api_key=config.api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unknown provider: {config.provider}")


def get_embeddings(config: LLMConfig):
    """Return a LangChain embeddings client, or None if the provider has none."""
    if not config.has_embeddings:
        return None
    if config.provider == GOOGLE:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(
            model=config.embed_model,
            google_api_key=config.api_key,
        )
    if config.provider == OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=config.embed_model,
            api_key=config.api_key,
        )
    raise ValueError(f"Embeddings not implemented for provider: {config.provider}")
