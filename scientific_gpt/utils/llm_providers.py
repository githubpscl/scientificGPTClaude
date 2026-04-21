"""Unified LLM provider abstraction with quota-aware fallback chain.

Supports: Google Gemini, OpenAI (ChatGPT), Anthropic Claude, Groq (Llama).

LLMChain wraps an ordered list of LLMConfigs. `invoke_chat` tries them in
order and automatically falls back on quota / rate-limit errors — useful when
a bundled free-tier key gets exhausted and a backup key is configured.

Embeddings can't be mixed across providers (different vector dimensions),
so `embed()` picks one provider for BOTH the query and the documents.
"""
from __future__ import annotations
from dataclasses import dataclass, field

GOOGLE = "Google Gemini"
OPENAI = "OpenAI (ChatGPT)"
ANTHROPIC = "Anthropic (Claude)"
GROQ = "Groq (Llama 3.3)"

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
    GROQ: {
        "chat_model": "llama-3.3-70b-versatile",
        "embed_model": None,
        "has_embeddings": False,
        "key_label": "Groq API Key",
        "key_hint": "https://console.groq.com/keys",
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
    if config.provider == GROQ:
        # Groq exposes an OpenAI-compatible endpoint, so reuse ChatOpenAI.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.chat_model,
            api_key=config.api_key,
            base_url="https://api.groq.com/openai/v1",
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


# ── Quota detection ──────────────────────────────────────────────────────────
_QUOTA_MARKERS = (
    "429",
    "quota",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "resource_exhausted",
    "resourceexhausted",
    "exceeded your current quota",
    "too many requests",
    "insufficient_quota",
)


def is_quota_error(exc: BaseException) -> bool:
    """Heuristically detect a provider-side quota/rate-limit error."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    return (
        any(m in msg for m in _QUOTA_MARKERS)
        or name in ("resourceexhausted", "ratelimiterror")
    )


# ── Fallback chain ───────────────────────────────────────────────────────────
@dataclass
class LLMChain:
    """Ordered list of configs with auto-fallback on quota errors.

    last_used / fell_back are set after invoke_chat / embed so the UI can
    show "switched to provider X because primary hit quota".
    """
    configs: list[LLMConfig]
    last_used: LLMConfig | None = None
    fell_back: bool = False

    def __post_init__(self):
        if not self.configs:
            raise ValueError("LLMChain requires at least one LLMConfig")

    @property
    def primary(self) -> LLMConfig:
        return self.configs[0]

    @property
    def has_embeddings(self) -> bool:
        return any(c.has_embeddings for c in self.configs)

    def embeddings_config(self) -> LLMConfig | None:
        """First config in the chain that supports embeddings."""
        for c in self.configs:
            if c.has_embeddings:
                return c
        return None

    def invoke_chat(self, prompt, temperature: float = 0.1) -> str:
        """Invoke chat; on quota error, transparently fall back to the next config."""
        last_exc: BaseException | None = None
        for i, cfg in enumerate(self.configs):
            try:
                llm = get_chat_llm(cfg, temperature=temperature)
                resp = llm.invoke(prompt)
                self.last_used = cfg
                self.fell_back = i > 0
                return resp.content
            except Exception as e:
                last_exc = e
                if is_quota_error(e) and i + 1 < len(self.configs):
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    def embed(self, question: str, documents: list[str]):
        """Embed question + documents with the SAME provider.

        Returns (q_vec, doc_vecs, used_config). Falls back to the next
        embeddings-capable config on quota errors. Raises if no provider with
        embeddings is configured.
        """
        embed_configs = [c for c in self.configs if c.has_embeddings]
        if not embed_configs:
            raise RuntimeError("No provider in the chain supports embeddings.")

        last_exc: BaseException | None = None
        for i, cfg in enumerate(embed_configs):
            try:
                emb = get_embeddings(cfg)
                q_vec = emb.embed_query(question)
                doc_vecs = emb.embed_documents(documents)
                self.last_used = cfg
                self.fell_back = self.configs.index(cfg) > 0
                return q_vec, doc_vecs, cfg
            except Exception as e:
                last_exc = e
                if is_quota_error(e) and i + 1 < len(embed_configs):
                    continue
                raise
        assert last_exc is not None
        raise last_exc
