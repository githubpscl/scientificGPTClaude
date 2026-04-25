import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from .llm_backend import LLMChain, get_embeddings

VECTORSTORE_PATH = "vectorstore/faiss_index"


class EmbeddingsUnavailableError(RuntimeError):
    """Raised when no provider in the chain offers embeddings."""


_SYSTEM = """You are a rigorous scientific research assistant. Follow these rules strictly:

1. Answer in formal, objective academic language.
2. Cite every factual claim inline using bracketed numbers matching the sources below, e.g. [1], [2].
3. You may combine sources: [1, 3] or use ranges: [1–3].
4. At the very end of your answer, output a section titled "## References" that lists every cited source formatted in **{citation_style}** style.
5. If the context does not contain enough information to answer, state this explicitly — do not speculate.
6. Never repeat the question in your answer.

--- Context Sources ---
{context}
--- End of Sources ---

Question: {question}

Answer (with inline [N] citations and a ## References section at the end):"""


def _pick_embeddings(chain: LLMChain):
    cfg = chain.embeddings_config()
    if cfg is None:
        raise EmbeddingsUnavailableError(
            f"None of the configured providers ({', '.join(c.provider for c in chain.configs)}) "
            f"offers an embeddings endpoint. PDF indexing requires embeddings — "
            f"switch to Google Gemini or OpenAI."
        )
    return get_embeddings(cfg)


def build_vectorstore(docs: list[tuple[str, str]], chain: LLMChain) -> FAISS:
    """Build/replace the FAISS index from a list of (filename, text) pairs.

    Each chunk keeps its source filename in metadata so PDF references can
    be rendered correctly in combined-mode answers.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    all_texts: list[str] = []
    all_meta: list[dict] = []
    for filename, text in docs:
        chunks = splitter.split_text(text)
        all_texts.extend(chunks)
        all_meta.extend({"source": filename} for _ in chunks)

    embeddings = _pick_embeddings(chain)
    # Batch — Gemini rejects >100 texts per call; 90 is safe across providers.
    BATCH = 90
    vs: FAISS | None = None
    for i in range(0, len(all_texts), BATCH):
        batch_texts = all_texts[i:i + BATCH]
        batch_meta = all_meta[i:i + BATCH]
        if vs is None:
            vs = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_meta)
        else:
            vs.add_texts(batch_texts, metadatas=batch_meta)
    assert vs is not None
    vs.save_local(VECTORSTORE_PATH)
    return vs


def load_vectorstore(chain: LLMChain) -> FAISS | None:
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    embeddings = _pick_embeddings(chain)
    return FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )


def retrieve_chunks(vectorstore: FAISS, question: str, k: int = 4):
    """Return the top-k PDF chunks relevant to the question."""
    return vectorstore.similarity_search(question, k=k)


def _build_rag_prompt(
    question: str,
    vectorstore: FAISS,
    citation_style: str,
    source_labels: list[str] | None,
) -> str:
    docs = vectorstore.similarity_search(question, k=4)
    context_parts = []
    for i, doc in enumerate(docs, 1):
        label = (source_labels[i - 1] if source_labels and i - 1 < len(source_labels)
                 else doc.metadata.get("source", f"Source {i}"))
        context_parts.append(f"[{i}] ({label})\n{doc.page_content.strip()}")
    context = "\n\n".join(context_parts)
    return _SYSTEM.format(
        citation_style=citation_style,
        context=context,
        question=question,
    )


def query_rag(
    question: str,
    vectorstore: FAISS,
    chain: LLMChain,
    citation_style: str = "APA",
    source_labels: list[str] | None = None,
) -> str:
    prompt = _build_rag_prompt(question, vectorstore, citation_style, source_labels)
    return chain.invoke_chat(prompt, temperature=0.1)


def stream_rag(
    question: str,
    vectorstore: FAISS,
    chain: LLMChain,
    citation_style: str = "APA",
    source_labels: list[str] | None = None,
):
    """Streaming variant of query_rag."""
    prompt = _build_rag_prompt(question, vectorstore, citation_style, source_labels)
    return chain.stream_chat(prompt, temperature=0.1)
