import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

VECTORSTORE_PATH = "vectorstore/faiss_index"

# Academic system prompt — citation numbers map to numbered context chunks passed in.
# The {citation_style} token is filled at query time so the reference list format adapts.
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


def build_vectorstore(text: str, api_key: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    # Batch embeddings — Gemini rejects >100 texts per call and large PDFs
    # easily produce 500+ chunks.
    BATCH = 90
    if len(chunks) <= BATCH:
        vs = FAISS.from_texts(chunks, embeddings)
    else:
        vs = FAISS.from_texts(chunks[:BATCH], embeddings)
        for i in range(BATCH, len(chunks), BATCH):
            vs.add_texts(chunks[i:i + BATCH])
    vs.save_local(VECTORSTORE_PATH)
    return vs


def load_vectorstore(api_key: str) -> FAISS | None:
    if not os.path.exists(VECTORSTORE_PATH):
        return None
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )
    return FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )


def query_rag(
    question: str,
    vectorstore: FAISS,
    api_key: str,
    citation_style: str = "APA",
    source_labels: list[str] | None = None,
) -> str:
    """Query the RAG chain.

    source_labels: optional list of human-readable labels for each retrieved chunk,
    e.g. ["Smith et al. (2023), p.3", ...]. When provided they are embedded in the
    context so the model can build accurate reference entries.
    """
    docs = vectorstore.similarity_search(question, k=4)

    context_parts = []
    for i, doc in enumerate(docs, 1):
        label = (source_labels[i - 1] if source_labels and i - 1 < len(source_labels)
                 else f"Source {i}")
        context_parts.append(f"[{i}] ({label})\n{doc.page_content.strip()}")

    context = "\n\n".join(context_parts)

    prompt = _SYSTEM.format(
        citation_style=citation_style,
        context=context,
        question=question,
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.1,
    )
    response = llm.invoke(prompt)
    return response.content
