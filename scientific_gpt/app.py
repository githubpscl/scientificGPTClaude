import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_handler import extract_text_from_pdf
from utils.rag import build_vectorstore, load_vectorstore, query_rag
from utils.semantic_scholar import search_papers, format_citation

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Scientific GPT",
    page_icon="🔬",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
if "search_results" not in st.session_state:
    st.session_state.search_results: list[dict] = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files: list[str] = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
        help="Get your key at https://aistudio.google.com/",
    )

    st.divider()

    citation_style = st.radio(
        "Citation Style",
        options=["APA", "MLA", "IEEE"],
        index=0,
        help="Applies to Semantic Scholar results AND inline citations in RAG answers.",
    )

    st.divider()

    if st.session_state.indexed_files:
        st.markdown("**Indexed PDFs**")
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
        if st.button("🗑️ Clear Index"):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            import shutil
            if os.path.exists("vectorstore/faiss_index"):
                shutil.rmtree("vectorstore/faiss_index")
            st.rerun()

    st.divider()
    st.caption("Scientific GPT · Gemini 1.5 Flash · FAISS · Semantic Scholar")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Scientific GPT")
st.markdown("*AI-powered academic research assistant*")

if not api_key:
    st.warning("Please enter your Google Gemini API key in the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rag, tab_search = st.tabs(["📄 PDF / RAG", "🔍 Semantic Scholar"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PDF Upload & RAG
# ════════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.header("Ask Questions About Your PDFs")
    st.caption(
        "Upload papers → build a FAISS index with Gemini Embeddings → "
        "ask questions. Answers include inline **[N]** citations and a "
        f"**{citation_style}** reference list."
    )

    # ── Upload ────────────────────────────────────────────────────────────────
    uploaded_files = st.file_uploader(
        "Upload PDF papers (one or more)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Build / Update FAISS Index", type="primary"):
            with st.spinner("Extracting text and computing Gemini embeddings…"):
                full_text = ""
                names = []
                for f in uploaded_files:
                    text = extract_text_from_pdf(f)
                    if text.strip():
                        full_text += f"\n\n=== {f.name} ===\n\n" + text
                        names.append(f.name)
                    else:
                        st.warning(f"Could not extract text from **{f.name}** — skipped.")

                if not full_text.strip():
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    vs = build_vectorstore(full_text, api_key)
                    st.session_state.vectorstore = vs
                    st.session_state.indexed_files = names
                    st.success(
                        f"Index built from {len(names)} file(s): "
                        + ", ".join(f"**{n}**" for n in names)
                    )

    # ── Q&A ───────────────────────────────────────────────────────────────────
    st.divider()

    rag_question = st.text_area(
        "Research question",
        placeholder="e.g. What are the main limitations of the proposed methodology?",
        height=100,
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_rag = st.button("🧠 Ask (RAG)", type="primary", key="rag_btn")
    with col_info:
        st.caption(
            f"Answers will use **{citation_style}** style for the reference list. "
            "Change the style in the sidebar — it applies immediately to the next query."
        )

    if run_rag:
        vs = st.session_state.vectorstore
        if vs is None:
            vs = load_vectorstore(api_key)
            if vs is not None:
                st.session_state.vectorstore = vs

        if vs is None:
            st.warning("No FAISS index found. Please upload and process PDFs first.")
        elif not rag_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Querying Gemini with RAG context…"):
                answer = query_rag(
                    question=rag_question,
                    vectorstore=vs,
                    api_key=api_key,
                    citation_style=citation_style,
                )
            st.markdown("### Answer")
            st.markdown(answer)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Semantic Scholar Search
# ════════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Search Semantic Scholar")
    st.caption(
        "Results include full metadata stored in session state. "
        f"Citations are formatted in **{citation_style}** — change style in the sidebar."
    )

    col_q, col_n = st.columns([4, 1])
    with col_q:
        search_query = st.text_input(
            "Search query",
            placeholder="e.g. transformer attention mechanism",
        )
    with col_n:
        result_limit = st.number_input("Results", min_value=1, max_value=20, value=5)

    if st.button("🔍 Search", type="primary"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            with st.spinner("Querying Semantic Scholar API…"):
                try:
                    papers = search_papers(search_query, limit=int(result_limit))
                    st.session_state.search_results = papers
                except Exception as e:
                    st.error(f"Search failed: {e}")
                    st.session_state.search_results = []

    # ── Results (re-rendered on every run so style change is instant) ─────────
    papers = st.session_state.search_results
    if papers:
        st.success(f"{len(papers)} result(s) · Citations formatted as **{citation_style}**")

        for i, paper in enumerate(papers, 1):
            title = paper.get("title") or "No Title"
            year = paper.get("year") or "n.d."
            authors = [a.get("name", "") for a in paper.get("authors", [])]
            abstract = paper.get("abstract") or ""
            url = paper.get("url") or ""
            citations = paper.get("citationCount")
            venue = paper.get("venue") or ""

            with st.expander(f"{i}. {title} ({year})"):
                cols = st.columns(3)
                cols[0].markdown(f"**Authors**  \n{', '.join(authors) or 'Unknown'}")
                cols[1].markdown(f"**Venue**  \n{venue or '—'}")
                cols[2].markdown(f"**Citations**  \n{citations if citations is not None else '—'}")

                if abstract:
                    st.markdown("**Abstract**")
                    st.markdown(abstract)

                if url:
                    st.markdown(f"**Link:** [{url}]({url})")

                # Citation block — reflects current sidebar selection live
                citation_text = format_citation(paper, citation_style)
                st.markdown(f"**{citation_style} Citation**")
                st.code(citation_text, language=None)

                col_copy, _ = st.columns([1, 3])
                col_copy.button(
                    "📋 Copy",
                    key=f"copy_{i}",
                    on_click=lambda c=citation_text: st.write(c),
                    help="Click to display the citation text for manual copy",
                )
    elif not papers and "search_results" in st.session_state:
        pass  # no search run yet — show nothing
