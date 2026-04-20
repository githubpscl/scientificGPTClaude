import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_handler import extract_text_from_pdf
from utils.rag import build_vectorstore, load_vectorstore, query_rag
from utils.search_engine import search as multi_search, available_sources, is_source_available, SOURCE_COLORS
from utils.sources.crossref import lookup_doi

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scientific GPT", page_icon="🔬", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("search_results", []),
    ("search_errors", {}),
    ("vectorstore", None),
    ("indexed_files", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = st.text_input(
        "Google Gemini API Key",
        value=os.getenv("GOOGLE_API_KEY", ""),
        type="password",
    )

    st.divider()

    citation_style = st.radio("Citation Style", ["APA", "MLA", "IEEE"], index=0)

    st.divider()
    st.markdown("**Active PDF Index**")
    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
        if st.button("🗑️ Clear Index"):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            import shutil
            if os.path.exists("vectorstore/faiss_index"):
                shutil.rmtree("vectorstore/faiss_index")
            st.rerun()
    else:
        st.caption("No PDFs indexed yet.")

    st.divider()
    st.caption("Scientific GPT · Gemini 1.5 Flash · FAISS\nSemantic Scholar · OpenAlex · arXiv · PubMed · Crossref · CORE")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Scientific GPT")
st.markdown("*AI-powered academic research assistant — 6 sources, one search*")

if not api_key:
    st.warning("Please enter your Google Gemini API key in the sidebar.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_rag, tab_search, tab_doi = st.tabs(["📄 PDF / RAG", "🔍 Multi-Source Search", "🔗 DOI Lookup"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PDF / RAG
# ════════════════════════════════════════════════════════════════════════════════
with tab_rag:
    st.header("Ask Questions About Your PDFs")
    st.caption(
        f"Upload papers → FAISS index via Gemini Embeddings → answers with "
        f"inline **[N]** citations and a **{citation_style}** reference list."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF papers", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("📥 Build / Update FAISS Index", type="primary"):
            with st.spinner("Extracting text and computing Gemini embeddings…"):
                full_text, names = "", []
                for f in uploaded_files:
                    t = extract_text_from_pdf(f)
                    if t.strip():
                        full_text += f"\n\n=== {f.name} ===\n\n" + t
                        names.append(f.name)
                    else:
                        st.warning(f"No text extracted from **{f.name}** — skipped.")
                if not full_text.strip():
                    st.error("No text could be extracted.")
                else:
                    vs = build_vectorstore(full_text, api_key)
                    st.session_state.vectorstore = vs
                    st.session_state.indexed_files = names
                    st.success(f"Index built from: {', '.join(f'**{n}**' for n in names)}")

    st.divider()
    rag_question = st.text_area(
        "Research question",
        placeholder="e.g. What are the main limitations of the proposed methodology?",
        height=100,
    )
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_rag = st.button("🧠 Ask (RAG)", type="primary")
    with col_info:
        st.caption(f"Style in sidebar = **{citation_style}** — change applies to next query.")

    if run_rag:
        vs = st.session_state.vectorstore or load_vectorstore(api_key)
        if vs is None:
            st.warning("No FAISS index. Upload and process PDFs first.")
        elif not rag_question.strip():
            st.warning("Please enter a question.")
        else:
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = vs
            with st.spinner("Querying Gemini…"):
                answer = query_rag(rag_question, vs, api_key, citation_style)
            st.markdown("### Answer")
            st.markdown(answer)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-Source Search
# ════════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Multi-Source Academic Search")

    # ── Query + options row ───────────────────────────────────────────────────
    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input("Search query", placeholder="e.g. transformer attention mechanism")
    with col_n:
        limit = st.number_input("Per source", min_value=1, max_value=15, value=5)

    # ── Source selection ──────────────────────────────────────────────────────
    st.markdown("**Sources**")
    all_sources = available_sources()
    cols = st.columns(len(all_sources))
    selected_sources: list[str] = []
    for col, name in zip(cols, all_sources):
        available = is_source_available(name)
        color = SOURCE_COLORS.get(name, "#888")
        label = name if available else f"{name} *(key needed)*"
        checked = col.checkbox(label, value=available, disabled=not available, key=f"src_{name}")
        if checked and available:
            selected_sources.append(name)

    if st.button("🔍 Search All Selected Sources", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query.")
        elif not selected_sources:
            st.warning("Please select at least one source.")
        else:
            with st.spinner(f"Searching {len(selected_sources)} source(s) in parallel…"):
                papers, errors = multi_search(query, selected_sources, int(limit))
                st.session_state.search_results = papers
                st.session_state.search_errors = errors

    # ── Error notices ─────────────────────────────────────────────────────────
    for src, err in st.session_state.search_errors.items():
        st.warning(f"**{src}** failed: {err}")

    # ── Results ───────────────────────────────────────────────────────────────
    papers = st.session_state.search_results
    if papers:
        # Source breakdown
        src_counts: dict[str, int] = {}
        for p in papers:
            src_counts[p.source] = src_counts.get(p.source, 0) + 1

        summary_parts = [
            f"<span style='background:{SOURCE_COLORS.get(s,'#888')};color:white;"
            f"padding:2px 8px;border-radius:4px;font-size:0.8em'>{s} {n}</span>"
            for s, n in src_counts.items()
        ]
        st.markdown(
            f"**{len(papers)} unique result(s)** &nbsp; " + " &nbsp; ".join(summary_parts),
            unsafe_allow_html=True,
        )

        # Export all citations
        all_citations = "\n\n".join(
            f"[{i}] {p.format_citation(citation_style)}" for i, p in enumerate(papers, 1)
        )
        st.download_button(
            "📥 Export All Citations",
            data=all_citations,
            file_name=f"citations_{citation_style}.txt",
            mime="text/plain",
        )

        st.divider()

        for i, paper in enumerate(papers, 1):
            color = SOURCE_COLORS.get(paper.source, "#888")
            badge = (
                f"<span style='background:{color};color:white;"
                f"padding:1px 7px;border-radius:3px;font-size:0.75em'>"
                f"{paper.source}</span>"
            )
            year_str = f" ({paper.year})" if paper.year else ""
            title_md = f"{i}. {paper.title}{year_str} &nbsp; {badge}"

            with st.expander(f"{i}. {paper.title}{year_str}  [{paper.source}]"):
                # Header row
                st.markdown(title_md, unsafe_allow_html=True)

                meta_cols = st.columns(4)
                meta_cols[0].markdown(
                    f"**Authors**  \n{', '.join(paper.authors[:3]) + (' et al.' if len(paper.authors) > 3 else '') or '—'}"
                )
                meta_cols[1].markdown(f"**Venue**  \n{paper.venue or '—'}")
                meta_cols[2].markdown(f"**Citations**  \n{paper.citation_count if paper.citation_count is not None else '—'}")
                meta_cols[3].markdown(f"**PDF**  \n{'[Download](' + paper.pdf_url + ')' if paper.pdf_url else '—'}")

                # TLDR (Semantic Scholar)
                if paper.tldr:
                    st.info(f"**TL;DR** {paper.tldr}")

                # Abstract
                if paper.abstract:
                    with st.container():
                        st.markdown("**Abstract**")
                        st.markdown(paper.abstract)

                if paper.url:
                    st.markdown(f"**Link:** [{paper.url}]({paper.url})")

                # Citation (reacts to sidebar style change instantly)
                citation_text = paper.format_citation(citation_style)
                st.markdown(f"**{citation_style} Citation**")
                st.code(citation_text, language=None)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DOI Lookup (Crossref)
# ════════════════════════════════════════════════════════════════════════════════
with tab_doi:
    st.header("DOI Lookup & Citation Validator")
    st.caption(
        "Resolve any DOI via Crossref to get 100 % accurate metadata — "
        "prevents citation hallucinations in your reference list."
    )

    doi_input = st.text_input("DOI", placeholder="e.g. 10.48550/arXiv.1706.03762")

    if st.button("🔗 Resolve DOI", type="primary"):
        if not doi_input.strip():
            st.warning("Please enter a DOI.")
        else:
            with st.spinner("Querying Crossref…"):
                try:
                    paper = lookup_doi(doi_input.strip())
                except Exception as e:
                    st.error(f"Lookup failed: {e}")
                    paper = None

            if paper is None:
                st.error("DOI not found in Crossref.")
            else:
                st.success("DOI resolved successfully.")
                st.markdown(f"**Title:** {paper.title}")
                st.markdown(f"**Authors:** {', '.join(paper.authors)}")
                st.markdown(f"**Year:** {paper.year}")
                st.markdown(f"**Venue:** {paper.venue or '—'}")
                st.markdown(f"**Citations:** {paper.citation_count if paper.citation_count is not None else '—'}")
                if paper.url:
                    st.markdown(f"**URL:** [{paper.url}]({paper.url})")
                st.markdown(f"**{citation_style} Citation**")
                st.code(paper.format_citation(citation_style), language=None)
