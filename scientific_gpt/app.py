import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_handler import extract_text_from_pdf
from utils.rag import build_vectorstore, load_vectorstore, query_rag
from utils.search_engine import search as multi_search, available_sources, is_source_available, SOURCE_COLORS
from utils.sources.crossref import lookup_doi
from utils.answer_engine import answer_from_papers

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scientific GPT", page_icon="🔬", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("search_results", []),
    ("search_errors", {}),
    ("answer_text", None),
    ("answer_papers", []),
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
                    try:
                        vs = build_vectorstore(full_text, api_key)
                        st.session_state.vectorstore = vs
                        st.session_state.indexed_files = names
                        st.success(f"Index built from: {', '.join(f'**{n}**' for n in names)}")
                    except Exception as e:
                        # Surface the real error — Streamlit Cloud redacts uncaught errors
                        import traceback
                        st.error(f"**Embedding failed:** {type(e).__name__}: {e}")
                        with st.expander("Full traceback"):
                            st.code(traceback.format_exc())

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
# TAB 2 — Multi-Source Search + AI Answer
# ════════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Multi-Source Academic Search")

    # ── Research question (drives the AI answer) ──────────────────────────────
    research_question = st.text_area(
        "Research question",
        placeholder="e.g. What are the advantages of transformer architectures over RNNs?",
        height=90,
        help="Gemini will write a cited academic answer based on the found papers.",
    )

    # ── Keyword query + per-source limit ─────────────────────────────────────
    col_q, col_n = st.columns([4, 1])
    with col_q:
        query = st.text_input(
            "Search keywords",
            placeholder="e.g. transformer attention mechanism",
            help="Keywords sent to the databases. Can be the same as your question or more specific.",
        )
    with col_n:
        limit = st.number_input("Per source", min_value=1, max_value=15, value=5)

    # ── Source selection ──────────────────────────────────────────────────────
    st.markdown("**Sources**")
    all_srcs = available_sources()
    src_cols = st.columns(len(all_srcs))
    selected_sources: list[str] = []
    for col, name in zip(src_cols, all_srcs):
        avail = is_source_available(name)
        label = name if avail else f"{name} *(key needed)*"
        if col.checkbox(label, value=avail, disabled=not avail, key=f"src_{name}") and avail:
            selected_sources.append(name)

    if st.button("🔍 Search & Answer", type="primary"):
        if not query.strip():
            st.warning("Please enter search keywords.")
        elif not selected_sources:
            st.warning("Please select at least one source.")
        else:
            with st.spinner(f"Searching {len(selected_sources)} source(s) in parallel…"):
                papers, errors = multi_search(query, selected_sources, int(limit))
            st.session_state.search_results = papers
            st.session_state.search_errors = errors
            st.session_state.answer_text = None
            st.session_state.answer_papers = []

            if papers and research_question.strip():
                with st.spinner(f"Re-ranking {len(papers)} papers by relevance & generating answer…"):
                    try:
                        body, ranked = answer_from_papers(
                            research_question, papers, api_key, citation_style
                        )
                        st.session_state.answer_text = body
                        st.session_state.answer_papers = ranked
                    except Exception as e:
                        import traceback
                        st.session_state.answer_text = (
                            f"⚠️ **Answer generation failed:** {type(e).__name__}: {e}\n\n"
                            f"```\n{traceback.format_exc()[-800:]}\n```"
                        )
                        st.session_state.answer_papers = []

    # ── Source errors ─────────────────────────────────────────────────────────
    for src, err in st.session_state.search_errors.items():
        st.warning(f"**{src}** failed: {err}")

    papers = st.session_state.search_results

    # ── AI Answer + References ────────────────────────────────────────────────
    if st.session_state.get("answer_text"):
        answer_papers = st.session_state.answer_papers
        st.divider()
        st.markdown("## 🧠 Answer")
        if answer_papers and papers:
            st.caption(
                f"Cited answer built from the **{len(answer_papers)}** most relevant "
                f"of {len(papers)} retrieved papers (semantic re-ranking via Gemini embeddings)."
            )
        st.markdown(st.session_state.answer_text)

        # References section — app-rendered so links are clickable
        st.markdown("---")
        st.markdown("## References")
        for i, p in enumerate(answer_papers, 1):
            citation = p.format_citation(citation_style)
            color = SOURCE_COLORS.get(p.source, "#888")
            badge_html = (
                f"<span style='background:{color};color:white;"
                f"padding:1px 6px;border-radius:3px;font-size:0.72em'>{p.source}</span>"
            )
            # Link buttons
            links = []
            if p.url:
                links.append(f"[🔗 Open]({p.url})")
            if p.pdf_url and p.pdf_url != p.url:
                links.append(f"[📄 PDF]({p.pdf_url})")
            link_str = " &nbsp; ".join(links)

            st.markdown(
                f"**[{i}]** {citation} &nbsp; {badge_html} &nbsp; {link_str}",
                unsafe_allow_html=True,
            )

        # Export
        export_text = (st.session_state.answer_text + "\n\n## References\n\n" +
                       "\n".join(f"[{i}] {p.format_citation(citation_style)}"
                                 for i, p in enumerate(answer_papers, 1)))
        st.download_button(
            "📥 Export Answer + References",
            data=export_text,
            file_name=f"answer_{citation_style}.txt",
            mime="text/plain",
        )

    # ── All found papers (collapsible) ────────────────────────────────────────
    if papers:
        st.divider()

        # Source badge summary
        src_counts: dict[str, int] = {}
        for p in papers:
            src_counts[p.source] = src_counts.get(p.source, 0) + 1
        summary_parts = [
            f"<span style='background:{SOURCE_COLORS.get(s,'#888')};color:white;"
            f"padding:2px 8px;border-radius:4px;font-size:0.8em'>{s} {n}</span>"
            for s, n in src_counts.items()
        ]
        st.markdown(
            f"**{len(papers)} unique papers found** &nbsp; " + " &nbsp; ".join(summary_parts),
            unsafe_allow_html=True,
        )

        all_citations = "\n\n".join(
            f"[{i}] {p.format_citation(citation_style)}" for i, p in enumerate(papers, 1)
        )
        st.download_button(
            "📥 Export All Citations",
            data=all_citations,
            file_name=f"citations_{citation_style}.txt",
            mime="text/plain",
        )

        st.markdown("**All Results**")
        for i, paper in enumerate(papers, 1):
            color = SOURCE_COLORS.get(paper.source, "#888")
            badge = (
                f"<span style='background:{color};color:white;"
                f"padding:1px 7px;border-radius:3px;font-size:0.75em'>{paper.source}</span>"
            )
            year_str = f" ({paper.year})" if paper.year else ""
            with st.expander(f"{i}. {paper.title}{year_str}  [{paper.source}]"):
                st.markdown(f"**{paper.title}**{year_str} &nbsp; {badge}", unsafe_allow_html=True)

                m = st.columns(4)
                m[0].markdown(f"**Authors**  \n{', '.join(paper.authors[:3]) + (' et al.' if len(paper.authors) > 3 else '') or '—'}")
                m[1].markdown(f"**Venue**  \n{paper.venue or '—'}")
                m[2].markdown(f"**Citations**  \n{paper.citation_count if paper.citation_count is not None else '—'}")
                m[3].markdown(f"**PDF**  \n{'[Download](' + paper.pdf_url + ')' if paper.pdf_url else '—'}")

                if paper.tldr:
                    st.info(f"**TL;DR** {paper.tldr}")
                if paper.abstract:
                    st.markdown("**Abstract**")
                    st.markdown(paper.abstract)
                if paper.url:
                    st.markdown(f"**Link:** [{paper.url}]({paper.url})")
                st.code(paper.format_citation(citation_style), language=None)

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
