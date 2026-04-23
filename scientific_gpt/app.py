import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_handler import extract_text_from_pdf
from utils.rag import (
    build_vectorstore,
    load_vectorstore,
    retrieve_chunks,
    EmbeddingsUnavailableError,
)
from utils.search_engine import search as multi_search, available_sources, is_source_available, SOURCE_COLORS
from utils.sources.base import CITATION_STYLES
from utils.sources.crossref import lookup_doi
from utils.answer_engine import answer_from_papers, answer_from_mixed
from utils.claim_support import find_evidence, Evidence
from utils.llm_backend import (
    LLMConfig, LLMChain, PROVIDERS, GOOGLE, OPENAI, ANTHROPIC, GROQ,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Scientific GPT", page_icon="🔬", layout="wide")

# Modes for the unified Ask tab
MODE_PDF = "📄 Only PDFs (RAG)"
MODE_ONLINE = "🌐 Only online sources"
MODE_MIXED = "🔗 Combined (PDFs + online)"

# Scope options for claim verification
SCOPE_PDF = "📄 PDFs only"
SCOPE_ONLINE = "🌐 Online sources only"
SCOPE_MIXED = "🔗 Combined"

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [
    ("search_results", []),
    ("search_errors", {}),
    ("answer_text", None),
    ("answer_mode", None),
    ("answer_papers", []),
    ("answer_pdf_refs", []),
    ("vectorstore", None),
    ("indexed_files", []),
    ("verify_evidence", []),
    ("verify_claim_shown", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _secret(name: str) -> str:
    try:
        v = st.secrets.get(name, "")
        if v:
            return v
    except Exception:
        pass
    return os.getenv(name, "")


_FALLBACK_ORDER: list[tuple[str, str]] = [
    (GOOGLE, "GOOGLE_API_KEY"),
    (GROQ, "GROQ_API_KEY"),
    (OPENAI, "OPENAI_API_KEY"),
    (ANTHROPIC, "ANTHROPIC_API_KEY"),
]


def _bundled_chain() -> LLMChain | None:
    configs = [
        LLMConfig(provider=prov, api_key=_secret(env))
        for prov, env in _FALLBACK_ORDER
        if _secret(env)
    ]
    if not configs:
        return None
    return LLMChain(configs=configs)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    use_own_key = st.checkbox(
        "Use my own API key",
        value=False,
        help="By default the app uses bundled keys with automatic fallback. "
             "Enable this to use your own key with any supported provider.",
    )

    chain: LLMChain | None = None
    config_error: str | None = None

    if use_own_key:
        provider = st.selectbox(
            "Provider",
            options=list(PROVIDERS.keys()),
            index=0,
            help="Pick which LLM backend your key belongs to.",
        )
        meta = PROVIDERS[provider]
        user_key = st.text_input(
            meta["key_label"],
            value="",
            type="password",
            help=f"Get a key at {meta['key_hint']}",
        )
        if user_key.strip():
            chain = LLMChain(configs=[LLMConfig(provider=provider, api_key=user_key.strip())])
        else:
            config_error = "Enter your API key to continue."
        if not meta["has_embeddings"]:
            st.info(
                f"ℹ️ {provider} has no embedding model — PDF indexing and "
                f"semantic re-ranking will be skipped for this provider."
            )
    else:
        chain = _bundled_chain()
        if chain is None:
            config_error = (
                "No bundled API key is configured. Enable *Use my own API key* "
                "in the sidebar to continue."
            )
        elif len(chain.configs) > 1:
            st.caption(
                "🔁 Auto-fallback active: "
                + " → ".join(c.provider for c in chain.configs)
            )
        else:
            st.caption(f"Using bundled **{chain.primary.provider}** key.")

    st.divider()

    citation_style = st.selectbox(
        "Citation Style",
        options=CITATION_STYLES,
        index=0,
        help=(
            "APA — Social Sciences · MLA — Humanities · IEEE — Engineering/CS · "
            "Chicago — History · Harvard — Business · Vancouver — Medicine · "
            "ACM — Computer Science"
        ),
    )

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
    st.caption(
        "Scientific GPT · Gemini · ChatGPT · Claude · Groq · FAISS\n"
        "Semantic Scholar · OpenAlex · arXiv · PubMed · Crossref · CORE"
    )

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔬 Scientific GPT")
st.markdown("*AI-powered academic research assistant — 6 sources, one search*")

if chain is None:
    st.warning(config_error or "API key not configured.")
    st.stop()


def _notify_fallback():
    if chain.fell_back and chain.last_used is not None:
        st.info(
            f"ℹ️ Primary provider hit quota — automatically switched to "
            f"**{chain.last_used.provider}**."
        )


def _render_online_reference(i: int, p, style: str):
    citation = p.format_citation(style)
    color = SOURCE_COLORS.get(p.source, "#888")
    badge_html = (
        f"<span style='background:{color};color:white;"
        f"padding:1px 6px;border-radius:3px;font-size:0.72em'>{p.source}</span>"
    )
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


def _render_pdf_reference(ref):
    badge = (
        "<span style='background:#6c757d;color:white;"
        "padding:1px 6px;border-radius:3px;font-size:0.72em'>PDF</span>"
    )
    st.markdown(
        f"**[{ref.index}]** {ref.filename} &nbsp; {badge}",
        unsafe_allow_html=True,
    )
    with st.expander(f"Excerpt from [{ref.index}]"):
        st.markdown(f"> {ref.preview}…")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pdfs, tab_ask, tab_verify, tab_doi = st.tabs(
    ["📄 PDFs", "🔬 Ask", "✅ Verify Claim", "🔗 DOI Lookup"]
)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PDF Management
# ════════════════════════════════════════════════════════════════════════════════
with tab_pdfs:
    st.header("PDF Index")
    st.caption(
        "Upload papers to build a local FAISS index. The index is reused across "
        "questions — upload once, ask many times."
    )

    if not chain.has_embeddings:
        st.error(
            "None of the active providers offers embeddings. "
            "Switch to Google Gemini or OpenAI in the sidebar to enable PDF indexing."
        )
    else:
        uploaded_files = st.file_uploader(
            "Upload PDF papers", type=["pdf"], accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("📥 Build / Update FAISS Index", type="primary"):
                with st.spinner("Extracting text and computing embeddings…"):
                    docs: list[tuple[str, str]] = []
                    for f in uploaded_files:
                        t = extract_text_from_pdf(f)
                        if t.strip():
                            docs.append((f.name, t))
                        else:
                            st.warning(f"No text extracted from **{f.name}** — skipped.")
                    if not docs:
                        st.error("No text could be extracted.")
                    else:
                        try:
                            vs = build_vectorstore(docs, chain)
                            st.session_state.vectorstore = vs
                            st.session_state.indexed_files = [n for n, _ in docs]
                            st.success(
                                f"Index built from: "
                                f"{', '.join(f'**{n}**' for n, _ in docs)}"
                            )
                        except Exception as e:
                            import traceback
                            st.error(f"**Embedding failed:** {type(e).__name__}: {e}")
                            with st.expander("Full traceback"):
                                st.code(traceback.format_exc())

    st.divider()
    st.markdown("**Current index**")
    if st.session_state.indexed_files:
        for n in st.session_state.indexed_files:
            st.markdown(f"• {n}")
    else:
        st.caption("No PDFs indexed yet.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Unified Ask
# ════════════════════════════════════════════════════════════════════════════════
with tab_ask:
    st.header("Ask a research question")

    mode = st.radio(
        "Sources to use",
        options=[MODE_PDF, MODE_ONLINE, MODE_MIXED],
        index=1,
        horizontal=True,
        help=(
            "• **Only PDFs** — answer from your uploaded papers only (RAG).\n"
            "• **Only online** — answer from multi-database academic search.\n"
            "• **Combined** — blend both; citations mix PDF excerpts and external papers."
        ),
    )

    research_question = st.text_area(
        "Research question",
        placeholder="e.g. What are the advantages of transformer architectures over RNNs?",
        height=100,
    )

    # Online-mode / Combined-mode: search configuration
    need_online = mode in (MODE_ONLINE, MODE_MIXED)
    selected_sources: list[str] = []
    query = ""
    limit = 5
    if need_online:
        col_q, col_n = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "Search keywords",
                placeholder="e.g. transformer attention mechanism",
                help="Keywords sent to the databases. Can equal your question or be more specific.",
            )
        with col_n:
            limit = st.number_input("Per source", min_value=1, max_value=15, value=5)

        st.markdown("**Sources**")
        all_srcs = available_sources()
        src_cols = st.columns(len(all_srcs))
        for col, name in zip(src_cols, all_srcs):
            avail = is_source_available(name)
            label = name if avail else f"{name} *(key needed)*"
            if col.checkbox(label, value=avail, disabled=not avail, key=f"src_{name}") and avail:
                selected_sources.append(name)

    # PDF-mode / Combined-mode: warn if no index
    need_pdf = mode in (MODE_PDF, MODE_MIXED)
    if need_pdf and not st.session_state.indexed_files:
        st.warning(
            "No PDF index available. Go to the **PDFs** tab and build an index first."
        )

    if st.button("🧠 Ask", type="primary"):
        if not research_question.strip():
            st.warning("Please enter a research question.")
        elif need_online and not query.strip():
            st.warning("Please enter search keywords.")
        elif need_online and not selected_sources:
            st.warning("Please select at least one online source.")
        elif need_pdf and not st.session_state.indexed_files:
            st.warning("Please build a PDF index in the **PDFs** tab first.")
        else:
            st.session_state.answer_text = None
            st.session_state.answer_mode = mode
            st.session_state.answer_papers = []
            st.session_state.answer_pdf_refs = []
            st.session_state.search_results = []
            st.session_state.search_errors = {}

            # Load / reuse vectorstore for PDF + Mixed modes
            vs = None
            if need_pdf:
                try:
                    vs = st.session_state.vectorstore or load_vectorstore(chain)
                    if vs is not None and st.session_state.vectorstore is None:
                        st.session_state.vectorstore = vs
                except EmbeddingsUnavailableError as e:
                    st.error(str(e))
                    st.stop()
                if vs is None:
                    st.error("FAISS index missing — rebuild it in the **PDFs** tab.")
                    st.stop()

            # Run the mode
            try:
                if mode == MODE_PDF:
                    from utils.rag import query_rag
                    with st.spinner(f"Querying {chain.primary.provider} over your PDFs…"):
                        answer = query_rag(
                            research_question, vs, chain, citation_style
                        )
                    st.session_state.answer_text = answer

                elif mode == MODE_ONLINE:
                    with st.spinner(
                        f"Searching {len(selected_sources)} source(s) in parallel…"
                    ):
                        papers, errors = multi_search(query, selected_sources, int(limit))
                    st.session_state.search_results = papers
                    st.session_state.search_errors = errors
                    if papers:
                        with st.spinner(
                            f"Re-ranking {len(papers)} papers & generating answer…"
                        ):
                            body, ranked = answer_from_papers(
                                research_question, papers, chain, citation_style
                            )
                            st.session_state.answer_text = body
                            st.session_state.answer_papers = ranked
                    else:
                        st.session_state.answer_text = (
                            "No online papers returned for these keywords."
                        )

                elif mode == MODE_MIXED:
                    with st.spinner("Retrieving relevant PDF excerpts…"):
                        chunks = retrieve_chunks(vs, research_question, k=4)
                    with st.spinner(
                        f"Searching {len(selected_sources)} source(s) in parallel…"
                    ):
                        papers, errors = multi_search(query, selected_sources, int(limit))
                    st.session_state.search_results = papers
                    st.session_state.search_errors = errors
                    with st.spinner("Blending PDF + online sources into answer…"):
                        body, pdf_refs, ranked = answer_from_mixed(
                            research_question, chunks, papers, chain, citation_style
                        )
                        st.session_state.answer_text = body
                        st.session_state.answer_pdf_refs = pdf_refs
                        st.session_state.answer_papers = ranked

            except Exception as e:
                import traceback
                st.session_state.answer_text = (
                    f"⚠️ **Answer generation failed:** {type(e).__name__}: {e}\n\n"
                    f"```\n{traceback.format_exc()[-800:]}\n```"
                )

    # ── Render errors + answer ────────────────────────────────────────────────
    for src, err in st.session_state.search_errors.items():
        st.warning(f"**{src}** failed: {err}")

    if st.session_state.answer_text:
        st.divider()
        _notify_fallback()
        st.markdown("## 🧠 Answer")
        answered_mode = st.session_state.answer_mode or mode
        st.caption(f"Mode: **{answered_mode}**")
        st.markdown(st.session_state.answer_text)

        # References section — differs by mode
        pdf_refs = st.session_state.answer_pdf_refs
        online_refs = st.session_state.answer_papers

        if pdf_refs or online_refs:
            st.markdown("---")
            st.markdown("## References")
            for ref in pdf_refs:
                _render_pdf_reference(ref)
            offset = len(pdf_refs)
            for i, p in enumerate(online_refs, start=offset + 1):
                _render_online_reference(i, p, citation_style)

            # Export
            export_lines = [st.session_state.answer_text, "", "## References", ""]
            for ref in pdf_refs:
                export_lines.append(f"[{ref.index}] PDF — {ref.filename}")
            for i, p in enumerate(online_refs, start=offset + 1):
                export_lines.append(f"[{i}] {p.format_citation(citation_style)}")
            st.download_button(
                "📥 Export Answer + References",
                data="\n".join(export_lines),
                file_name=f"answer_{citation_style}.txt",
                mime="text/plain",
            )

    # ── All found online papers (online / combined modes) ────────────────────
    papers = st.session_state.search_results
    if papers:
        st.divider()
        src_counts: dict[str, int] = {}
        for p in papers:
            src_counts[p.source] = src_counts.get(p.source, 0) + 1
        summary_parts = [
            f"<span style='background:{SOURCE_COLORS.get(s,'#888')};color:white;"
            f"padding:2px 8px;border-radius:4px;font-size:0.8em'>{s} {n}</span>"
            for s, n in src_counts.items()
        ]
        st.markdown(
            f"**{len(papers)} unique online papers retrieved** &nbsp; " + " &nbsp; ".join(summary_parts),
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

        with st.expander("Show all retrieved papers"):
            for i, paper in enumerate(papers, 1):
                color = SOURCE_COLORS.get(paper.source, "#888")
                badge = (
                    f"<span style='background:{color};color:white;"
                    f"padding:1px 7px;border-radius:3px;font-size:0.75em'>{paper.source}</span>"
                )
                year_str = f" ({paper.year})" if paper.year else ""
                st.markdown(
                    f"**{i}. {paper.title}**{year_str} &nbsp; {badge}",
                    unsafe_allow_html=True,
                )
                m = st.columns(4)
                m[0].markdown(
                    f"**Authors**  \n"
                    f"{', '.join(paper.authors[:3]) + (' et al.' if len(paper.authors) > 3 else '') or '—'}"
                )
                m[1].markdown(f"**Venue**  \n{paper.venue or '—'}")
                m[2].markdown(
                    f"**Citations**  \n"
                    f"{paper.citation_count if paper.citation_count is not None else '—'}"
                )
                m[3].markdown(
                    f"**PDF**  \n"
                    f"{'[Download](' + paper.pdf_url + ')' if paper.pdf_url else '—'}"
                )
                if paper.tldr:
                    st.info(f"**TL;DR** {paper.tldr}")
                if paper.abstract:
                    st.markdown("**Abstract**")
                    st.markdown(paper.abstract)
                if paper.url:
                    st.markdown(f"**Link:** [{paper.url}]({paper.url})")
                st.code(paper.format_citation(citation_style), language=None)
                st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Verify Claim (evidence search)
# ════════════════════════════════════════════════════════════════════════════════
with tab_verify:
    st.header("Find evidence for a claim")
    st.caption(
        "Paste a specific statement and we'll search your PDFs and/or online "
        "databases for sources that **support**, **contradict**, or are "
        "**unrelated** to it."
    )

    claim_text = st.text_area(
        "Claim",
        placeholder="e.g. Attention mechanisms outperform recurrence on long-range "
                    "dependencies in language modeling.",
        height=90,
        key="claim_text",
    )

    v_scope = st.radio(
        "Sources to check",
        options=[SCOPE_PDF, SCOPE_ONLINE, SCOPE_MIXED],
        index=1,
        horizontal=True,
        help=(
            "• **PDFs only** — search your uploaded papers.\n"
            "• **Online only** — search multi-database academic sources.\n"
            "• **Combined** — both."
        ),
    )

    v_need_online = v_scope in (SCOPE_ONLINE, SCOPE_MIXED)
    v_need_pdf = v_scope in (SCOPE_PDF, SCOPE_MIXED)

    v_selected_sources: list[str] = []
    v_query = ""
    v_limit = 5
    if v_need_online:
        col_q, col_n = st.columns([4, 1])
        with col_q:
            v_query = st.text_input(
                "Search keywords",
                placeholder="keywords for the online database search",
                help="Usually the core terms of your claim.",
                key="verify_query",
            )
        with col_n:
            v_limit = st.number_input(
                "Per source", min_value=1, max_value=15, value=5, key="verify_limit"
            )

        st.markdown("**Sources**")
        v_srcs = available_sources()
        v_cols = st.columns(len(v_srcs))
        for col, name in zip(v_cols, v_srcs):
            avail = is_source_available(name)
            label = name if avail else f"{name} *(key needed)*"
            if col.checkbox(
                label, value=avail, disabled=not avail, key=f"vsrc_{name}"
            ) and avail:
                v_selected_sources.append(name)

    if v_need_pdf and not st.session_state.indexed_files:
        st.warning(
            "No PDF index available. Go to the **PDFs** tab and build an index first."
        )

    if st.button("🔍 Find Evidence", type="primary", key="verify_btn"):
        if not claim_text.strip():
            st.warning("Please enter a claim.")
        elif v_need_online and not v_query.strip():
            st.warning("Please enter search keywords.")
        elif v_need_online and not v_selected_sources:
            st.warning("Please select at least one online source.")
        elif v_need_pdf and not st.session_state.indexed_files:
            st.warning("Please build a PDF index in the **PDFs** tab first.")
        else:
            st.session_state.verify_evidence = []
            st.session_state.verify_claim_shown = claim_text.strip()

            vs = None
            if v_need_pdf:
                try:
                    vs = st.session_state.vectorstore or load_vectorstore(chain)
                    if vs is not None and st.session_state.vectorstore is None:
                        st.session_state.vectorstore = vs
                except EmbeddingsUnavailableError as e:
                    st.error(str(e))
                    st.stop()
                if vs is None:
                    st.error("FAISS index missing — rebuild it in the **PDFs** tab.")
                    st.stop()

            pdf_chunks = []
            online_papers = []
            try:
                if v_need_pdf:
                    with st.spinner("Retrieving relevant PDF excerpts…"):
                        pdf_chunks = retrieve_chunks(vs, claim_text.strip(), k=5)

                if v_need_online:
                    with st.spinner(
                        f"Searching {len(v_selected_sources)} source(s) in parallel…"
                    ):
                        online_papers, v_errors = multi_search(
                            v_query.strip(), v_selected_sources, int(v_limit)
                        )
                    for src, err in v_errors.items():
                        st.warning(f"**{src}** failed: {err}")

                if not pdf_chunks and not online_papers:
                    st.info("No candidate sources were found to judge.")
                else:
                    with st.spinner("Judging each source against the claim…"):
                        evidence = find_evidence(
                            claim_text.strip(),
                            chain,
                            pdf_chunks=pdf_chunks,
                            online_papers=online_papers,
                        )
                    st.session_state.verify_evidence = evidence

            except Exception as e:
                import traceback
                st.error(f"**Evidence search failed:** {type(e).__name__}: {e}")
                with st.expander("Full traceback"):
                    st.code(traceback.format_exc())

    # ── Render evidence groups ────────────────────────────────────────────────
    evidence = st.session_state.verify_evidence
    if evidence:
        st.divider()
        _notify_fallback()
        st.markdown(f"**Claim:** _{st.session_state.verify_claim_shown}_")

        supports = [e for e in evidence if e.verdict == "supports"]
        contradicts = [e for e in evidence if e.verdict == "contradicts"]
        unrelated = [e for e in evidence if e.verdict == "unrelated"]

        summary = (
            f"✅ **{len(supports)}** support &nbsp; · &nbsp; "
            f"❌ **{len(contradicts)}** contradict &nbsp; · &nbsp; "
            f"➖ **{len(unrelated)}** unrelated"
        )
        st.markdown(summary)

        def _render_evidence(ev: Evidence):
            if ev.kind == "paper" and ev.paper is not None:
                citation = ev.paper.format_citation(citation_style)
                color = SOURCE_COLORS.get(ev.paper.source, "#888")
                badge = (
                    f"<span style='background:{color};color:white;"
                    f"padding:1px 6px;border-radius:3px;font-size:0.72em'>"
                    f"{ev.paper.source}</span>"
                )
                link = f"[🔗 Open]({ev.paper.url})" if ev.paper.url else ""
                st.markdown(
                    f"**[{ev.index}]** {citation} &nbsp; {badge} &nbsp; {link}",
                    unsafe_allow_html=True,
                )
            else:
                pdf_badge = (
                    "<span style='background:#6c757d;color:white;"
                    "padding:1px 6px;border-radius:3px;font-size:0.72em'>PDF</span>"
                )
                st.markdown(
                    f"**[{ev.index}]** {ev.filename or 'PDF'} &nbsp; {pdf_badge}",
                    unsafe_allow_html=True,
                )
            if ev.reason:
                st.markdown(f"> {ev.reason}")
            with st.expander(f"Excerpt from [{ev.index}]"):
                st.markdown(f"> {ev.excerpt}…")

        if supports:
            st.markdown("### ✅ Supporting sources")
            for ev in supports:
                _render_evidence(ev)
        if contradicts:
            st.markdown("### ❌ Contradicting sources")
            for ev in contradicts:
                _render_evidence(ev)
        if unrelated:
            with st.expander(f"➖ {len(unrelated)} unrelated sources"):
                for ev in unrelated:
                    _render_evidence(ev)

        # Export
        lines = [f"Claim: {st.session_state.verify_claim_shown}", ""]
        for label, group in (
            ("Supporting", supports),
            ("Contradicting", contradicts),
            ("Unrelated", unrelated),
        ):
            if not group:
                continue
            lines.append(f"## {label}")
            for ev in group:
                if ev.kind == "paper" and ev.paper is not None:
                    lines.append(
                        f"[{ev.index}] {ev.paper.format_citation(citation_style)}"
                    )
                else:
                    lines.append(f"[{ev.index}] PDF — {ev.filename or 'PDF'}")
                if ev.reason:
                    lines.append(f"    Reason: {ev.reason}")
            lines.append("")
        st.download_button(
            "📥 Export Evidence Report",
            data="\n".join(lines),
            file_name=f"claim_evidence_{citation_style}.txt",
            mime="text/plain",
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — DOI Lookup (Crossref)
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
                st.markdown(
                    f"**Citations:** "
                    f"{paper.citation_count if paper.citation_count is not None else '—'}"
                )
                if paper.url:
                    st.markdown(f"**URL:** [{paper.url}]({paper.url})")
                st.markdown(f"**{citation_style} Citation**")
                st.code(paper.format_citation(citation_style), language=None)
