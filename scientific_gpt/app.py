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
from utils.sources.base import CITATION_STYLES, LANGUAGES
from utils.sources.crossref import lookup_doi
from utils.answer_engine import stream_from_papers, stream_from_mixed
from utils.literature_review import stream_review
from utils.claim_support import find_evidence, Evidence
from utils.secrets_filter import scrub as _scrub
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
    ("ask_keywords", []),
    ("verify_keywords", []),
    ("question_history", []),     # list[str], most recent first, capped at 12
    ("rq_widget_id", 0),           # bumps to force textarea re-render on recall
    ("recall_question", ""),
    ("review_text", None),
    ("review_papers", []),
    ("review_keywords", []),
    ("review_topic", ""),
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
    with st.expander("🔎 Search Settings", expanded=False):
        st.markdown("**Enabled online sources**")
        _all_srcs = available_sources()
        enabled_sources: list[str] = []
        for name in _all_srcs:
            avail = is_source_available(name)
            label = name if avail else f"{name} *(key needed)*"
            if st.checkbox(
                label,
                value=avail,
                disabled=not avail,
                key=f"settings_src_{name}",
            ) and avail:
                enabled_sources.append(name)

        st.markdown("**Filters**")
        min_year_input = st.number_input(
            "Minimum publication year",
            min_value=0,
            max_value=2100,
            value=0,
            step=1,
            help="0 = no filter. Papers with unknown year are dropped when a "
                 "minimum is set.",
        )
        min_year: int | None = int(min_year_input) if min_year_input else None

        lang_labels = [label for label, _ in LANGUAGES]
        lang_choice = st.selectbox(
            "Language",
            options=lang_labels,
            index=0,
            help="Papers without a declared language are kept — only filters out "
                 "papers the source explicitly marks as a different language.",
        )
        language_code = dict(LANGUAGES)[lang_choice] or None

        open_access_only = st.checkbox(
            "Open access only",
            value=False,
            help="Drop papers that are explicitly marked as non-OA. "
                 "Papers with unknown access status are kept.",
        )
        min_cit_input = st.number_input(
            "Minimum citation count",
            min_value=0, max_value=100000, value=0, step=1,
            help="0 = no filter. Papers without a citation count are kept.",
        )
        min_citations: int | None = int(min_cit_input) if min_cit_input else None
        exclude_retracted = st.checkbox(
            "Exclude retracted papers", value=True,
            help="Drops papers flagged as retracted by Crossref."
        )
        sort_by = st.selectbox(
            "Sort results",
            options=["relevance", "year", "citations"],
            index=0,
            help="Order of the retrieved papers list.",
        )
    st.session_state["enabled_sources"] = enabled_sources
    st.session_state["min_year"] = min_year
    st.session_state["language"] = language_code
    st.session_state["open_access_only"] = open_access_only
    st.session_state["min_citations"] = min_citations
    st.session_state["exclude_retracted"] = exclude_retracted
    st.session_state["sort_by"] = sort_by

    cache_size = len(st.session_state.get("search_cache") or {})
    if cache_size:
        if st.button(f"🗑️ Clear search cache ({cache_size})", key="clear_search_cache"):
            st.session_state["search_cache"] = {}
            st.rerun()

    st.divider()
    if st.session_state.question_history:
        with st.expander("🕘 Recent questions", expanded=False):
            for i, q in enumerate(st.session_state.question_history):
                preview = q if len(q) <= 60 else q[:57] + "…"
                if st.button(preview, key=f"hist_{i}", use_container_width=True):
                    st.session_state.recall_question = q
                    st.session_state.rq_widget_id += 1
                    st.rerun()
            if st.button("Clear history", key="clear_hist"):
                st.session_state.question_history = []
                st.rerun()

    st.divider()
    st.markdown("**Active PDF Index**")
    if st.session_state.indexed_files:
        for name in st.session_state.indexed_files:
            st.caption(f"• {name}")
        if st.button("🗑️ Clear Index"):
            st.session_state.vectorstore = None
            st.session_state.indexed_files = []
            st.session_state.pdf_hash = None
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


def _badges(p) -> str:
    """Source + open-access + retraction badges as one HTML string."""
    color = SOURCE_COLORS.get(p.source, "#888")
    parts = [
        f"<span style='background:{color};color:white;"
        f"padding:1px 6px;border-radius:3px;font-size:0.72em'>{p.source}</span>"
    ]
    if p.is_open_access:
        parts.append(
            "<span style='background:#0f9d58;color:white;"
            "padding:1px 6px;border-radius:3px;font-size:0.72em'>OA</span>"
        )
    if p.is_retracted:
        parts.append(
            "<span style='background:#c62828;color:white;"
            "padding:1px 6px;border-radius:3px;font-size:0.72em'>⚠️ RETRACTED</span>"
        )
    return " &nbsp; ".join(parts)


def _render_online_reference(i: int, p, style: str):
    citation = p.format_citation(style)
    links = []
    if p.url:
        links.append(f"[🔗 Open]({p.url})")
    if p.pdf_url and p.pdf_url != p.url:
        links.append(f"[📄 PDF]({p.pdf_url})")
    link_str = " &nbsp; ".join(links)
    st.markdown(
        f"**[{i}]** {citation} &nbsp; {_badges(p)} &nbsp; {link_str}",
        unsafe_allow_html=True,
    )


def _cached_search(query: str, sources: list[str], limit: int, **filters):
    """Per-session cache for multi_search.

    Same (query, sources, limit, filter) tuple returns the previous result
    instead of re-hitting all APIs — critical when the user iterates on the
    research question without changing the search terms.
    """
    cache: dict = st.session_state.setdefault("search_cache", {})
    key = (
        query.strip().lower(),
        tuple(sources),
        int(limit),
        tuple(sorted((k, v) for k, v in filters.items())),
    )
    if key not in cache:
        cache[key] = multi_search(query, sources, limit, **filters)
    return cache[key]


def _filter_kwargs() -> dict:
    """Pack all sidebar-configured search filters for multi_search."""
    return dict(
        min_year=st.session_state.get("min_year"),
        language=st.session_state.get("language"),
        open_access_only=st.session_state.get("open_access_only", False),
        min_citations=st.session_state.get("min_citations"),
        exclude_retracted=st.session_state.get("exclude_retracted", True),
        sort_by=st.session_state.get("sort_by", "relevance"),
    )


def _keyword_input(
    label: str,
    state_key: str,
    *,
    placeholder: str,
    help_text: str | None = None,
) -> str:
    """Chip-style keyword input. Enter adds a keyword, clicking a chip removes it.

    Returns the space-joined keyword string sent to the search APIs.
    """
    input_key = f"{state_key}__input"

    def _on_submit():
        v = st.session_state.get(input_key, "").strip()
        kws = st.session_state.setdefault(state_key, [])
        if v and v not in kws:
            kws.append(v)
        st.session_state[input_key] = ""

    st.text_input(
        label,
        key=input_key,
        on_change=_on_submit,
        placeholder=placeholder,
        help=help_text,
    )

    kws: list[str] = st.session_state.setdefault(state_key, [])
    if kws:
        cap = st.columns([8, 2])
        with cap[0]:
            st.caption(
                f"**{len(kws)} keyword{'s' if len(kws) != 1 else ''}** — click to remove"
            )
        with cap[1]:
            if st.button("Clear all", key=f"{state_key}__clear"):
                st.session_state[state_key] = []
                st.rerun()
        cols = st.columns(min(6, len(kws)))
        for i, kw in enumerate(kws):
            col = cols[i % len(cols)]
            if col.button(f"✕  {kw}", key=f"{state_key}__rm_{i}",
                          use_container_width=True):
                kws.pop(i)
                st.rerun()
    return " ".join(kws).strip()


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
tab_pdfs, tab_ask, tab_verify, tab_review, tab_doi = st.tabs(
    ["📄 PDFs", "🔬 Ask", "✅ Verify Claim", "📚 Review", "🔗 DOI Lookup"]
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
            import hashlib

            def _hash_uploads(files) -> str:
                h = hashlib.sha256()
                for f in sorted(files, key=lambda x: x.name):
                    h.update(f.name.encode("utf-8"))
                    f.seek(0)
                    h.update(f.read())
                    f.seek(0)
                return h.hexdigest()

            current_hash = _hash_uploads(uploaded_files)
            if st.session_state.get("pdf_hash") == current_hash and st.session_state.vectorstore:
                st.success(
                    "✅ These exact files are already indexed — no need to rebuild."
                )

            if st.button("📥 Build / Update FAISS Index", type="primary"):
                if st.session_state.get("pdf_hash") == current_hash and st.session_state.vectorstore:
                    st.info("Skipped — same files as the existing index.")
                else:
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
                                st.session_state.pdf_hash = current_hash
                                st.success(
                                    f"Index built from: "
                                    f"{', '.join(f'**{n}**' for n, _ in docs)}"
                                )
                            except Exception as e:
                                import traceback
                                st.error(
                                    f"**Embedding failed:** {type(e).__name__}: {_scrub(str(e))}"
                                )
                                with st.expander("Full traceback"):
                                    st.code(_scrub(traceback.format_exc()))

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
        value=st.session_state.recall_question,
        placeholder="e.g. What are the advantages of transformer architectures over RNNs?",
        height=100,
        key=f"rq_input_{st.session_state.rq_widget_id}",
    )
    # Clear the recall slot once the widget has consumed it.
    if st.session_state.recall_question:
        st.session_state.recall_question = ""

    # Online-mode / Combined-mode: search configuration
    need_online = mode in (MODE_ONLINE, MODE_MIXED)
    selected_sources: list[str] = list(st.session_state.get("enabled_sources") or [])
    query = ""
    limit = 5
    if need_online:
        col_q, col_n = st.columns([4, 1])
        with col_q:
            query = _keyword_input(
                "Search keywords",
                state_key="ask_keywords",
                placeholder="Type a keyword and press Enter",
                help_text="Keywords are combined (space-separated) before being sent "
                          "to the databases. Press Enter to add each keyword.",
            )
        with col_n:
            limit = st.number_input("Per source", min_value=1, max_value=15, value=5)

        if selected_sources:
            st.caption(
                "Sources: " + " · ".join(f"**{s}**" for s in selected_sources)
                + "  _(configure in sidebar → 🔎 Search Settings)_"
            )
        else:
            st.info(
                "No online sources are enabled. Open **🔎 Search Settings** in "
                "the sidebar to enable at least one."
            )

    # PDF-mode / Combined-mode: warn if no index
    need_pdf = mode in (MODE_PDF, MODE_MIXED)
    if need_pdf and not st.session_state.indexed_files:
        st.warning(
            "No PDF index available. Go to the **PDFs** tab and build an index first."
        )

    just_streamed = False

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

            # Push to history (dedup, cap at 12, most recent first)
            hist: list = st.session_state.question_history
            q = research_question.strip()
            if q in hist:
                hist.remove(q)
            hist.insert(0, q)
            del hist[12:]

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

            # Run the mode — streams tokens directly into the page
            try:
                if mode == MODE_PDF:
                    from utils.rag import stream_rag
                    st.divider()
                    st.markdown("## 🧠 Answer")
                    st.caption(f"Mode: **{mode}**")
                    with st.spinner(f"Querying {chain.primary.provider} over your PDFs…"):
                        stream_iter = stream_rag(
                            research_question, vs, chain, citation_style
                        )
                    answer = st.write_stream(stream_iter)
                    st.session_state.answer_text = answer
                    just_streamed = True

                elif mode == MODE_ONLINE:
                    with st.spinner(
                        f"Searching {len(selected_sources)} source(s) in parallel…"
                    ):
                        papers, errors = _cached_search(
                            query, selected_sources, int(limit), **_filter_kwargs()
                        )
                    st.session_state.search_results = papers
                    st.session_state.search_errors = errors
                    for src, err in errors.items():
                        st.warning(f"**{src}** failed: {err}")
                    st.divider()
                    st.markdown("## 🧠 Answer")
                    st.caption(f"Mode: **{mode}**")
                    with st.spinner(
                        f"Re-ranking {len(papers)} papers & starting answer…"
                    ):
                        stream_iter, ranked = stream_from_papers(
                            research_question, papers, chain
                        )
                    answer = st.write_stream(stream_iter)
                    st.session_state.answer_text = answer
                    st.session_state.answer_papers = ranked
                    just_streamed = True

                elif mode == MODE_MIXED:
                    with st.spinner("Retrieving relevant PDF excerpts…"):
                        chunks = retrieve_chunks(vs, research_question, k=4)
                    with st.spinner(
                        f"Searching {len(selected_sources)} source(s) in parallel…"
                    ):
                        papers, errors = _cached_search(
                            query, selected_sources, int(limit), **_filter_kwargs()
                        )
                    st.session_state.search_results = papers
                    st.session_state.search_errors = errors
                    for src, err in errors.items():
                        st.warning(f"**{src}** failed: {err}")
                    st.divider()
                    st.markdown("## 🧠 Answer")
                    st.caption(f"Mode: **{mode}**")
                    with st.spinner("Blending PDF + online sources into answer…"):
                        stream_iter, pdf_refs, ranked = stream_from_mixed(
                            research_question, chunks, papers, chain
                        )
                    answer = st.write_stream(stream_iter)
                    st.session_state.answer_text = answer
                    st.session_state.answer_pdf_refs = pdf_refs
                    st.session_state.answer_papers = ranked
                    just_streamed = True

            except Exception as e:
                import traceback
                st.session_state.answer_text = (
                    f"⚠️ **Answer generation failed:** {type(e).__name__}: {_scrub(str(e))}\n\n"
                    f"```\n{_scrub(traceback.format_exc()[-800:])}\n```"
                )
                just_streamed = False

    # ── Render errors + answer ────────────────────────────────────────────────
    # Skip on the streaming run — they were already rendered above the answer.
    if not just_streamed:
        for src, err in st.session_state.search_errors.items():
            st.warning(f"**{src}** failed: {err}")

    if st.session_state.answer_text:
        if not just_streamed:
            st.divider()
            _notify_fallback()
            st.markdown("## 🧠 Answer")
            answered_mode = st.session_state.answer_mode or mode
            st.caption(f"Mode: **{answered_mode}**")
            st.markdown(st.session_state.answer_text)
        else:
            # Header + body already streamed; just surface a fallback notice if any.
            _notify_fallback()

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
            exp_cols = st.columns(2)
            with exp_cols[0]:
                st.download_button(
                    "📥 Export Answer + References (TXT)",
                    data="\n".join(export_lines),
                    file_name=f"answer_{citation_style}.txt",
                    mime="text/plain",
                )
            with exp_cols[1]:
                if online_refs:
                    bib = "\n\n".join(p.to_bibtex() for p in online_refs)
                    st.download_button(
                        "📚 Export references (BibTeX)",
                        data=bib,
                        file_name="references.bib",
                        mime="application/x-bibtex",
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
        all_bibtex = "\n\n".join(p.to_bibtex() for p in papers)
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "📥 Export All Citations (TXT)",
                data=all_citations,
                file_name=f"citations_{citation_style}.txt",
                mime="text/plain",
            )
        with ec2:
            st.download_button(
                "📚 Export All Citations (BibTeX)",
                data=all_bibtex,
                file_name="citations.bib",
                mime="application/x-bibtex",
            )

        with st.expander("Show all retrieved papers"):
            for i, paper in enumerate(papers, 1):
                year_str = f" ({paper.year})" if paper.year else ""
                st.markdown(
                    f"**{i}. {paper.title}**{year_str} &nbsp; {_badges(paper)}",
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

    v_selected_sources: list[str] = list(st.session_state.get("enabled_sources") or [])
    v_query = ""
    v_limit = 5
    if v_need_online:
        col_q, col_n = st.columns([4, 1])
        with col_q:
            v_query = _keyword_input(
                "Search keywords",
                state_key="verify_keywords",
                placeholder="Type a keyword and press Enter",
                help_text="Usually the core terms of your claim. Press Enter to add each keyword.",
            )
        with col_n:
            v_limit = st.number_input(
                "Per source", min_value=1, max_value=15, value=5, key="verify_limit"
            )

        if v_selected_sources:
            st.caption(
                "Sources: " + " · ".join(f"**{s}**" for s in v_selected_sources)
                + "  _(configure in sidebar → 🔎 Search Settings)_"
            )
        else:
            st.info(
                "No online sources are enabled. Open **🔎 Search Settings** in "
                "the sidebar to enable at least one."
            )

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
                        online_papers, v_errors = _cached_search(
                            v_query.strip(), v_selected_sources, int(v_limit),
                            **_filter_kwargs(),
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
                st.error(f"**Evidence search failed:** {type(e).__name__}: {_scrub(str(e))}")
                with st.expander("Full traceback"):
                    st.code(_scrub(traceback.format_exc()))

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
                link = f"[🔗 Open]({ev.paper.url})" if ev.paper.url else ""
                st.markdown(
                    f"**[{ev.index}]** {citation} &nbsp; {_badges(ev.paper)} &nbsp; {link}",
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
# TAB 4 — Literature Review
# ════════════════════════════════════════════════════════════════════════════════
with tab_review:
    st.header("Generate a structured literature review")
    st.caption(
        "Provide a topic and search keywords. The app pulls papers from the "
        "enabled online sources, re-ranks them, and drafts an academic review "
        "with introduction, methods, findings, gaps, and future directions."
    )

    rv_topic = st.text_area(
        "Review topic",
        value=st.session_state.review_topic,
        placeholder="e.g. Self-supervised pre-training for medical image segmentation",
        height=80,
        key="review_topic_input",
    )

    rv_selected_sources: list[str] = list(st.session_state.get("enabled_sources") or [])

    col_q, col_n, col_m = st.columns([4, 1, 1])
    with col_q:
        rv_query = _keyword_input(
            "Search keywords",
            state_key="review_keywords",
            placeholder="Type a keyword and press Enter",
            help_text="Keywords used to retrieve candidate papers from the "
                      "enabled databases.",
        )
    with col_n:
        rv_limit = st.number_input(
            "Per source", min_value=1, max_value=20, value=8, key="review_limit"
        )
    with col_m:
        rv_max = st.number_input(
            "Max in review", min_value=3, max_value=30, value=12, key="review_max",
            help="Top-N papers (after re-ranking) to feed into the review.",
        )

    if rv_selected_sources:
        st.caption(
            "Sources: " + " · ".join(f"**{s}**" for s in rv_selected_sources)
            + "  _(configure in sidebar → 🔎 Search Settings)_"
        )
    else:
        st.info(
            "No online sources are enabled. Open **🔎 Search Settings** in the "
            "sidebar to enable at least one."
        )

    review_just_streamed = False

    if st.button("📝 Generate Review", type="primary", key="review_btn"):
        if not rv_topic.strip():
            st.warning("Please enter a review topic.")
        elif not rv_query.strip():
            st.warning("Please enter at least one keyword.")
        elif not rv_selected_sources:
            st.warning("Please enable at least one online source.")
        else:
            st.session_state.review_text = None
            st.session_state.review_papers = []
            st.session_state.review_topic = rv_topic.strip()

            try:
                with st.spinner(
                    f"Searching {len(rv_selected_sources)} source(s) in parallel…"
                ):
                    papers, errors = _cached_search(
                        rv_query, rv_selected_sources, int(rv_limit),
                        **_filter_kwargs(),
                    )
                for src, err in errors.items():
                    st.warning(f"**{src}** failed: {err}")

                if not papers:
                    st.info("No papers returned. Refine the keywords or enable more sources.")
                else:
                    st.divider()
                    st.markdown(f"## 📚 Literature Review — *{rv_topic.strip()}*")
                    with st.spinner(
                        f"Re-ranking {len(papers)} papers & drafting review…"
                    ):
                        stream_iter, ranked = stream_review(
                            rv_topic.strip(), papers, chain, max_papers=int(rv_max)
                        )
                    body = st.write_stream(stream_iter)
                    st.session_state.review_text = body
                    st.session_state.review_papers = ranked
                    review_just_streamed = True

            except Exception as e:
                import traceback
                st.session_state.review_text = (
                    f"⚠️ **Review generation failed:** {type(e).__name__}: "
                    f"{_scrub(str(e))}\n\n```\n{_scrub(traceback.format_exc()[-800:])}\n```"
                )
                review_just_streamed = False

    # ── Render persisted review ──────────────────────────────────────────────
    if st.session_state.review_text:
        if not review_just_streamed:
            st.divider()
            _notify_fallback()
            st.markdown(
                f"## 📚 Literature Review — *{st.session_state.review_topic}*"
            )
            st.markdown(st.session_state.review_text)
        else:
            _notify_fallback()

        rv_refs = st.session_state.review_papers
        if rv_refs:
            st.markdown("---")
            st.markdown("## References")
            for i, p in enumerate(rv_refs, 1):
                _render_online_reference(i, p, citation_style)

            export_lines = [
                f"# Literature Review — {st.session_state.review_topic}",
                "",
                st.session_state.review_text,
                "",
                "## References",
                "",
            ]
            for i, p in enumerate(rv_refs, 1):
                export_lines.append(f"[{i}] {p.format_citation(citation_style)}")
            ec1, ec2 = st.columns(2)
            with ec1:
                st.download_button(
                    "📥 Export Review (TXT)",
                    data="\n".join(export_lines),
                    file_name=f"literature_review_{citation_style}.txt",
                    mime="text/plain",
                    key="review_dl_txt",
                )
            with ec2:
                bib = "\n\n".join(p.to_bibtex() for p in rv_refs)
                st.download_button(
                    "📚 Export references (BibTeX)",
                    data=bib,
                    file_name="literature_review.bib",
                    mime="application/x-bibtex",
                    key="review_dl_bib",
                )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — DOI Lookup (Crossref)
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
                if paper.is_retracted:
                    st.error(
                        "⚠️ **This paper has been retracted.** Do not cite it as "
                        "supporting evidence — Crossref lists a retraction notice."
                    )
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
