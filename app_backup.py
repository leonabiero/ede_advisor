import streamlit as st
import anthropic
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EDE Social Advisor",
    page_icon="🌿",
    layout="wide"
)

# ── GLOBAL STYLES ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Serif+Display&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F4F7F4;
}

/* Remove default Streamlit padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 860px;
}

/* ── HERO ── */
.ede-hero {
    background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 100%);
    border-radius: 18px;
    padding: 40px 36px 32px;
    margin-bottom: 32px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(27,67,50,0.18);
}
.ede-hero h1 {
    font-family: 'DM Serif Display', serif;
    color: #D8F3DC;
    font-size: 34px;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
.ede-hero p {
    color: #95D5B2;
    font-size: 14.5px;
    margin: 0;
    line-height: 1.6;
}
.ede-badge {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #B7E4C7;
    font-size: 11px;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 16px;
    letter-spacing: 1px;
    text-transform: uppercase;
    border: 1px solid rgba(183,228,199,0.3);
}

/* ── SECTION LABELS ── */
.step-label {
    font-size: 11px;
    font-weight: 700;
    color: #2D6A4F;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 32px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.step-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #C8E6C9;
}

/* ── SIMILARITY BARS ── */
.chunk-card {
    background: #ffffff;
    border: 1px solid #C8E6C9;
    border-left: 5px solid #40916C;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 14px;
    font-size: 13.5px;
    line-height: 1.75;
    color: #1F2937;
    box-shadow: 0 1px 6px rgba(64,145,108,0.07);
}
.chunk-meta {
    font-size: 11px;
    font-weight: 700;
    color: #40916C;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sim-bar-wrap {
    flex: 1;
    background: #E8F5E9;
    border-radius: 6px;
    height: 7px;
    overflow: hidden;
    max-width: 160px;
}
.sim-bar-fill {
    height: 100%;
    border-radius: 6px;
    background: linear-gradient(90deg, #40916C, #74C69D);
}

/* ── ADVISORY OUTPUT ── */
.advisory-section {
    background: #ffffff;
    border: 1px solid #C8E6C9;
    border-radius: 14px;
    padding: 28px 30px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(64,145,108,0.08);
}
.advisory-section-title {
    font-size: 12px;
    font-weight: 700;
    color: #2D6A4F;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 2px solid #E8F5E9;
}
.advisory-body {
    font-size: 14.5px;
    line-height: 1.85;
    color: #111827;
    white-space: pre-wrap;
}

/* ── ERROR BOX ── */
.error-box {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 5px solid #EF4444;
    border-radius: 10px;
    padding: 16px 20px;
    font-size: 13.5px;
    color: #7F1D1D;
    line-height: 1.7;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background-color: #1B4332;
}
section[data-testid="stSidebar"] * {
    color: #D8F3DC !important;
}
section[data-testid="stSidebar"] .sidebar-title {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: #D8F3DC !important;
    margin-bottom: 4px;
}
section[data-testid="stSidebar"] .sidebar-sub {
    font-size: 12px;
    color: #95D5B2 !important;
    margin-bottom: 20px;
    line-height: 1.5;
}
section[data-testid="stSidebar"] hr {
    border-color: #2D6A4F !important;
    margin: 16px 0;
}
section[data-testid="stSidebar"] .sidebar-step {
    background: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 13px;
    line-height: 1.6;
    color: #B7E4C7 !important;
}
section[data-testid="stSidebar"] .sidebar-step strong {
    color: #D8F3DC !important;
    display: block;
    margin-bottom: 3px;
}
section[data-testid="stSidebar"] .sidebar-tip {
    background: rgba(64,145,108,0.25);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
    color: #95D5B2 !important;
    line-height: 1.6;
    margin-top: 16px;
}

/* ── BUTTON ── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2D6A4F, #40916C);
    border: none;
    border-radius: 10px;
    color: white;
    font-weight: 600;
    font-size: 15px;
    padding: 0.65rem 1.2rem;
    box-shadow: 0 3px 12px rgba(45,106,79,0.35);
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #1B4332, #2D6A4F);
    box-shadow: 0 5px 18px rgba(27,67,50,0.45);
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button[kind="secondary"] {
    border: 1.5px solid #40916C;
    color: #2D6A4F;
    border-radius: 10px;
    font-weight: 600;
    background: white;
    transition: all 0.2s ease;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    background: #E8F5E9;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🌿 EDE Social Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">BGT 4th Edition · EDE Fundazioa<br>RAG-Powered Case Intelligence</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**📋 How to use this tool**", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-step">
        <strong>① Describe the client</strong>
        Write a brief case profile — who they are, their situation, barriers, and what support they need.
    </div>
    <div class="sidebar-step">
        <strong>② Run the search</strong>
        The system encodes your case and retrieves the 3 most similar records from 13 historical reports stored in Qdrant.
    </div>
    <div class="sidebar-step">
        <strong>③ Read the advisory</strong>
        Claude analyses the retrieved records and generates a structured intervention brief tailored to your client.
    </div>
    <div class="sidebar-step">
        <strong>④ Download or reset</strong>
        Save the brief as a .txt file, or click <em>New Case</em> to start fresh.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**ℹ️ About**", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-tip">
    This tool uses <strong>Retrieval-Augmented Generation (RAG)</strong>: your query is matched against a local Qdrant vector database, and retrieved context is sent to Claude to ground the advisory in real case history — not hallucinated responses.<br><br>
    Built with: <strong>Streamlit · Qdrant · sentence-transformers · Claude API</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.caption("EDE Fundazioa · BGT 4th Edition · 2025")

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ede-hero">
    <div class="ede-badge">🌿 EDE Fundazioa · BGT Challenge</div>
    <h1>Social Support Advisor</h1>
    <p>Describe a client below — the system retrieves similar historical cases from your 13 reports<br>
    and generates a tailored, evidence-grounded intervention brief via Claude.</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD MODELS ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_qdrant():
    return QdrantClient(host="localhost", port=6333)

@st.cache_resource
def load_claude():
    return anthropic.Anthropic()

try:
    embedder = load_embedder()
    qdrant   = load_qdrant()
    claude   = load_claude()
except Exception as e:
    st.markdown(f"""
    <div class="error-box">
    ⚠️ <strong>Connection problem:</strong><br><br>
    {str(e)}<br><br>
    Make sure:<br>
    • Docker is running and your Qdrant container is started<br>
    • Your <code>ANTHROPIC_API_KEY</code> is set in your terminal<br>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

COLLECTION = "social_work_reports"
TOP_K      = 3

# ── SESSION STATE ──────────────────────────────────────────────────────────────
if "result_ready" not in st.session_state:
    st.session_state.result_ready = False
if "advisory" not in st.session_state:
    st.session_state.advisory = ""
if "client_case_snapshot" not in st.session_state:
    st.session_state.client_case_snapshot = ""
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

# ── INPUT ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="step-label">✍️ Step 1 — Describe the client</div>', unsafe_allow_html=True)

client_case = st.text_area(
    label="Client profile and challenges:",
    height=160,
    placeholder="Example: Amara, age 28, recently arrived refugee, limited language skills, two young children, struggling to access housing support and employment services...",
    label_visibility="collapsed"
)

col1, col2 = st.columns([3, 1])
with col1:
    run = st.button("🔍 Find Similar Cases & Generate Advisory", type="primary", use_container_width=True)
with col2:
    reset = st.button("🔄 New Case", type="secondary", use_container_width=True)

if reset:
    st.session_state.result_ready = False
    st.session_state.advisory = ""
    st.session_state.client_case_snapshot = ""
    st.session_state.retrieved_chunks = []
    st.rerun()

# ── PIPELINE ───────────────────────────────────────────────────────────────────
if run:
    if not client_case.strip():
        st.warning("Please describe the client case before running.")
        st.stop()

    st.session_state.result_ready = False

    # STEP 1 — Embed
    with st.expander("⚙️ Technical log — click to expand", expanded=False):
        log = st.empty()
        log.info("Encoding your case profile into a search vector...")
        query_vector = embedder.encode(client_case).tolist()
        log.success("✅ Case profile encoded successfully.")

        # STEP 2 — Search Qdrant
        log.info("Searching Qdrant collection for similar records...")
        try:
            search_result = qdrant.query_points(
                collection_name=COLLECTION,
                query=query_vector,
                limit=TOP_K,
                with_payload=True
            )
            results = search_result.points
            log.success(f"✅ Retrieved {len(results)} records from Qdrant.")
        except Exception as e:
            log.error(f"Qdrant search failed: {str(e)}")
            st.markdown(f"""
            <div class="error-box">
            <strong>Could not search Qdrant.</strong><br><br>
            Error: {str(e)}<br><br>
            Check that Qdrant is running and the collection <code>social_work_reports</code> exists.<br>
            Verify at: <a href="http://localhost:6333/collections" target="_blank">http://localhost:6333/collections</a>
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    if not results:
        st.info("No matching records found. Your Qdrant collection may be empty.")
        st.stop()

    # Save chunks to session state
    chunks = []
    retrieved_context = ""
    for i, hit in enumerate(results, 1):
        score_pct = round(hit.score * 100, 1)
        chunk_text = (
            hit.payload.get("text") or
            hit.payload.get("content") or
            hit.payload.get("chunk") or
            hit.payload.get("page_content") or
            str(hit.payload)
        )
        source = hit.payload.get("source", f"Report {i}")
        chunks.append({"index": i, "score": score_pct, "source": source, "text": chunk_text})
        retrieved_context += f"\n\n--- Historical Record {i} (similarity: {score_pct}%) from {source} ---\n{chunk_text}"

    # STEP 3 — Claude
    system_prompt = """You are an experienced social work advisor supporting caseworkers at EDE Fundazioa in the Basque Country, Spain.

You will receive:
1. A new client profile written by a social worker
2. Retrieved excerpts from historical case reports similar to the new case

Structure your response using exactly these four labelled sections:

RISK FACTORS & BARRIERS
[Identify the client's key risk factors and barriers from their profile]

LESSONS FROM HISTORICAL RECORDS
[Draw specific lessons from the retrieved historical records — reference them by number e.g. Record 1, Record 2]

RECOMMENDED INTERVENTIONS
[Recommend 2–3 prioritised intervention options with clear rationale for each]

INFORMATION GAPS
[Note any missing details the social worker should gather before proceeding]

Be concise, professional, and evidence-based. Write in plain English suitable for a frontline social worker.
Do NOT invent case details that are not in the provided records or client profile."""

    user_prompt = f"""NEW CLIENT PROFILE:
{client_case}

RETRIEVED HISTORICAL RECORDS FROM DATABASE:
{retrieved_context}

Please write a structured advisory brief for the social worker handling this case."""

    with st.spinner("Claude is reading the records and writing your advisory brief..."):
        try:
            response = claude.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1200,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            advisory = response.content[0].text
        except Exception as e:
            st.markdown(f"""
            <div class="error-box">
            <strong>Claude API call failed.</strong><br><br>
            Error: {str(e)}<br><br>
            Check that your <code>ANTHROPIC_API_KEY</code> is set correctly in your terminal.
            </div>
            """, unsafe_allow_html=True)
            st.stop()

    st.session_state.advisory = advisory
    st.session_state.client_case_snapshot = client_case
    st.session_state.retrieved_chunks = chunks
    st.session_state.result_ready = True

# ── DISPLAY RESULTS ────────────────────────────────────────────────────────────
if st.session_state.result_ready:

    # Retrieved records
    st.markdown('<div class="step-label">📂 Step 2 — Most similar historical records</div>', unsafe_allow_html=True)

    for chunk in st.session_state.retrieved_chunks:
        bar_width = chunk["score"]
        st.markdown(f"""
        <div class="chunk-card">
            <div class="chunk-meta">
                Record {chunk["index"]} &nbsp;·&nbsp; {chunk["source"]}
                <div class="sim-bar-wrap">
                    <div class="sim-bar-fill" style="width:{bar_width}%"></div>
                </div>
                {chunk["score"]}% match
            </div>
            {chunk["text"][:600]}{"…" if len(chunk["text"]) > 600 else ""}
        </div>
        """, unsafe_allow_html=True)

    # Advisory brief — split into sections
    st.markdown('<div class="step-label">📋 Step 3 — Claude Advisory Brief</div>', unsafe_allow_html=True)

    advisory_text = st.session_state.advisory

    section_keys = [
        ("RISK FACTORS & BARRIERS", "⚠️ Risk Factors & Barriers"),
        ("LESSONS FROM HISTORICAL RECORDS", "📖 Lessons from Historical Records"),
        ("RECOMMENDED INTERVENTIONS", "✅ Recommended Interventions"),
        ("INFORMATION GAPS", "🔍 Information Gaps"),
    ]

    def md_to_html(text):
        """Convert basic markdown to HTML for clean rendering."""
        import re
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = text.replace('\n', '<br>')
        return text

    def extract_section(text, start_key, all_keys):
        """Extract content between start_key and the next section key."""
        start_idx = text.upper().find(start_key)
        if start_idx == -1:
            return None
        content_start = text.find("\n", start_idx) + 1
        end_idx = len(text)
        for k, _ in all_keys:
            if k == start_key:
                continue
            idx = text.upper().find(k, content_start)
            if idx != -1 and idx < end_idx:
                end_idx = idx
        return text[content_start:end_idx].strip()

    sections_found = False
    for raw_key, display_label in section_keys:
        content = extract_section(advisory_text, raw_key, section_keys)
        if content:
            sections_found = True
            st.markdown(f"""
            <div class="advisory-section">
                <div class="advisory-section-title">{display_label}</div>
                <div class="advisory-body">{md_to_html(content)}</div>
            </div>
            """, unsafe_allow_html=True)

    # Fallback: if Claude didn't use the section format, show full output
    if not sections_found:
        st.markdown(f"""
        <div class="advisory-section">
            <div class="advisory-section-title">📋 Advisory Brief</div>
            <div class="advisory-body">{md_to_html(advisory_text)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download
    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="⬇️ Download Advisory Brief (.txt)",
        data=f"EDE FUNDAZIOA — SOCIAL SUPPORT ADVISORY BRIEF\n{'='*50}\n\nCLIENT PROFILE:\n{st.session_state.client_case_snapshot}\n\n{'='*50}\n\nADVISORY BRIEF:\n{st.session_state.advisory}",
        file_name="ede_advisory_brief.txt",
        mime="text/plain",
        use_container_width=True
    )
