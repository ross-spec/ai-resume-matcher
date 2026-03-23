import streamlit as st
import os
import requests
import pandas as pd
import PyPDF2
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sentence_transformers import SentenceTransformer, util

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="HireAI – Resume Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# GLOBAL CSS  (premium dark + electric accent)
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14 !important;
    color: #e8eaf0 !important;
}

/* ── hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 3rem 4rem !important; max-width: 1400px !important; }

/* ── animated gradient background ── */
body::before {
    content: '';
    position: fixed;
    top: -40%;
    left: -20%;
    width: 70vw;
    height: 70vw;
    background: radial-gradient(circle, rgba(56,189,248,0.06) 0%, transparent 70%);
    animation: drift 18s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
body::after {
    content: '';
    position: fixed;
    bottom: -30%;
    right: -10%;
    width: 55vw;
    height: 55vw;
    background: radial-gradient(circle, rgba(139,92,246,0.07) 0%, transparent 70%);
    animation: drift2 22s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes drift  { to { transform: translate(6%, 8%); } }
@keyframes drift2 { to { transform: translate(-6%,-8%); } }

/* ── hero headline ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    line-height: 1.12;
    letter-spacing: -0.03em;
    margin: 0 0 .6rem;
}
.hero-title span.glow {
    background: linear-gradient(120deg, #38bdf8 0%, #818cf8 60%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    font-weight: 300;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 1.8rem;
}
.hero-desc {
    font-size: 1.05rem;
    color: #94a3b8;
    line-height: 1.7;
    max-width: 560px;
}
.feature-pill {
    display: inline-block;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.18);
    border-radius: 999px;
    padding: .22rem .85rem;
    font-family: 'DM Mono', monospace;
    font-size: .72rem;
    color: #38bdf8;
    margin: .25rem .25rem .25rem 0;
    letter-spacing: .06em;
}

/* ── upload panel card ── */
.upload-card {
    background: linear-gradient(160deg, rgba(15,23,42,0.95) 0%, rgba(15,23,42,0.8) 100%);
    border: 1px solid rgba(56,189,248,0.14);
    border-radius: 20px;
    padding: 2rem 1.8rem;
    box-shadow: 0 0 40px rgba(56,189,248,0.04), 0 20px 60px rgba(0,0,0,0.5);
    backdrop-filter: blur(20px);
}
.card-label {
    font-family: 'DM Mono', monospace;
    font-size: .68rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: .5rem;
}

/* ── metric tiles ── */
[data-testid="metric-container"] {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(56,189,248,0.12) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    backdrop-filter: blur(12px);
    transition: border-color .2s;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(56,189,248,0.35) !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important;
    letter-spacing: .14em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}

/* ── section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: -.02em;
    color: #f1f5f9;
    margin: 2.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: .55rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.25), transparent);
    margin-left: .5rem;
}

/* ── top candidate banner ── */
.top-banner {
    background: linear-gradient(120deg, rgba(56,189,248,0.1) 0%, rgba(129,140,248,0.1) 100%);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.top-banner-icon { font-size: 2rem; }
.top-banner-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #38bdf8;
}
.top-banner-score {
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    color: #64748b;
}

/* ── candidate card ── */
.cand-card {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(56,189,248,0.10);
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
    transition: border-color .2s, box-shadow .2s;
}
.cand-card:hover {
    border-color: rgba(56,189,248,0.28);
    box-shadow: 0 0 24px rgba(56,189,248,0.07);
}
.cand-rank {
    font-family: 'DM Mono', monospace;
    font-size: .7rem;
    color: #475569;
    letter-spacing: .12em;
}
.cand-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: .15rem 0 .6rem;
}
.score-bar-bg {
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    height: 6px;
    margin-bottom: 1.2rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg,#38bdf8,#818cf8);
    transition: width .8s ease;
}
.skills-block, .rec-block {
    background: rgba(0,0,0,0.2);
    border-radius: 10px;
    padding: .9rem 1.1rem;
    font-family: 'DM Mono', monospace;
    font-size: .78rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-top: .5rem;
}
.block-title {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .18em;
    color: #475569;
    margin-bottom: .4rem;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid rgba(56,189,248,0.10) !important;
}

/* ── buttons ── */
.stButton > button, [data-testid="stDownloadButton"] > button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    border-radius: 10px !important;
    transition: all .2s !important;
}
.stButton > button[kind="primary"], .stButton > button {
    background: linear-gradient(120deg,#38bdf8,#818cf8) !important;
    color: #080c14 !important;
    border: none !important;
    padding: .6rem 2rem !important;
}
.stButton > button:hover {
    opacity: .88 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(56,189,248,0.25) !important;
}
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid rgba(56,189,248,0.30) !important;
    color: #38bdf8 !important;
}

/* ── file uploader & text area ── */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.2) !important;
    border: 1px dashed rgba(56,189,248,0.22) !important;
    border-radius: 12px !important;
    padding: .5rem !important;
}
textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(56,189,248,0.14) !important;
    border-radius: 10px !important;
    color: #cbd5e1 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .82rem !important;
}
textarea:focus { border-color: rgba(56,189,248,0.40) !important; }

/* ── spinner ── */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* ── divider ── */
hr { border-color: rgba(56,189,248,0.08) !important; }

/* ── questions block ── */
.q-block {
    background: rgba(15,23,42,0.8);
    border-left: 3px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.4rem;
    font-family: 'DM Mono', monospace;
    font-size: .82rem;
    color: #94a3b8;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────
col_hero, col_panel = st.columns([1.15, 1], gap="large")

with col_hero:
    st.markdown("""
    <p class="hero-sub">⚡ Powered by AI · Built for modern hiring</p>
    <h1 class="hero-title">
        Screen smarter.<br>
        <span class="glow">Hire faster.</span>
    </h1>
    <p class="hero-desc">
        Drop your resumes and job description — HireAI ranks every candidate
        using semantic AI matching, extracts skills automatically, generates
        interview questions, and gives you a sharp hiring recommendation.
    </p>
    <div style="margin-top:1.4rem">
        <span class="feature-pill">⚡ Semantic Matching</span>
        <span class="feature-pill">🔍 Skill Extraction</span>
        <span class="feature-pill">💬 Interview Questions</span>
        <span class="feature-pill">📋 Hire Recommendations</span>
        <span class="feature-pill">📥 CSV Export</span>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# UPLOAD PANEL
# ──────────────────────────────────────────────
with col_panel:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)

    st.markdown('<p class="card-label">📂 Resume Upload</p>', unsafe_allow_html=True)
    resume_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="card-label">📝 Job Description</p>', unsafe_allow_html=True)
    jd_input = st.text_area(
        "Paste job description here…",
        height=200,
        placeholder="e.g. We are looking for a Senior Python Developer with 5+ years of experience in FastAPI, AWS, and machine learning pipelines…",
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze = st.button("⚡  Analyze Candidates", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────
# AI CONFIG
# ──────────────────────────────────────────────
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def call_ai(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }
    r = requests.post(API_URL, headers=headers, json=payload)
    if r.status_code != 200:
        return "⚠️ AI service unavailable."
    return r.json()["choices"][0]["message"]["content"]


def extract_text(file) -> str:
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        reader = PyPDF2.PdfReader(file)
        return "".join(p.extract_text() or "" for p in reader.pages)
    return docx2txt.process(file)


def extract_skills(resume_text: str) -> str:
    return call_ai(f"""
Extract the top 8–12 professional skills from this resume.
Return ONLY a clean bullet list, no preamble.

Resume:
{resume_text[:2500]}
""")


def compute_similarity(resume_texts, jd_text):
    jd_emb = embedding_model.encode(jd_text, convert_to_tensor=True)
    results = []
    for name, text in resume_texts:
        emb = embedding_model.encode(text, convert_to_tensor=True)
        score = round(util.cos_sim(jd_emb, emb).item() * 100, 1)
        results.append((name, text, score))
    return sorted(results, key=lambda x: x[2], reverse=True)


def generate_questions(jd: str) -> str:
    return call_ai(f"""
Generate exactly 10 sharp, role-specific interview questions based on this job description.
Return ONLY numbered questions, no extra commentary.

Job Description:
{jd}
""")


def generate_recommendation(jd: str, resume: str, score: float) -> str:
    return call_ai(f"""
You are a senior hiring manager. Analyse this candidate concisely.

Match Score: {score}%

Job Description:
{jd[:1200]}

Resume:
{resume[:1800]}

Return three short sections:
★ STRENGTHS (2–3 bullets)
⚠ GAPS (2–3 bullets)
✅ RECOMMENDATION (1–2 sentences — Hire / Maybe / Skip)
""")


def score_color(score: float) -> str:
    if score >= 70:
        return "#22c55e"
    elif score >= 45:
        return "#f59e0b"
    return "#ef4444"


# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────
if analyze:
    if not resume_files or not jd_input.strip():
        st.warning("⚠️  Please upload at least one resume and provide a job description.")
        st.stop()

    with st.spinner("🔍  Running semantic analysis…"):
        resume_texts = [(f.name, extract_text(f)) for f in resume_files]
        results = compute_similarity(resume_texts, jd_input)
        questions = generate_questions(jd_input)

    scores = [r[2] for r in results]
    avg_score = round(float(np.mean(scores)), 1)
    top_score = max(scores)

    # ── KPI TILES ──
    st.markdown('<div class="section-header">📊 Screening Dashboard</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Resumes Uploaded",  len(resume_files))
    k2.metric("Candidates Ranked", len(results))
    k3.metric("Top Match Score",   f"{top_score}%")
    k4.metric("Avg Match Score",   f"{avg_score}%")

    # ── TOP CANDIDATE BANNER ──
    top = results[0]
    st.markdown(f"""
    <div class="top-banner">
        <div class="top-banner-icon">🏆</div>
        <div>
            <div class="top-banner-name">{top[0]}</div>
            <div class="top-banner-score">Best overall match · {top[2]}% similarity score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── SCORE CHART ──
    st.markdown('<div class="section-header">📈 Match Score Comparison</div>', unsafe_allow_html=True)

    names  = [r[0] for r in results]
    values = [r[2] for r in results]
    bar_colors = [score_color(v) for v in values]

    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.6)))
    fig.patch.set_facecolor("#080c14")
    ax.set_facecolor("#0d1626")

    bars = ax.barh(names, values, color=bar_colors, height=0.55, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Match Score (%)", color="#64748b", fontsize=9, fontfamily="monospace")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.xaxis.grid(True, color="rgba(255,255,255,0.05)", zorder=0)
    ax.set_axisbelow(True)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", ha="left", color="#94a3b8",
                fontsize=8, fontfamily="monospace")
    plt.tight_layout()
    st.pyplot(fig)

    # ── RANKING TABLE ──
    st.markdown('<div class="section-header">🗂 Candidate Ranking</div>', unsafe_allow_html=True)
    df = pd.DataFrame(
        [(i + 1, r[0], f"{r[2]}%") for i, r in enumerate(results)],
        columns=["Rank", "Candidate", "Match Score"]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Export CSV", csv_data, "candidate_ranking.csv", "text/csv")

    # ── INTERVIEW QUESTIONS ──
    st.markdown('<div class="section-header">💬 Interview Questions</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="q-block">{questions.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

    # ── CANDIDATE DEEP-DIVE ──
    st.markdown('<div class="section-header">🔬 Candidate Analysis</div>', unsafe_allow_html=True)

    for rank, (name, text, score) in enumerate(results, 1):

        fill_pct = int(score)
        bar_col = score_color(score)

        st.markdown(f"""
        <div class="cand-card">
            <div class="cand-rank">RANK #{rank}</div>
            <div class="cand-name">{name}</div>
            <div style="display:flex;align-items:center;gap:.8rem;margin-bottom:.5rem">
                <div class="score-bar-bg" style="flex:1">
                    <div class="score-bar-fill" style="width:{fill_pct}%;background:linear-gradient(90deg,{bar_col},{bar_col}88)"></div>
                </div>
                <span style="font-family:'DM Mono',monospace;font-size:.82rem;color:{bar_col};font-weight:500">{score}%</span>
            </div>
        """, unsafe_allow_html=True)

        with st.spinner(f"Extracting skills for {name}…"):
            skills = extract_skills(text)
        st.markdown('<div class="block-title">🔍 Extracted Skills</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="skills-block">{skills.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        with st.spinner(f"Generating recommendation for {name}…"):
            rec = generate_recommendation(jd_input, text, score)
        st.markdown('<div class="block-title">🤖 AI Hiring Recommendation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-block">{rec.replace(chr(10),"<br>")}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ── FOOTER ──
    st.markdown("""
    <div style="text-align:center;margin-top:4rem;padding-top:2rem;
                border-top:1px solid rgba(56,189,248,0.08)">
        <span style="font-family:'DM Mono',monospace;font-size:.7rem;
                     color:#334155;letter-spacing:.14em">
            HIREAI · AI-POWERED RESUME INTELLIGENCE · BUILT WITH ♥
        </span>
    </div>
    """, unsafe_allow_html=True)
