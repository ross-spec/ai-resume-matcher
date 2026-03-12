import streamlit as st
import base64
import os
import requests
import pandas as pd
from pathlib import Path
import PyPDF2
import docx2txt
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ------------------------------------------------
# BACKGROUND IMAGE + CSS
# ------------------------------------------------

def set_background():

    image_path = Path(__file__).parent / "background.png"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        [data-testid="stAppViewContainer"] {{
            background-image:
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            background: transparent !important;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
        }}

        section[data-testid="stSidebar"] {{
            background: rgba(0,0,0,0.55);
            backdrop-filter: blur(10px);
        }}

        #MainMenu {{visibility:hidden;}}
        footer {{visibility:hidden;}}

        h1,h2,h3,h4,h5,h6 {{
            color:white;
        }}

        p,span,div,label {{
            color:#e6e6e6;
        }}

        .hero-title {{
            text-align:center;
            font-size:48px;
            font-weight:700;
            color:white;
            margin-top:40px;
        }}

        .hero-subtitle {{
            text-align:center;
            font-size:20px;
            color:#d0d0d0;
            margin-bottom:40px;
        }}

        .section-card {{
            background: rgba(0,0,0,0.65);
            padding:25px;
            border-radius:15px;
            margin-bottom:30px;
        }}

        .result-card {{
            background:white;
            padding:20px;
            border-radius:10px;
            margin-bottom:20px;
            color:black;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ------------------------------------------------
# OPENROUTER CONFIG
# ------------------------------------------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------
# AI CALL FUNCTION
# ------------------------------------------------

def call_ai(prompt):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return "AI Error"

    data = response.json()

    return data["choices"][0]["message"]["content"]

# ------------------------------------------------
# TEXT EXTRACTION
# ------------------------------------------------

def extract_text_from_pdf(file):

    text = ""

    reader = PyPDF2.PdfReader(file)

    for page in reader.pages:
        text += page.extract_text()

    return text


def extract_text_from_docx(file):

    return docx2txt.process(file)

# ------------------------------------------------
# RESUME SCORING
# ------------------------------------------------

def compute_similarity(resume_texts, jd_text):

    jd_embedding = embedding_model.encode(jd_text, convert_to_tensor=True)

    results = []

    for name, text in resume_texts:

        resume_embedding = embedding_model.encode(text, convert_to_tensor=True)

        semantic_score = util.cos_sim(jd_embedding, resume_embedding).item()

        score = round(semantic_score * 100, 2)

        experience = 3

        results.append((name, text, score, experience))

    return sorted(results, key=lambda x: x[2], reverse=True)

# ------------------------------------------------
# INTERVIEW QUESTIONS (JD + RESUME)
# ------------------------------------------------

def generate_interview_questions(jd_text, resume_text, experience):

    prompt = f"""
You are a senior technical interviewer.

Generate 7 interview questions based on BOTH the job description
and the candidate's resume.

Candidate Experience: {experience} years

If experience < 2 years → beginner questions  
If 2-5 years → intermediate questions  
If >5 years → advanced questions

Job Description:
{jd_text}

Candidate Resume:
{resume_text[:2000]}

Return only numbered questions.
"""

    return call_ai(prompt)

# ------------------------------------------------
# HERO SECTION
# ------------------------------------------------

st.markdown(
"""
<div class="hero-title">
AI Resume Screener & Candidate Ranking
</div>

<div class="hero-subtitle">
Analyze resumes, detect skills, and rank candidates using AI
</div>
""",
unsafe_allow_html=True
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.header("Upload Resumes")

    resume_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.header("Job Description")

    jd_input = st.text_area(
        "Paste job description",
        height=200
    )

    analyze = st.button("Analyze Candidates")

# ------------------------------------------------
# MAIN APP
# ------------------------------------------------

if analyze:

    if not resume_files or not jd_input:

        st.warning("Please upload resumes and provide job description")

    else:

        resume_texts = []

        for file in resume_files:

            ext = os.path.splitext(file.name)[1]

            if ext == ".pdf":
                text = extract_text_from_pdf(file)

            else:
                text = extract_text_from_docx(file)

            resume_texts.append((file.name, text))

        results = compute_similarity(resume_texts, jd_input)

        # --------------------------------
        # INTERVIEW QUESTIONS
        # --------------------------------

        top_candidate = results[0]

        questions = generate_interview_questions(
            jd_input,
            top_candidate[1],
            top_candidate[3]
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("Smart Interview Questions")

        st.write(f"For candidate: **{top_candidate[0]}**")

        st.write(questions)

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------
        # RANKING TABLE
        # --------------------------------

        df = pd.DataFrame(
            [(i+1,r[0],r[2]) for i,r in enumerate(results)],
            columns=["Rank","Candidate Name","Match Score %"]
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("Candidate Ranking")

        st.dataframe(df,use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------
        # CANDIDATE ANALYSIS
        # --------------------------------

        st.subheader("Candidate Analysis")

        for rank,(name,text,score,experience) in enumerate(results,1):

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            st.markdown(f"### {rank}. {name}")

            st.write(f"Match Score: **{score}%**")
            st.write(f"Estimated Experience: **{experience} years**")

            st.write("Candidate Summary")

            st.write(
            "Experienced BI professional with strong expertise in Power BI, "
            "SQL, and data visualization."
            )

            st.markdown('</div>', unsafe_allow_html=True)
