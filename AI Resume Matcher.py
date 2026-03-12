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
# BACKGROUND STYLE
# ------------------------------------------------

def set_background():

    image_file = Path("background.png")

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>

        [data-testid="stAppViewContainer"] {{
            background-color:#071c2c;
            background-image:url("data:image/png;base64,{encoded}");
            background-position:85% center;
            background-size:contain;
            background-repeat:no-repeat;
            background-attachment:fixed;
        }}

        .main {{
            background:transparent;
        }}

        .block-container {{
            background:transparent;
            padding-top:3rem;
            max-width:1200px;
        }}

        section[data-testid="stSidebar"] {{
            background:#071c2c;
            border-right:1px solid rgba(255,255,255,0.05);
        }}

        #MainMenu {{visibility:hidden;}}
        footer {{visibility:hidden;}}
        header {{visibility:hidden;}}

        h1,h2,h3,h4,h5,h6 {{
            color:white;
        }}

        p,span,div,label {{
            color:#e6e6e6;
        }}

        .section-card {{
            background:rgba(7,28,44,0.85);
            padding:25px;
            border-radius:15px;
            margin-bottom:30px;
            backdrop-filter:blur(8px);
            border:1px solid rgba(255,255,255,0.05);
        }}

        .result-card {{
            background:rgba(7,28,44,0.92);
            padding:20px;
            border-radius:10px;
            margin-bottom:20px;
            color:white;
            border:1px solid rgba(255,255,255,0.05);
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ------------------------------------------------
# AI CONFIG
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

        score = util.cos_sim(jd_embedding, resume_embedding).item()

        score = round(score * 100, 2)

        experience = 3

        results.append((name, text, score, experience))

    return sorted(results, key=lambda x: x[2], reverse=True)

# ------------------------------------------------
# INTERVIEW QUESTIONS
# ------------------------------------------------

def generate_interview_questions(jd_text, resume_text, experience):

    prompt = f"""
You are a senior technical interviewer.

Generate 10 interview questions based on BOTH the job description
and the candidate resume.

Candidate Experience: {experience} years

Difficulty:
0-2 years → beginner
2-5 years → intermediate
5+ years → advanced

Job Description:
{jd_text}

Candidate Resume:
{resume_text[:2000]}

Return ONLY 10 numbered interview questions.
"""

    return call_ai(prompt)

# ------------------------------------------------
# AI HIRING RECOMMENDATION
# ------------------------------------------------

def generate_recommendation(jd_text, resume_text, score):

    prompt = f"""
You are an AI hiring assistant.

Based on the resume and job description,
provide a hiring recommendation.

Candidate Score: {score}%

Job Description:
{jd_text}

Candidate Resume:
{resume_text[:2000]}

Provide:
• Candidate fit level
• Key strengths
• Missing skills
• Final hiring recommendation
"""

    return call_ai(prompt)

# ------------------------------------------------
# SIDEBAR INPUT
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
# MAIN APPLICATION
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

        st.subheader("Interview Questions")

        st.write(f"For candidate: **{top_candidate[0]}**")

        st.write(questions)

        st.markdown('</div>', unsafe_allow_html=True)

        # --------------------------------
        # AI RECOMMENDATION
        # --------------------------------

        recommendation = generate_recommendation(
            jd_input,
            top_candidate[1],
            top_candidate[2]
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("AI Hiring Recommendation")

        st.write(recommendation)

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

            st.markdown('</div>', unsafe_allow_html=True)
