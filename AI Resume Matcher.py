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
    layout="wide",
)

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

# ------------------------------------------------
# BACKGROUND FUNCTION
# ------------------------------------------------

def set_background(show_image=True):

    base_color = "#071c2c"

    if show_image:

        image_path = "background.png"

        with open(image_path, "rb") as img:
            encoded = base64.b64encode(img.read()).decode()

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-color:{base_color};
                background-image:url("data:image/png;base64,{encoded}");
                background-size:cover;
                background-position:center;
                background-repeat:no-repeat;
                background-attachment:fixed;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <style>

            .stApp {{
                background-color:{base_color};
                background-image:none;
            }}

            </style>
            """,
            unsafe_allow_html=True
        )

set_background(not st.session_state.analysis_done)

# ------------------------------------------------
# GLOBAL UI STYLE
# ------------------------------------------------

st.markdown(
"""
<style>

.block-container{
padding-top:0rem;
padding-left:1rem;
padding-right:1rem;
max-width:100%;
}

section[data-testid="stSidebar"]{
background:#061a29;
}

h1,h2,h3,h4,h5,h6{
color:white;
}

p,span,label{
color:#e6e6e6;
}

.stButton>button{
background:#ffffff;
color:black;
font-weight:bold;
border-radius:8px;
}

.section-card{
background:rgba(7,28,44,0.9);
padding:25px;
border-radius:15px;
margin-bottom:30px;
}

.result-card{
background:rgba(7,28,44,0.95);
padding:20px;
border-radius:10px;
margin-bottom:20px;
}

</style>
""",
unsafe_allow_html=True
)

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
        "messages": [{"role":"user","content":prompt}],
        "temperature":0.2
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        return "AI Error"

    return response.json()["choices"][0]["message"]["content"]

# ------------------------------------------------
# FILE EXTRACTION
# ------------------------------------------------

def extract_text_from_pdf(file):

    reader = PyPDF2.PdfReader(file)

    text=""

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

    results=[]

    for name,text in resume_texts:

        resume_embedding = embedding_model.encode(text, convert_to_tensor=True)

        score = util.cos_sim(jd_embedding,resume_embedding).item()

        score = round(score*100,2)

        experience = 3

        results.append((name,text,score,experience))

    return sorted(results,key=lambda x:x[2],reverse=True)

# ------------------------------------------------
# INTERVIEW QUESTIONS
# ------------------------------------------------

def generate_interview_questions(jd_text):

    prompt=f"""
Create 10 interview questions strictly based on this job description.

Focus on skills and tools mentioned.

Job Description:
{jd_text}

Return numbered questions only.
"""

    return call_ai(prompt)

# ------------------------------------------------
# AI RECOMMENDATION
# ------------------------------------------------

def generate_recommendation(jd,resume,score):

    prompt=f"""
Analyze candidate suitability.

Candidate Score: {score}

Job Description:
{jd}

Resume:
{resume[:2000]}

Provide:

Candidate fit
Strengths
Missing skills
Hiring recommendation
"""

    return call_ai(prompt)

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
# MAIN LOGIC
# ------------------------------------------------

if analyze:

    st.session_state.analysis_done=True

    if not resume_files or not jd_input:

        st.warning("Please upload resumes and add job description")

    else:

        resume_texts=[]

        for file in resume_files:

            ext=os.path.splitext(file.name)[1]

            if ext==".pdf":
                text=extract_text_from_pdf(file)
            else:
                text=extract_text_from_docx(file)

            resume_texts.append((file.name,text))

        results=compute_similarity(resume_texts,jd_input)

        top_candidate=results[0]

        questions=generate_interview_questions(jd_input)

        recommendation=generate_recommendation(
            jd_input,
            top_candidate[1],
            top_candidate[2]
        )

        # ------------------------------------------------
        # INTERVIEW QUESTIONS
        # ------------------------------------------------

        st.markdown('<div class="section-card">',unsafe_allow_html=True)

        st.subheader("Interview Questions")

        st.write(questions)

        st.markdown('</div>',unsafe_allow_html=True)

        # ------------------------------------------------
        # AI RECOMMENDATION
        # ------------------------------------------------

        st.markdown('<div class="section-card">',unsafe_allow_html=True)

        st.subheader("AI Hiring Recommendation")

        st.write(recommendation)

        st.markdown('</div>',unsafe_allow_html=True)

        # ------------------------------------------------
        # RANKING
        # ------------------------------------------------

        df=pd.DataFrame(
            [(i+1,r[0],r[2]) for i,r in enumerate(results)],
            columns=["Rank","Candidate Name","Match Score %"]
        )

        st.markdown('<div class="section-card">',unsafe_allow_html=True)

        st.subheader("Candidate Ranking")

        st.dataframe(df,use_container_width=True)

        st.markdown('</div>',unsafe_allow_html=True)

        # ------------------------------------------------
        # CANDIDATE ANALYSIS
        # ------------------------------------------------

        st.subheader("Candidate Analysis")

        for rank,(name,text,score,experience) in enumerate(results,1):

            st.markdown('<div class="result-card">',unsafe_allow_html=True)

            st.markdown(f"### {rank}. {name}")

            st.write(f"Match Score: **{score}%**")

            st.write(f"Estimated Experience: **{experience} years**")

            st.markdown('</div>',unsafe_allow_html=True)
