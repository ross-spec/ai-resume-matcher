import os
import re
import json
import requests
import PyPDF2
import docx2txt
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# Use Streamlit Secrets for API key
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "mistralai/mistral-7b-instruct:free"

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# AI CALL FUNCTION
# -----------------------------

def call_ai(prompt):

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(API_URL, headers=headers, data=json.dumps(body))

    if response.status_code != 200:
        return ""

    result = response.json()

    return result["choices"][0]["message"]["content"]

# -----------------------------
# TEXT EXTRACTION
# -----------------------------

def extract_text_from_pdf(path):

    text = ""

    with open(path, "rb") as file:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            txt = page.extract_text()

            if txt:
                text += txt

    return text


def extract_text_from_docx(path):

    return docx2txt.process(path)

# -----------------------------
# KEYWORD EXTRACTION
# -----------------------------

def extract_keywords(text):

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    freq = pd.Series(words).value_counts()

    return list(freq.head(20).index)

# -----------------------------
# AI SKILL EXTRACTION
# -----------------------------

@st.cache_data
def extract_skills_ai(text):

    prompt = f"""
    Extract technical skills from the text below.

    Return comma separated list only.

    Text:
    {text[:2000]}
    """

    result = call_ai(prompt)

    skills = result.split(",")

    return [s.strip().lower() for s in skills]

# -----------------------------
# CANDIDATE SUMMARY
# -----------------------------

@st.cache_data
def generate_candidate_summary(resume_text):

    prompt = f"""
    Analyze this resume and provide:

    - Candidate role
    - Years of experience
    - Key skills
    - Strengths

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)

# -----------------------------
# SKILL GAP ANALYSIS
# -----------------------------

@st.cache_data
def skill_gap_analysis(jd_text, resume_text):

    prompt = f"""
    Compare the job description and resume.

    Provide:

    1. Required skills
    2. Candidate skills
    3. Missing skills

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}
    """

    return call_ai(prompt)

# -----------------------------
# INTERVIEW QUESTIONS
# -----------------------------

@st.cache_data
def generate_interview_questions(jd_text, resume_text):

    prompt = f"""
    Based on the job description and resume,
    generate 5 technical interview questions.

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}
    """

    return call_ai(prompt)

# -----------------------------
# SIMILARITY SCORING
# -----------------------------

def compute_similarity(resume_texts, jd_text):

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    jd_keywords = extract_keywords(jd_text)

    jd_skills = extract_skills_ai(jd_text)

    results = []

    for name, text in resume_texts:

        if not text.strip():

            results.append((name, 0))

            continue

        resume_embedding = model.encode(text, convert_to_tensor=True)

        semantic_score = util.cos_sim(jd_embedding, resume_embedding).item()

        keyword_matches = sum(1 for k in jd_keywords if k in text.lower())

        keyword_score = keyword_matches / len(jd_keywords) if jd_keywords else 0

        resume_skills = extract_skills_ai(text)

        skill_matches = len(set(jd_skills) & set(resume_skills))

        skill_score = skill_matches / len(jd_skills) if jd_skills else 0

        total_score = (
            semantic_score * 0.5 +
            keyword_score * 0.2 +
            skill_score * 0.3
        )

        results.append((name, round(total_score * 100, 2)))

    return sorted(results, key=lambda x: x[1], reverse=True)

# -----------------------------
# UI HEADER
# -----------------------------

st.markdown(
"""
<style>
.title{
background-color:#4B8BBE;
color:white;
padding:15px;
border-radius:10px;
text-align:center;
font-size:30px;
font-weight:bold;
}
</style>
""",
unsafe_allow_html=True
)

st.markdown("<div class='title'>📄 AI Resume Screener & JD Matcher</div>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("Upload Resumes")

    resume_files = st.file_uploader(
        "Upload Resume Files",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

# -----------------------------
# MAIN LAYOUT
# -----------------------------

col1, col2 = st.columns([1,2])

with col1:

    st.subheader("Job Description")

    jd_input = st.text_area(
        "Paste Job Description",
        height=300
    )

with col2:

    st.subheader("Matching Results")

    if st.button("Match Resumes"):

        if not resume_files or not jd_input.strip():

            st.warning("Please upload resumes and enter job description.")

        else:

            st.info("Processing resumes...")

            resume_texts = []

            for uploaded_file in resume_files:

                ext = os.path.splitext(uploaded_file.name)[1]

                temp = uploaded_file.name

                with open(temp, "wb") as f:

                    f.write(uploaded_file.read())

                if ext == ".pdf":

                    text = extract_text_from_pdf(temp)

                else:

                    text = extract_text_from_docx(temp)

                resume_texts.append((uploaded_file.name, text))

                os.remove(temp)

            scores = compute_similarity(resume_texts, jd_input)

            st.success("Matching Complete")

            df = pd.DataFrame(scores, columns=["Candidate Name", "Match Score (%)"])

            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "resume_results.csv",
                "text/csv"
            )

            st.markdown("---")

            for i,(name,score) in enumerate(scores,1):

                st.markdown(f"### {i}. {name}")

                st.write(f"Match Score: {score}%")

                resume_text = next(t for f,t in resume_texts if f==name)

                colA,colB,colC = st.columns(3)

                with colA:

                    if st.button(f"AI Summary {i}"):

                        summary = generate_candidate_summary(resume_text)

                        st.write(summary)

                with colB:

                    if st.button(f"Skill Gap {i}"):

                        gap = skill_gap_analysis(jd_input,resume_text)

                        st.write(gap)

                with colC:

                    if st.button(f"Interview Questions {i}"):

                        questions = generate_interview_questions(jd_input,resume_text)

                        st.write(questions)
