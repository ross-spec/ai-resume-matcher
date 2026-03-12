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

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL = "mistralai/mistral-7b-instruct:free"

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
        return "AI API Error"

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

            page_text = page.extract_text()

            if page_text:
                text += page_text

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
# AI FUNCTIONS
# -----------------------------

@st.cache_data
def generate_candidate_summary(resume_text):

    prompt = f"""
    Analyze the resume and provide:

    - Candidate role
    - Estimated experience
    - Key technical skills
    - Strengths

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)


@st.cache_data
def skill_gap_analysis(jd_text, resume_text):

    prompt = f"""
    Compare job description and resume.

    Provide:

    Required Skills
    Candidate Skills
    Missing Skills

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}
    """

    return call_ai(prompt)


@st.cache_data
def generate_interview_questions(jd_text, resume_text):

    prompt = f"""
    Generate 5 technical interview questions based on the job description and resume.

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}
    """

    return call_ai(prompt)

# -----------------------------
# MATCHING FUNCTION
# -----------------------------

def compute_similarity(resume_texts, jd_text):

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    jd_keywords = extract_keywords(jd_text)

    results = []

    for name, text in resume_texts:

        if not text.strip():

            results.append((name, 0))

            continue

        resume_embedding = model.encode(text, convert_to_tensor=True)

        semantic_score = util.cos_sim(jd_embedding, resume_embedding).item()

        keyword_matches = sum(1 for k in jd_keywords if k in text.lower())

        keyword_score = keyword_matches / len(jd_keywords) if jd_keywords else 0

        total_score = (semantic_score * 0.7) + (keyword_score * 0.3)

        results.append((name, round(total_score * 100, 2)))

    return sorted(results, key=lambda x: x[1], reverse=True)

# -----------------------------
# UI HEADER
# -----------------------------

st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📄 AI Resume Screener & JD Matcher</div>", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.header("Upload Resumes")

    resume_files = st.file_uploader(
        "Upload Resume Files",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

# -----------------------------
# MAIN LAYOUT
# -----------------------------

col1,col2 = st.columns([1,2])

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

            with st.spinner("Analyzing resumes..."):

                resume_texts = []

                for uploaded_file in resume_files:

                    ext = os.path.splitext(uploaded_file.name)[1]

                    temp_file = uploaded_file.name

                    with open(temp_file,"wb") as f:

                        f.write(uploaded_file.read())

                    if ext == ".pdf":

                        text = extract_text_from_pdf(temp_file)

                    else:

                        text = extract_text_from_docx(temp_file)

                    resume_texts.append((uploaded_file.name,text))

                    os.remove(temp_file)

                scores = compute_similarity(resume_texts,jd_input)

                st.success("Matching Complete")

                df = pd.DataFrame(scores, columns=["Candidate Name","Match Score (%)"])

                df.index = df.index + 1

                st.dataframe(df,use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results",
                    csv,
                    "resume_results.csv",
                    "text/csv"
                )

                st.markdown("---")

                # Save resume texts to session
                st.session_state["resume_texts"] = resume_texts
                st.session_state["scores"] = scores

# -----------------------------
# BUTTON ACTIONS
# -----------------------------

if "scores" in st.session_state:

    scores = st.session_state["scores"]
    resume_texts = st.session_state["resume_texts"]

    for i,(name,score) in enumerate(scores,1):

        st.markdown(f"### {i}. {name}")
        st.write(f"Match Score: {score}%")

        resume_text = next(t for f,t in resume_texts if f==name)

        colA,colB,colC = st.columns(3)

        # Summary
        with colA:

            if st.button("AI Summary", key=f"summary_{i}"):

                st.session_state[f"summary_{i}"] = generate_candidate_summary(resume_text)

        if f"summary_{i}" in st.session_state:

            st.write(st.session_state[f"summary_{i}"])

        # Skill Gap
        with colB:

            if st.button("Skill Gap", key=f"gap_{i}"):

                st.session_state[f"gap_{i}"] = skill_gap_analysis(jd_input,resume_text)

        if f"gap_{i}" in st.session_state:

            st.write(st.session_state[f"gap_{i}"])

        # Interview Questions
        with colC:

            if st.button("Interview Questions", key=f"question_{i}"):

                st.session_state[f"question_{i}"] = generate_interview_questions(jd_input,resume_text)

        if f"question_{i}" in st.session_state:

            st.write(st.session_state[f"question_{i}"])
