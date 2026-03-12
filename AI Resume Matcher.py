import os
import re
import requests
import PyPDF2
import docx2txt
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ---------------------------------
# PAGE CONFIG
# ---------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------
# OPENROUTER CONFIG
# ---------------------------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

MODEL = "mistralai/mistral-7b-instruct:free"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------
# AI FUNCTION
# ---------------------------------

def call_ai(prompt):

    try:

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = requests.post(API_URL, headers=headers, json=data)

        if response.status_code != 200:
            return f"API Error: {response.text}"

        result = response.json()

        if "choices" in result:
            return result["choices"][0]["message"]["content"]
        else:
            return f"AI Error: {result}"

    except Exception as e:
        return f"AI Exception: {str(e)}"


# ---------------------------------
# TEXT EXTRACTION
# ---------------------------------

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


# ---------------------------------
# KEYWORD EXTRACTION
# ---------------------------------

def extract_keywords(text):

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    freq = pd.Series(words).value_counts()

    return list(freq.head(20).index)


# ---------------------------------
# AI ANALYSIS FUNCTIONS
# ---------------------------------

def generate_candidate_summary(resume_text):

    prompt = f"""
    Analyze this resume.

    Provide:
    Candidate Role
    Years of Experience
    Key Skills
    Strengths

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)


def skill_gap_analysis(jd_text, resume_text):

    prompt = f"""
    Compare the job description and resume.

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


def generate_interview_questions(jd_text, resume_text):

    prompt = f"""
    Generate 5 interview questions based on the resume and job description.

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}
    """

    return call_ai(prompt)


# ---------------------------------
# RESUME MATCHING
# ---------------------------------

def compute_similarity(resume_texts, jd_text):

    jd_embedding = model.encode(jd_text, convert_to_tensor=True)

    jd_keywords = extract_keywords(jd_text)

    results = []

    for name, text in resume_texts:

        resume_embedding = model.encode(text, convert_to_tensor=True)

        semantic_score = util.cos_sim(jd_embedding, resume_embedding).item()

        keyword_matches = sum(1 for k in jd_keywords if k in text.lower())

        keyword_score = keyword_matches / len(jd_keywords) if jd_keywords else 0

        total_score = (semantic_score * 0.7) + (keyword_score * 0.3)

        results.append((name, text, round(total_score * 100, 2)))

    return sorted(results, key=lambda x: x[2], reverse=True)


# ---------------------------------
# UI HEADER
# ---------------------------------

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


# ---------------------------------
# SIDEBAR
# ---------------------------------

with st.sidebar:

    st.header("Upload Resumes")

    resume_files = st.file_uploader(
        "Upload Resume Files",
        type=["pdf","docx"],
        accept_multiple_files=True
    )


# ---------------------------------
# MAIN LAYOUT
# ---------------------------------

col1, col2 = st.columns([1,2])

with col1:

    st.subheader("Job Description")

    jd_input = st.text_area(
        "Paste Job Description",
        height=300
    )


with col2:

    st.subheader("Resume Ranking")

    if st.button("Analyze Resumes"):

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

                results = compute_similarity(resume_texts,jd_input)

                df = pd.DataFrame(
                    [(i+1,r[0],r[2]) for i,r in enumerate(results)],
                    columns=["Rank","Candidate Name","Match Score %"]
                )

                st.dataframe(df,use_container_width=True)

                st.download_button(
                    "Download Results",
                    df.to_csv(index=False),
                    "resume_results.csv"
                )

                st.markdown("---")

                st.header("Candidate Analysis")

                for rank,(name,text,score) in enumerate(results,1):

                    st.subheader(f"{rank}. {name}")

                    st.write(f"Match Score: {score}%")

                    with st.spinner("Running AI analysis..."):

                        summary = generate_candidate_summary(text)
                        gap = skill_gap_analysis(jd_input,text)
                        questions = generate_interview_questions(jd_input,text)

                    st.markdown("### Candidate Summary")
                    st.info(summary)

                    st.markdown("### Skill Gap")
                    st.warning(gap)

                    st.markdown("### Interview Questions")
                    st.success(questions)

                    st.markdown("---")
