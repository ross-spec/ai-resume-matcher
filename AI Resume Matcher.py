import os
import re
import requests
import PyPDF2
import docx2txt
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------
# OPENROUTER CONFIG
# -----------------------------------

try:
    OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
except:
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "openai/gpt-4o-mini"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------
# AI CALL FUNCTION
# -----------------------------------

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
        return f"API Error: {response.text}"

    data = response.json()

    if "choices" not in data:
        return f"API Error: {data}"

    return data["choices"][0]["message"]["content"]

# -----------------------------------
# FILE TEXT EXTRACTION
# -----------------------------------

def extract_text_from_pdf(path):

    text = ""

    with open(path,"rb") as file:

        reader = PyPDF2.PdfReader(file)

        for page in reader.pages:

            txt = page.extract_text()

            if txt:
                text += txt

    return text


def extract_text_from_docx(path):

    return docx2txt.process(path)

# -----------------------------------
# AI EXTRACTION FUNCTIONS
# -----------------------------------

def extract_required_skills(jd_text):

    prompt = f"""
    Extract required technical skills from this job description.
    Return comma separated skills.

    Job Description:
    {jd_text[:2000]}
    """

    return call_ai(prompt)


def extract_candidate_skills(resume_text):

    prompt = f"""
    Extract technical skills from this resume.
    Return comma separated skills.

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)


def extract_experience(resume_text):

    prompt = f"""
    Estimate total years of experience from this resume.
    Return only a number.

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)

# -----------------------------------
# CANDIDATE SUMMARY
# -----------------------------------

def generate_candidate_summary(resume_text):

    prompt = f"""
    Analyze the resume and provide:

    Candidate Role
    Years of Experience
    Key Skills
    Strengths

    Resume:
    {resume_text[:2000]}
    """

    return call_ai(prompt)

# -----------------------------------
# SKILL GAP ANALYSIS
# -----------------------------------

def skill_gap_analysis(jd_text,resume_text):

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

# -----------------------------------
# INTERVIEW QUESTIONS (JD BASED)
# -----------------------------------

def generate_interview_questions_for_jd(jd_text):

    prompt = f"""
    Generate 7 interview questions for this job description.

    Job Description:
    {jd_text}

    Questions should include:
    - technical questions
    - real scenario questions
    - problem solving questions
    """

    return call_ai(prompt)

# -----------------------------------
# ADVANCED ATS SCORING
# -----------------------------------

def compute_similarity(resume_texts,jd_text):

    jd_embedding = embedding_model.encode(jd_text,convert_to_tensor=True)

    required_skills = extract_required_skills(jd_text)
    required_skills_list = [s.strip().lower() for s in required_skills.split(",")]

    results = []

    for name,text in resume_texts:

        resume_embedding = embedding_model.encode(text,convert_to_tensor=True)

        semantic_score = util.cos_sim(jd_embedding,resume_embedding).item()

        candidate_skills = extract_candidate_skills(text)
        candidate_skills_list = [s.strip().lower() for s in candidate_skills.split(",")]

        skill_matches = len(set(required_skills_list) & set(candidate_skills_list))
        skill_score = skill_matches / len(required_skills_list) if required_skills_list else 0

        experience = extract_experience(text)

        try:
            experience = float(experience)
        except:
            experience = 0

        experience_score = min(experience/5,1)

        final_score = (
            semantic_score*0.4 +
            skill_score*0.4 +
            experience_score*0.2
        )

        results.append((name,text,round(final_score*100,2),experience))

    return sorted(results,key=lambda x:x[2],reverse=True)

# -----------------------------------
# CUSTOM CSS DESIGN
# -----------------------------------

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1551288049-bebda4e38f71");
    background-size: cover;
    background-attachment: fixed;
}

.main-title{
text-align:center;
font-size:48px;
font-weight:700;
color:white;
margin-bottom:30px;
}

.section-card{
background-color:rgba(0,0,0,0.7);
padding:25px;
border-radius:15px;
margin-bottom:20px;
color:white;
}

.result-card{
background-color:white;
padding:20px;
border-radius:10px;
margin-bottom:15px;
}

</style>
""",unsafe_allow_html=True)

st.markdown("<div class='main-title'>AI Resume Screener & Candidate Ranking</div>",unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------

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

# -----------------------------------
# MAIN APPLICATION
# -----------------------------------

if analyze:

    if not resume_files or not jd_input:

        st.warning("Please upload resumes and provide job description")

    else:

        with st.spinner("Analyzing resumes..."):

            resume_texts=[]

            for uploaded_file in resume_files:

                ext=os.path.splitext(uploaded_file.name)[1]

                temp_file=uploaded_file.name

                with open(temp_file,"wb") as f:
                    f.write(uploaded_file.read())

                if ext==".pdf":
                    text=extract_text_from_pdf(temp_file)
                else:
                    text=extract_text_from_docx(temp_file)

                resume_texts.append((uploaded_file.name,text))

                os.remove(temp_file)

            results=compute_similarity(resume_texts,jd_input)

        # Interview questions
        st.markdown("<div class='section-card'>",unsafe_allow_html=True)
        st.subheader("Interview Questions")
        questions = generate_interview_questions_for_jd(jd_input)
        st.write(questions)
        st.markdown("</div>",unsafe_allow_html=True)

        # Ranking table
        df=pd.DataFrame(
            [(i+1,r[0],r[2]) for i,r in enumerate(results)],
            columns=["Rank","Candidate Name","Match Score %"]
        )

        st.markdown("<div class='section-card'>",unsafe_allow_html=True)
        st.subheader("Candidate Ranking")
        st.dataframe(df,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

        # Candidate analysis
        st.subheader("Candidate Analysis")

        for rank,(name,text,score,experience) in enumerate(results,1):

            st.markdown("<div class='result-card'>",unsafe_allow_html=True)

            st.markdown(f"### {rank}. {name}")
            st.write(f"Match Score: **{score}%**")
            st.write(f"Estimated Experience: **{experience} years**")

            summary=generate_candidate_summary(text)
            gap=skill_gap_analysis(jd_input,text)

            st.markdown("**Candidate Summary**")
            st.write(summary)

            st.markdown("**Skill Gap**")
            st.write(gap)

            st.markdown("</div>",unsafe_allow_html=True)
