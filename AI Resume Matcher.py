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

# Local embedding model
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
        "messages": [
            {"role": "user", "content": prompt}
        ],
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
    Return only comma separated skills.

    Job Description:
    {jd_text[:2000]}
    """

    return call_ai(prompt)


def extract_candidate_skills(resume_text):

    prompt = f"""
    Extract technical skills from this resume.
    Return only comma separated skills.

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
    Analyze this resume and provide:

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
# EXPERIENCE BASED INTERVIEW QUESTIONS
# -----------------------------------

def generate_interview_questions(jd_text,resume_text,experience):

    prompt = f"""
    Generate interview questions based on candidate experience.

    Candidate Experience: {experience} years

    If experience <2 years → beginner
    2-5 years → intermediate
    >5 years → advanced

    Job Description:
    {jd_text[:1500]}

    Resume:
    {resume_text[:1500]}

    Generate 5 interview questions.
    """

    return call_ai(prompt)

# -----------------------------------
# KEYWORD EXTRACTION
# -----------------------------------

def extract_keywords(text):

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())

    freq = pd.Series(words).value_counts()

    return list(freq.head(20).index)

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
# UI HEADER
# -----------------------------------

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
""",unsafe_allow_html=True)

st.markdown("<div class='title'>📄 AI Resume Screener & JD Matcher</div>",unsafe_allow_html=True)

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.header("Upload Resumes")

    resume_files = st.file_uploader(
        "Upload Resume Files",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

# -----------------------------------
# MAIN LAYOUT
# -----------------------------------

col1,col2 = st.columns([1,2])

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

            st.warning("Upload resumes and provide JD")

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

                df=pd.DataFrame(
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

                for rank,(name,text,score,experience) in enumerate(results,1):

                    st.subheader(f"{rank}. {name}")

                    st.write(f"Match Score: {score}%")
                    st.write(f"Estimated Experience: {experience} years")

                    with st.spinner("Running AI analysis..."):

                        summary=generate_candidate_summary(text)
                        gap=skill_gap_analysis(jd_input,text)
                        questions=generate_interview_questions(jd_input,text,experience)

                    st.markdown("### Candidate Summary")
                    st.info(summary)

                    st.markdown("### Skill Gap")
                    st.warning(gap)

                    st.markdown("### Interview Questions")
                    st.success(questions)

                    st.markdown("---")
