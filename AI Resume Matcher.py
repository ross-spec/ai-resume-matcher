import streamlit as st
import os
import requests
import pandas as pd
import PyPDF2
import docx2txt
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="HireAI Resume Screener",
    page_icon="📄",
    layout="wide"
)

st.title("HireAI – Smart Resume Screening")

# ------------------------------------------------
# SESSION STATE
# ------------------------------------------------

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ------------------------------------------------
# CSS (WHITE BACKGROUND OPTIMIZED)
# ------------------------------------------------

st.markdown("""
<style>

header {display:none;}

.block-container{
padding-top:1rem;
max-width:100%;
}

/* TEXT */

p,label{
color:#333333;
font-size:15px;
}

h1,h2,h3{
color:#111111;
}

/* RIGHT PANEL */

.panel{
background:#f6f8fb;
padding:25px;
border-radius:12px;
border:1px solid #e5e7eb;
}

/* SECTION HEADERS */

.section-header{
background:#eaf3ff;
padding:10px;
border-radius:8px;
color:#0f172a;
font-weight:700;
}

/* RESULT CARDS */

.result-card{
background:white;
padding:20px;
border-radius:10px;
border:1px solid #e5e7eb;
margin-bottom:15px;
color:#111111;
box-shadow:0 2px 6px rgba(0,0,0,0.05);
}

/* BUTTON */

.stButton > button {
background:#2563eb;
color:white;
font-weight:700;
border-radius:8px;
padding:10px 16px;
}

.stButton > button:hover {
background:#1d4ed8;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] button{
color:black !important;
font-weight:600 !important;
}

</style>
""",unsafe_allow_html=True)

# ------------------------------------------------
# AI CONFIG
# ------------------------------------------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

MODEL="openai/gpt-4o-mini"

API_URL="https://openrouter.ai/api/v1/chat/completions"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------
# AI CALL
# ------------------------------------------------

def call_ai(prompt):

    headers={
        "Authorization":f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":"application/json"
    }

    payload={
        "model":MODEL,
        "messages":[{"role":"user","content":prompt}],
        "temperature":0.2
    }

    r=requests.post(API_URL,headers=headers,json=payload)

    if r.status_code!=200:
        return "AI Error"

    return r.json()["choices"][0]["message"]["content"]

# ------------------------------------------------
# FILE TEXT EXTRACTION
# ------------------------------------------------

def extract_text_from_pdf(file):

    reader=PyPDF2.PdfReader(file)

    text=""

    for page in reader.pages:
        text+=page.extract_text()

    return text


def extract_text_from_docx(file):

    return docx2txt.process(file)

# ------------------------------------------------
# SKILL EXTRACTION
# ------------------------------------------------

def extract_skills(resume_text):

    prompt=f"""
Extract top professional skills from this resume.

Return a simple bullet list.

Resume:
{resume_text[:2000]}
"""

    return call_ai(prompt)

# ------------------------------------------------
# RESUME MATCHING
# ------------------------------------------------

def compute_similarity(resume_texts,jd_text):

    jd_embedding=embedding_model.encode(jd_text,convert_to_tensor=True)

    results=[]

    for name,text in resume_texts:

        resume_embedding=embedding_model.encode(text,convert_to_tensor=True)

        score=util.cos_sim(jd_embedding,resume_embedding).item()

        score=round(score*100,2)

        experience=3

        results.append((name,text,score,experience))

    return sorted(results,key=lambda x:x[2],reverse=True)

# ------------------------------------------------
# INTERVIEW QUESTIONS
# ------------------------------------------------

def generate_questions(jd):

    prompt=f"""
Generate 10 interview questions strictly based on this job description.

Job Description:
{jd}

Return numbered questions only.
"""

    return call_ai(prompt)

# ------------------------------------------------
# AI RECOMMENDATION
# ------------------------------------------------

def generate_recommendation(jd,resume,score):

    prompt=f"""
Analyze candidate suitability.

Match Score: {score}

Job Description:
{jd}

Resume:
{resume[:2000]}

Provide:

Strengths
Missing skills
Hiring recommendation
"""

    return call_ai(prompt)

# ------------------------------------------------
# LAYOUT
# ------------------------------------------------

main,panel=st.columns([3,1])

# ------------------------------------------------
# RIGHT PANEL
# ------------------------------------------------

with panel:

    st.markdown('<div class="panel">',unsafe_allow_html=True)

    st.markdown('<div class="section-header">Upload Resumes</div>',unsafe_allow_html=True)

    resume_files=st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.markdown('<div class="section-header">Job Description</div>',unsafe_allow_html=True)

    jd_input=st.text_area("Paste job description",height=200)

    analyze=st.button("Analyze Candidates")

    st.markdown('</div>',unsafe_allow_html=True)

# ------------------------------------------------
# RESULTS
# ------------------------------------------------

with main:

    if analyze:

        st.session_state.analysis_done=True

        if not resume_files or not jd_input:

            st.warning("Upload resumes and provide job description")

        else:

            with st.spinner("Analyzing resumes with AI..."):

                resume_texts=[]

                for file in resume_files:

                    ext=os.path.splitext(file.name)[1]

                    if ext==".pdf":
                        text=extract_text_from_pdf(file)
                    else:
                        text=extract_text_from_docx(file)

                    resume_texts.append((file.name,text))

                results=compute_similarity(resume_texts,jd_input)

                questions=generate_questions(jd_input)

            df=pd.DataFrame(
                [(i+1,r[0],r[2]) for i,r in enumerate(results)],
                columns=["Rank","Candidate Name","Match Score"]
            )

            st.subheader("Candidate Ranking")

            st.dataframe(df,use_container_width=True)

            csv=df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Ranking CSV",
                csv,
                "candidate_ranking.csv",
                "text/csv"
            )

            st.subheader("Interview Questions")

            st.write(questions)

            st.subheader("Candidate Analysis")

            for rank,(name,text,score,exp) in enumerate(results,1):

                st.markdown('<div class="result-card">',unsafe_allow_html=True)

                st.markdown(f"### {rank}. {name}")

                st.write(f"Match Score: {score}%")

                st.progress(score/100)

                st.write(f"Estimated Experience: {exp} years")

                skills=extract_skills(text)

                st.write("Top Skills:")

                st.write(skills)

                recommendation=generate_recommendation(jd_input,text,score)

                st.write("AI Recommendation")

                st.write(recommendation)

                st.markdown("</div>",unsafe_allow_html=True)
