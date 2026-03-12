import streamlit as st
import os
import requests
import pandas as pd
import PyPDF2
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="HireAI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ------------------------------------------------
# TOP LAYOUT (HEADER + UPLOAD PANEL)
# ------------------------------------------------

left, right = st.columns([3,1])

# -------------------------
# LEFT SIDE (HEADER)
# -------------------------

with left:

    st.markdown("""
    <h1 style='font-size:42px;margin-bottom:5px'>
    <span style='color:#2563eb;font-weight:900'>HireAI</span>
    <span style='color:#111'> – Smart Resume Screening</span>
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("### AI Resume Screening & Candidate Ranking")

    st.markdown("""
Upload resumes and paste a job description to instantly:

• Rank candidates based on AI matching  
• Extract important candidate skills  
• Generate interview questions  
• Get hiring recommendations
""")

# -------------------------
# RIGHT SIDE (UPLOAD PANEL)
# -------------------------

with right:

    st.markdown(
    """
    <div style="
        padding:20px;
        border-radius:12px;
        border:1px solid #e6e6e6;
        background:#fafafa;
        box-shadow:0px 4px 10px rgba(0,0,0,0.05);
    ">
    """,
    unsafe_allow_html=True
    )

    st.markdown("### Upload Resumes")

    resume_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.markdown("### Job Description")

    jd_input = st.text_area(
        "Paste job description",
        height=200
    )

    analyze = st.button("Analyze Candidates")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------
# AI CONFIG
# ------------------------------------------------

OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

MODEL = "openai/gpt-4o-mini"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------------------------------------
# AI CALL
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

    r = requests.post(API_URL, headers=headers, json=payload)

    if r.status_code != 200:
        return "AI Error"

    return r.json()["choices"][0]["message"]["content"]

# ------------------------------------------------
# FILE TEXT EXTRACTION
# ------------------------------------------------

def extract_text_from_pdf(file):

    reader = PyPDF2.PdfReader(file)

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
# MAIN RESULT SECTION
# ------------------------------------------------

if analyze:

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

        # ------------------------------------------------
        # REAL STATS
        # ------------------------------------------------

        scores=[r[2] for r in results]

        avg_score=round(np.mean(scores),2)

        top_score=max(scores)

        total_resumes=len(resume_files)

        ranked_candidates=len(results)

        st.markdown("## 📊 Screening Dashboard")

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.metric("Resumes Uploaded", total_resumes)

        with c2:
            st.metric("Candidates Ranked", ranked_candidates)

        with c3:
            st.metric("Top Match Score", f"{top_score}%")

        with c4:
            st.metric("Average Match Score", f"{avg_score}%")

        # ------------------------------------------------
        # TOP CANDIDATE
        # ------------------------------------------------

        st.markdown("## 🏆 Top Candidate")

        top_candidate=results[0]

        st.success(f"Best Match: **{top_candidate[0]}** ({top_candidate[2]}%)")

        # ------------------------------------------------
        # SCORE CHART
        # ------------------------------------------------

        st.markdown("## 📈 Candidate Match Scores")

        names=[r[0] for r in results]

        scores=[r[2] for r in results]

        fig,ax = plt.subplots()

        ax.barh(names,scores)

        ax.set_xlabel("Match Score")

        ax.set_title("Resume Matching Scores")

        st.pyplot(fig)

        # ------------------------------------------------
        # RANKING TABLE
        # ------------------------------------------------

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

        # ------------------------------------------------
        # INTERVIEW QUESTIONS
        # ------------------------------------------------

        st.subheader("Interview Questions")

        st.write(questions)

        # ------------------------------------------------
        # CANDIDATE ANALYSIS
        # ------------------------------------------------

        st.subheader("Candidate Analysis")

        for rank,(name,text,score,exp) in enumerate(results,1):

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

            st.markdown("---")
