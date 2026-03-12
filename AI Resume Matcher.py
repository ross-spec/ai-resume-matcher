import streamlit as st
import base64
import pandas as pd
from pathlib import Path

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------
# BACKGROUND IMAGE FUNCTION
# ---------------------------------------------

def set_background(image_file):

    image_path = Path(image_file)

    if image_path.exists():

        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        css = f"""
        <style>

        [data-testid="stAppViewContainer"] {{
            background:
            linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
            url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        [data-testid="stHeader"] {{
            background: rgba(0,0,0,0);
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
            backdrop-filter: blur(8px);
        }}

        .result-card {{
            background:white;
            padding:20px;
            border-radius:10px;
            margin-bottom:20px;
            color:black;
        }}

        .stButton>button {{
            background-color:#2d8cff;
            color:white;
            border:none;
            border-radius:6px;
            padding:8px 18px;
        }}

        .stButton>button:hover {{
            background-color:#1b6ee0;
        }}

        </style>
        """

        st.markdown(css, unsafe_allow_html=True)

    else:
        st.error("Background image not found")

# APPLY BACKGROUND
set_background("background.png")

# ---------------------------------------------
# HERO SECTION
# ---------------------------------------------

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

# ---------------------------------------------
# SIDEBAR
# ---------------------------------------------

with st.sidebar:

    st.header("📂 Upload Resumes")

    resume_files = st.file_uploader(
        "Upload PDF or DOCX resumes",
        type=["pdf","docx"],
        accept_multiple_files=True
    )

    st.markdown("---")

    st.header("📋 Job Description")

    jd_input = st.text_area(
        "Paste job description",
        height=200
    )

    analyze = st.button("Analyze Candidates")

# ---------------------------------------------
# MOCK LOGIC (Replace with your AI logic)
# ---------------------------------------------

def compute_similarity(resumes, jd):

    results=[]

    for i,file in enumerate(resumes):

        score = 75 + (i*5)

        experience = 2 + i

        results.append((file.name,score,experience))

    return results


def generate_interview_questions(jd):

    return """
1. Explain the star schema used in Power BI data modeling.

2. What is the difference between calculated columns and measures?

3. How do you optimize Power BI reports for large datasets?

4. Explain how row-level security works in Power BI.

5. How would you design a data model for sales performance analytics?

6. How do you handle incremental refresh in Power BI?

7. Describe a challenging BI dashboard you have built.
"""

# ---------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------

if analyze:

    if not resume_files or not jd_input:

        st.warning("Please upload resumes and provide job description")

    else:

        results = compute_similarity(resume_files,jd_input)

        # ---------------------------
        # INTERVIEW QUESTIONS
        # ---------------------------

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("📌 Suggested Interview Questions")

        questions = generate_interview_questions(jd_input)

        st.write(questions)

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------
        # RANKING TABLE
        # ---------------------------

        df = pd.DataFrame(
            [(i+1,r[0],r[1]) for i,r in enumerate(results)],
            columns=["Rank","Candidate Name","Match Score %"]
        )

        st.markdown('<div class="section-card">', unsafe_allow_html=True)

        st.subheader("🏆 Candidate Ranking")

        st.dataframe(df,use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ---------------------------
        # CANDIDATE ANALYSIS
        # ---------------------------

        st.subheader("Candidate Analysis")

        for i,(name,score,experience) in enumerate(results,1):

            st.markdown('<div class="result-card">', unsafe_allow_html=True)

            st.markdown(f"### {i}. {name}")

            st.write(f"Match Score: **{score}%**")

            st.write(f"Estimated Experience: **{experience} years**")

            st.markdown("**Candidate Summary**")

            st.write(
            "Experienced Power BI developer with strong expertise in SQL, "
            "DAX calculations, dashboard development and data modeling."
            )

            st.markdown("**Skill Gap**")

            st.write(
            "Azure Data Factory, Databricks, Advanced Data Engineering."
            )

            st.markdown('</div>', unsafe_allow_html=True)
