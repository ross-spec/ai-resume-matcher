import streamlit as st
import base64
import os
import pandas as pd

# -----------------------------------------
# PAGE CONFIG
# -----------------------------------------

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="📄",
    layout="wide"
)

# -----------------------------------------
# BACKGROUND IMAGE FUNCTION
# -----------------------------------------

def set_background(image_file):

    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    page_bg = f"""
    <style>

    [data-testid="stAppViewContainer"] {{
        background:
        linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
        url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    h1,h2,h3,h4,h5,h6 {{
        color: #ffffff;
    }}

    p, span, div {{
        color: #e6e6e6;
    }}

    .main-title {{
        text-align:center;
        font-size:48px;
        font-weight:700;
        color:white;
        margin-bottom:20px;
    }}

    .sub-title {{
        text-align:center;
        font-size:20px;
        color:#d1d1d1;
        margin-bottom:40px;
    }}

    .section-card {{
        background: rgba(0,0,0,0.65);
        padding:25px;
        border-radius:12px;
        margin-bottom:25px;
        backdrop-filter: blur(6px);
    }}

    .result-card {{
        background: rgba(255,255,255,0.95);
        padding:20px;
        border-radius:10px;
        margin-bottom:20px;
        color:black;
    }}

    .stButton>button {{
        background-color:#1f77ff;
        color:white;
        border-radius:6px;
        padding:8px 20px;
        border:none;
    }}

    .stButton>button:hover {{
        background-color:#005ce6;
        color:white;
    }}

    </style>
    """

    st.markdown(page_bg, unsafe_allow_html=True)

# Apply background
set_background("background.png")

# -----------------------------------------
# HERO HEADER
# -----------------------------------------

st.markdown(
"""
<div class='main-title'>
AI Resume Screener & Candidate Ranking
</div>
<div class='sub-title'>
Analyze resumes, detect skills, and rank candidates using AI
</div>
""",
unsafe_allow_html=True
)

# -----------------------------------------
# SIDEBAR
# -----------------------------------------

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
        "Paste job description here",
        height=200
    )

    analyze = st.button("Analyze Candidates")

# -----------------------------------------
# MOCK FUNCTIONS (Replace with your AI logic)
# -----------------------------------------

def compute_similarity(resumes, jd):

    data = []

    for i,file in enumerate(resumes):

        data.append(
            (file.name, 75 + i*5, 2+i)
        )

    return data


def generate_interview_questions(jd):

    return """
1. Explain star schema in Power BI  
2. Difference between calculated column and measure  
3. How do you optimize large Power BI dashboards?  
4. Explain row level security  
5. Handling large datasets in Power BI
"""


# -----------------------------------------
# MAIN APP
# -----------------------------------------

if analyze:

    if not resume_files or not jd_input:

        st.warning("Please upload resumes and provide job description")

    else:

        results = compute_similarity(resume_files, jd_input)

        # -----------------------------------------
        # INTERVIEW QUESTIONS
        # -----------------------------------------

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        st.subheader("📌 Suggested Interview Questions")

        questions = generate_interview_questions(jd_input)

        st.write(questions)

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------------------
        # RANKING TABLE
        # -----------------------------------------

        df = pd.DataFrame(
            [(i+1,r[0],r[1]) for i,r in enumerate(results)],
            columns=["Rank","Candidate Name","Match Score %"]
        )

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        st.subheader("🏆 Candidate Ranking")

        st.dataframe(df,use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # -----------------------------------------
        # CANDIDATE ANALYSIS
        # -----------------------------------------

        st.subheader("Candidate Analysis")

        for i,(name,score,exp) in enumerate(results,1):

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)

            st.markdown(f"### {i}. {name}")

            st.write(f"Match Score: **{score}%**")

            st.write(f"Estimated Experience: **{exp} years**")

            st.write("Candidate Summary")

            st.write("Strong Power BI developer with experience in SQL, DAX and data modeling.")

            st.write("Skill Gap")

            st.write("Azure Data Factory, Databricks")

            st.markdown("</div>", unsafe_allow_html=True)
