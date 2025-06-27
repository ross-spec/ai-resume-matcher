import os
import re

try:
    import PyPDF2
    import docx2txt
    import pandas as pd
    from sentence_transformers import SentenceTransformer, util
    import streamlit as st
    from streamlit.components.v1 import html
    dependencies_available = True
except ModuleNotFoundError as e:
    missing = str(e).split("No module named")[-1].strip(" '")
    print(f"\n🔧 Required package is missing: {missing}\nPlease install it using 'pip install {missing}' before running the application.\n")
    dependencies_available = False

if dependencies_available:
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
        except:
            return ""

    def extract_text_from_docx(docx_path):
        try:
            return docx2txt.process(docx_path)
        except:
            return ""

    def extract_keywords(text, top_n=10):
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        freq = pd.Series(words).value_counts()
        return freq.head(top_n).index.tolist()

    def compute_similarity(resume_texts, jd_text, filename_weight=0.2):
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)
        jd_keywords = extract_keywords(jd_text)
        results = []

        for name, text in resume_texts:
            if not text.strip():
                results.append((name, 0))
                continue

            resume_embedding = model.encode(text, convert_to_tensor=True)
            content_score = util.cos_sim(jd_embedding, resume_embedding).item()

            # Filename match scoring
            name_clean = os.path.splitext(name)[0].lower().replace("_", " ").replace("-", " ")
            matches = sum(1 for keyword in jd_keywords if keyword in name_clean)
            filename_score = matches / len(jd_keywords) if jd_keywords else 0

            total_score = (1 - filename_weight) * content_score + filename_weight * filename_score
            results.append((name, round(total_score * 100, 2)))

        return sorted(results, key=lambda x: x[1], reverse=True)

    # Page UI setup
    st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
    st.markdown("""
        <style>
            .main {
                background-color: #f8f9fa;
                padding: 2rem;
                border-radius: 10px;
            }
            .title-style {
                background-color: #4B8BBE;
                color: white;
                padding: 1rem;
                border-radius: 10px;
                text-align: center;
                font-size: 32px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='title-style'>📄 AI Resume Screener & JD Matcher</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("📤 Upload Resumes")
        resume_files = st.file_uploader("Upload Resume PDFs or DOCX", type=["pdf", "docx"], accept_multiple_files=True)
        st.markdown("<hr style='border-top: 1px solid #bbb;'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### 📝 Job Description")
        jd_input = st.text_area("Paste the Job Description", height=250, placeholder="Enter job role, skills, and expectations...")

    with col2:
        st.markdown("### 📎 Resume Matching Results")

        if st.button("🔍 Match Resumes"):
            if not resume_files or not jd_input.strip():
                st.warning("⚠️ Please upload resumes and provide a job description.")
            else:
                st.info("⏳ Processing...")
                resume_texts = []

                for uploaded_file in resume_files:
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    temp_filename = uploaded_file.name
                    bytes_data = uploaded_file.read()

                    with open(temp_filename, "wb") as f:
                        f.write(bytes_data)

                    if file_ext == ".pdf":
                        text = extract_text_from_pdf(temp_filename)
                    elif file_ext == ".docx":
                        text = extract_text_from_docx(temp_filename)
                    else:
                        text = ""

                    resume_texts.append((uploaded_file.name, text))
                    os.remove(temp_filename)

                scores = compute_similarity(resume_texts, jd_input)
                st.success("✅ Matching Complete!")

                df_scores = pd.DataFrame(scores, columns=["Candidate Name", "Match Score (%)"])
                st.dataframe(df_scores, use_container_width=True)

                csv = df_scores.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name='resume_matching_results.csv',
                    mime='text/csv'
                )

                st.markdown("---")
                for i, (name, score) in enumerate(scores, 1):
                    st.markdown(f"**{i}. {name}** — 🎯 Match Score: **{score}%**")
